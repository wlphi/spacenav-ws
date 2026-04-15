import asyncio
import json
import logging
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, NamedTuple

# Capture X11 display info at import time (while the launching shell's env is available).
_X11_ENV: dict[str, str] = {}
for _k in ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "DBUS_SESSION_BUS_ADDRESS"):
    if _k in os.environ:
        _X11_ENV[_k] = os.environ[_k]

import numpy as np  # noqa: E402

from spacenav_ws.buttons import (  # noqa: E402
    CTRL_BUTTON_ID,
    SHIFT_BUTTON_ID,
    get_button_map,
    get_context_hotkey_map,
    get_ctrl_map,
    get_hotkeys,
    get_shift_map,
    load_config,
)
from spacenav_ws.display import EnterpriseDisplay, set_lock_led  # noqa: E402
from spacenav_ws.spacenav import ButtonEvent, MotionEvent, from_message  # noqa: E402
from spacenav_ws.views import get_view_matrix  # noqa: E402
from spacenav_ws.wamp import Call, CallResult, Prefix, Subscribe, WampSession  # noqa: E402

# Single display instance shared across all connections — USB interface is
# claimed once at startup and kept open for the lifetime of the process.
_display = EnterpriseDisplay()

# Saved custom views are persisted here so they survive restarts
_SAVED_VIEWS_PATH = Path.home() / ".config" / "spacenav-ws" / "saved_views.json"


def _load_saved_views() -> dict[int, dict]:
    if _SAVED_VIEWS_PATH.exists():
        try:
            raw = json.loads(_SAVED_VIEWS_PATH.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:
            logging.warning("Could not load saved views from %s", _SAVED_VIEWS_PATH)
    return {}


def _persist_saved_views(views: dict[int, dict]):
    try:
        _SAVED_VIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SAVED_VIEWS_PATH.write_text(json.dumps({str(k): v for k, v in views.items()}, indent=2))
    except Exception:
        logging.warning("Could not persist saved views to %s", _SAVED_VIEWS_PATH)


def _rotation_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation formula — returns a 3×3 rotation matrix.

    More numerically correct than decomposed Euler angles: simultaneous
    multi-axis input is handled as a single rotation, not three sequential ones.
    """
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle) < 1e-12:
        return np.eye(3, dtype=np.float64)
    u = axis / norm
    x, y, z = u
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    s, c = np.sin(angle), np.cos(angle)
    return np.eye(3) + s * skew + (1.0 - c) * (skew @ skew)


class CursorPivotResult(NamedTuple):
    """Result of compute_cursor_pivot — pure, no side effects."""
    pivot: np.ndarray        # world-space pivot point
    dist: float              # cursor distance from viewport centre in camera space
    viewport_half: float     # half of the larger viewport dimension
    used_cursor: bool        # True if cursor NDC was used (not model-centre fallback)
    source: str              # diagnostic tag: "cursor", "bbox_estimate", "model_center_oob"


def compute_cursor_pivot(
    nx: float,
    ny: float,
    model_extents: list,
    curr_affine: np.ndarray,
    extents: list | None,
    perspective: bool = False,
    frustum: list | None = None,
) -> CursorPivotResult:
    """Project the screen-space cursor to model-centre depth to get a pivot point.

    Pure function — no I/O, no side effects.  Returns a CursorPivotResult.

    Viewport bounds are resolved in priority order:
      1. view.frustum (perspective only — scaled to model-centre depth)
      2. view.extents (exact in ortho; approximate in perspective)
      3. Bounding-box projection estimate (always available)

    Falls back to the model bounding-box centre when the cursor NDC is more
    than 2× the viewport half-width from the viewport centre (e.g. a toolbar).
    """
    min_pt = np.array(model_extents[:3], dtype=np.float64)
    max_pt = np.array(model_extents[3:], dtype=np.float64)
    model_center = (min_pt + max_pt) * 0.5

    A = curr_affine.astype(np.float64)
    R = A[:3, :3]   # world→camera rotation
    R_cam = R.T     # camera→world rotation
    t = A[3, :3]    # view translation

    # Model centre in camera space — its Z gives the depth plane.
    mc_cam = model_center @ R + t
    depth = max(abs(float(mc_cam[2])), 1e-3)

    source = "cursor"

    if perspective and frustum and len(frustum) >= 6:
        fl, fr, fb, ft, fn = (float(frustum[i]) for i in range(5))
        near = max(abs(fn), 1e-9)
        cx_center = (fl + fr) * 0.5 * depth / near
        cx_half   = (fr - fl) * 0.5 * depth / near
        cy_center = (fb + ft) * 0.5 * depth / near
        cy_half   = (ft - fb) * 0.5 * depth / near
    elif extents and len(extents) >= 6:
        cx_center = (extents[0] + extents[3]) * 0.5
        cy_center = (extents[1] + extents[4]) * 0.5
        cx_half   = (extents[3] - extents[0]) * 0.5
        cy_half   = (extents[4] - extents[1]) * 0.5
    else:
        # No frustum / extents — estimate from bounding-box projection.
        mn_w = np.array(model_extents[:3], dtype=np.float64)
        mx_w = np.array(model_extents[3:], dtype=np.float64)
        diffs = mx_w - mn_w
        corners_cam = np.array(
            [
                (mn_w + np.array([dx * diffs[0], dy * diffs[1], dz * diffs[2]])) @ R + t
                for dx in (0.0, 1.0)
                for dy in (0.0, 1.0)
                for dz in (0.0, 1.0)
            ]
        )
        cx_center = float(mc_cam[0])
        cy_center = float(mc_cam[1])
        cx_half = max(float(np.max(np.abs(corners_cam[:, 0] - cx_center))) * 1.5, 1e-3)
        cy_half = max(float(np.max(np.abs(corners_cam[:, 1] - cy_center))) * 1.5, 1e-3)
        source = "bbox_estimate"

    cursor_cam = np.array([cx_center + nx * cx_half, cy_center + ny * cy_half, mc_cam[2]])
    cursor_dist = float(np.sqrt((nx * cx_half) ** 2 + (ny * cy_half) ** 2))
    viewport_half = max(cx_half, cy_half)

    if cursor_dist > viewport_half * 2.0:
        return CursorPivotResult(model_center, cursor_dist, viewport_half, False, "model_center_oob")

    pivot = (cursor_cam - t) @ R_cam
    return CursorPivotResult(pivot, cursor_dist, viewport_half, True, source)


class Mouse3d:
    def __init__(self):
        self.id = "mouse0"


class Controller:
    """Manage shared state and event streaming between a local 3D mouse and a remote client.

    Button handling
    ---------------
    Shift (button 20) is tracked as a held modifier.  When another button is
    pressed while Shift is down the shift_map is consulted first; otherwise
    the normal button_map is used.

    Lock rotation
    -------------
    When lock_rotation is True, pitch/yaw/roll are zeroed so only
    pan/zoom is active.

    Custom views (V1/V2/V3)
    -----------------------
    Pressing V1-V3 recalls a saved view (affine + extents + perspective).
    Shift+V1-V3 saves the current view to that slot and persists to disk.
    """

    CACHE_REFRESH_INTERVAL = 0.5  # seconds between slow-state refreshes

    def __init__(
        self,
        reader: asyncio.StreamReader,
        _: Mouse3d,
        wamp_state_handler: WampSession,
        client_metadata: dict,
    ):
        self.id = "controller0"
        self.client_metadata = client_metadata
        self.reader = reader
        self.wamp_state_handler = wamp_state_handler

        self.wamp_state_handler.wamp.subscribe_handlers[self.controller_uri] = self.subscribe
        self.wamp_state_handler.wamp.call_handlers["wss://127.51.68.120/3dconnexion#update"] = self.client_update

        self.subscribed = False
        self.focus = False
        self.lock_rotation = False
        self._horizon_lock: bool = False  # suppress roll only; pitch/yaw still active
        self.shift_held = False
        self.ctrl_held = False
        self._cursor_pivot_enabled: bool = True
        self._auto_lock_active: bool = False  # True when lock was applied automatically by context
        self._notification_task: asyncio.Task | None = None  # strong ref so GC won't cancel it
        self._icon_refresh_task: asyncio.Task | None = None  # strong ref for icon refresh tasks
        set_lock_led(False)  # reset LED to match initial lock_rotation state

        self.button_map: dict[int, str] = get_button_map()
        self.shift_map: dict[int, str] = get_shift_map()
        self.ctrl_map: dict[int, str] = get_ctrl_map()
        self.hotkeys: list[dict] = get_hotkeys()
        self.context_hotkey_map: dict[str, list[dict]] = get_context_hotkey_map()
        self.saved_views: dict[int, dict] = _load_saved_views()

        # Cached slow-changing state (refreshed every CACHE_REFRESH_INTERVAL seconds)
        self._cached_model_extents: list | None = None
        self._cached_perspective: bool | None = None
        self._cached_extents: list | None = None
        self._cached_frustum: list | None = None
        self._cache_time: float = 0.0

        # Pivot lock: world-space pivot is computed once at gesture start and held
        # for the entire gesture (so the model orbits a fixed point even as the
        # view matrix changes frame-to-frame).
        self._locked_pivot: np.ndarray | None = None
        self._last_motion_time: float = 0.0
        self._GESTURE_GAP_S: float = 0.15  # seconds of silence = new gesture

        # Workaround for spacenavd bug #78: Enterprise releases all buttons when a
        # keyboard key is pressed simultaneously. Track last key-inject time so we
        # can suppress a spurious Shift-release that arrives right after an inject.
        self._last_key_inject_time: float = 0.0

        # Cached viewport aspect ratio (width/height), updated by _fit_affine.
        # Used as fallback when view.extents is not yet available.
        self._viewport_ar: float = 16.0 / 9.0

        # Cursor NDC position received from the /cursor WebSocket.
        # Active once the userscript connects; used for cursor-based pivot.
        self._cursor_ndc: list[float] = [0.0, 0.0]
        self._cursor_active: bool = False
        self._cursor_debug_pivot: list[float] = [0.0, 0.0, 0.0]
        self._cursor_debug_dist: float = 0.0
        self._cursor_debug_viewport_half: float = 0.0
        self._cursor_debug_used_cursor: bool = False
        # "native", "cursor", "model_center", "no_extents"
        self._cursor_debug_pivot_source: str = "none"

        # Onshape-level motion sensitivity multipliers (config.json → "motion").
        # These scale on top of the baseline constants; 1.0 = unchanged behaviour.
        _mc = load_config().get("motion", {})
        self._base_rotation_scale: float = float(_mc.get("rotation_scale", 1.0))
        self._base_translation_scale: float = float(_mc.get("translation_scale", 1.0))
        self._base_zoom_scale: float = float(_mc.get("zoom_scale", 1.0))

        # Runtime sensitivity level (1–5); multiplied on top of base scales.
        # Level 3 = 1.0× (config value unchanged).
        self._SENSITIVITY_MULTIPLIERS = (0.2, 0.5, 1.0, 2.0, 4.0)
        self._sensitivity_level: int = 3  # 1-indexed
        self._apply_sensitivity()

        # Camera mode: when True, all rotation and lateral translation axes are
        # inverted so the puck feels like flying (camera moves) rather than
        # manipulating an object (model moves).
        self._camera_mode: bool = False

        # Context-aware commands sent by Onshape via client_update
        self._active_set: str = ""
        self._context_commands: dict[str, list[dict]] = {}  # activeSet -> [{id, label}, ...]
        self._svg_cache: dict[str, bytes] = {}  # command id -> raw SVG bytes
        self._last_display_key: tuple = ()  # debounce display updates

        self.display = _display
        asyncio.ensure_future(self._restore_context_display())

    async def subscribe(self, msg: Subscribe):
        logging.info("handling subscribe %s", msg)
        self.subscribed = True
        self.focus = True

    async def client_update(self, controller_id: str, args: dict[str, Any]):
        # Log unknown keys so we can discover undocumented properties
        known_keys = {"focus", "images", "commands"}
        extra = {k: args[k] for k in args if k not in known_keys}
        if extra:
            logging.info("3dx#update extra keys: %s", list(extra.keys()))

        if (focus := args.get("focus")) is not None:
            self.focus = focus

        if (imgs := args.get("images")) is not None:
            import base64

            updated_ids = set()
            for entry in imgs:
                cmd_id = entry.get("id", "")
                raw = entry.get("data", "")
                if cmd_id and raw:
                    self._svg_cache[cmd_id] = base64.b64decode(raw.replace(" ", ""))
                    updated_ids.add(cmd_id)
            # If any updated icon belongs to the currently displayed commands, refresh
            ctx_override = self.context_hotkey_map.get(self._active_set)
            if ctx_override is not None:
                override_ids = {hk.get("action", "")[8:] for hk in ctx_override if hk.get("action", "").startswith("onshape_")}
                if updated_ids & override_ids:
                    self._last_display_key = ()  # reset so next commands update also redraws
                    self._icon_refresh_task = asyncio.create_task(self._restore_context_display())
            else:
                display_cmds = self._context_commands.get(self._active_set, [])
                if updated_ids & {c["id"] for c in display_cmds[:12]}:
                    self._last_display_key = ()  # force redraw

        if (cmds := args.get("commands")) is not None:
            active_set = cmds.get("activeSet", "")
            tree = cmds.get("tree")
            if tree and not self._context_commands:
                # Log the raw tree once so we can inspect real command IDs
                logging.debug("commands tree (first receive): %s", json.dumps(tree)[:3000])

            if tree:
                # Tree contains all contexts at once; group commands per top-level category
                for cat_node in tree.get("nodes", []):
                    cat_id = cat_node.get("id", "")
                    flat = self._flatten_commands(cat_node)
                    if flat:
                        self._context_commands[cat_id] = flat

            if active_set != self._active_set:
                old_set = self._active_set
                logging.info("Context: %s → %s", old_set, active_set)
                self._active_set = active_set
                self._handle_context_lock(old_set, active_set)

            # Update display only when context commands actually change
            ctx_override = self.context_hotkey_map.get(active_set)
            if ctx_override is not None:
                # Per-context config overrides Onshape tree — stable key so it
                # only fires once per context switch, not on every tree refresh.
                display_key = (active_set, "__override__")
                if display_key != self._last_display_key:
                    self._last_display_key = display_key
                    enriched = self._enrich_override_hotkeys(ctx_override)
                    loop = asyncio.get_event_loop()
                    loop.run_in_executor(
                        None,
                        self.display.show_hotkeys,
                        enriched,
                        self._sensitivity_level,
                        self._camera_mode,
                        self._cursor_pivot_enabled,
                    )
            else:
                display_cmds = self._context_commands.get(active_set, [])
                display_key = (active_set, tuple(c["id"] for c in display_cmds[:12]))
                logging.info(
                    "display update check: active=%r cmds=%d key_changed=%s", active_set, len(display_cmds), display_key != self._last_display_key
                )
                if display_key != self._last_display_key:
                    self._last_display_key = display_key
                    loop = asyncio.get_event_loop()
                    if display_cmds:
                        hotkeys = self._commands_to_hotkeys(display_cmds[:12])

                        def _do_show(hk=hotkeys, sl=self._sensitivity_level, cm=self._camera_mode, cp=self._cursor_pivot_enabled):
                            try:
                                self.display.show_hotkeys(hk, sl, cm, cp)
                            except Exception as exc:
                                logging.warning("show_hotkeys failed: %s", exc)

                        loop.run_in_executor(None, _do_show)
                    else:
                        loop.run_in_executor(
                            None,
                            self.display.show_hotkeys,
                            self.hotkeys,
                            self._sensitivity_level,
                            self._camera_mode,
                            self._cursor_pivot_enabled,
                        )

    @property
    def controller_uri(self) -> str:
        return f"wss://127.51.68.120/3dconnexion3dcontroller/{self.id}"

    async def remote_write(self, *args):
        return await self.wamp_state_handler.client_rpc(self.controller_uri, "self:update", *args)

    async def remote_read(self, *args):
        return await self.wamp_state_handler.client_rpc(self.controller_uri, "self:read", *args)

    async def _refresh_cache(self):
        """Fetch slow-changing state from the client concurrently."""
        (
            self._cached_model_extents,
            self._cached_perspective,
            self._cached_extents,
        ) = await asyncio.gather(
            self.remote_read("model.extents"),
            self.remote_read("view.perspective"),
            self.remote_read("view.extents"),
        )
        self._cached_frustum = None  # view.frustum not supported by Onshape
        self._cache_time = time.monotonic()

    def _invalidate_cache(self):
        """Force a cache refresh on the next motion event (call after any action that changes view state)."""
        self._cache_time = 0.0

    @staticmethod
    def _flatten_commands(cat_node: dict) -> list[dict]:
        """Return all leaf commands (type 2) from a single category node."""
        result = []

        def walk(nodes):
            for node in nodes:
                if node.get("type") == 2 and "id" in node:
                    result.append({"id": node["id"], "label": node.get("label", "")})
                if "nodes" in node:
                    walk(node["nodes"])

        walk(cat_node.get("nodes", []))
        return result

    def _enrich_override_hotkeys(self, hotkeys: list[dict]) -> list[dict]:
        """Attach cached SVG icons to context-override hotkeys where available.

        For each hotkey whose action is 'onshape_<id>', look up <id> in the
        SVG cache populated when Onshape sends the Assembly command tree.
        """
        result = []
        for hk in hotkeys:
            hk = dict(hk)
            action = hk.get("action", "")
            if action.startswith("onshape_") and "svg" not in hk:
                svg = self._svg_cache.get(action[8:])
                if svg:
                    hk["svg"] = svg
            result.append(hk)
        return result

    def _commands_to_hotkeys(self, commands: list[dict]) -> list[dict]:
        """Convert a flat command list to the hotkey format used by the display."""
        hotkeys = []
        for c in commands:
            hk = {"label": (c["label"] or "")[:4].upper(), "action": f"onshape_{c['id']}"}
            svg = self._svg_cache.get(c["id"])
            if svg:
                hk["svg"] = svg
            hotkeys.append(hk)
        while len(hotkeys) < 12:
            hotkeys.append({"label": "", "action": "noop"})
        return hotkeys[:12]

    async def start_mouse_event_stream(self):
        logging.info("Starting the mouse stream")
        while True:
            mouse_event = await self.reader.read(32)
            if not (self.focus and self.subscribed):
                continue
            # Drain any queued events so we only process the most recent one.
            # wait_for(timeout=0) returns immediately if data is already buffered;
            # raises TimeoutError if the queue is empty — no private API needed.
            try:
                tail = await asyncio.wait_for(self.reader.read(32 * 64), timeout=0)
                combined = mouse_event + tail
                n = len(combined) // 32
                mouse_event = combined[(n - 1) * 32 : n * 32]
            except asyncio.TimeoutError:
                pass
            nums = struct.unpack("iiiiiiii", mouse_event)
            event = from_message(list(nums))
            try:
                await self.update_client(event)
            except Exception:
                logging.exception("Error handling mouse event — continuing")

    # ------------------------------------------------------------------ #
    #  Top-level event router                                             #
    # ------------------------------------------------------------------ #

    async def update_client(self, event: MotionEvent | ButtonEvent):
        if isinstance(event, ButtonEvent):
            await self._handle_button_event(event)
            return
        await self._handle_motion(event)

    async def _handle_button_event(self, event: ButtonEvent):
        # Track Shift modifier on both press and release.
        # Guard against spacenavd bug #78: on SpaceMouse Enterprise, pressing a
        # keyboard modifier key (Ctrl/Alt) causes all device buttons to be spuriously
        # released. If a key-inject happened within 150 ms, ignore the Shift release.
        if event.button_id == SHIFT_BUTTON_ID:
            if not event.pressed and (time.monotonic() - self._last_key_inject_time) < 0.15:
                logging.debug("Shift release suppressed (key inject within 150 ms)")
                return
            self.shift_held = event.pressed
            return

        # Track Ctrl modifier — same pattern as Shift.
        if event.button_id == CTRL_BUTTON_ID:
            self.ctrl_held = event.pressed
            return

        if not event.pressed:
            return  # ignore releases for all other buttons

        # Ctrl-modified action takes priority, then Shift, then base map.
        if self.ctrl_held and event.button_id in self.ctrl_map:
            action = self.ctrl_map[event.button_id]
            modifier = " [+Ctrl]"
        elif self.shift_held and event.button_id in self.shift_map:
            action = self.shift_map[event.button_id]
            modifier = " [+Shift]"
        else:
            action = self.button_map.get(event.button_id, "noop")
            modifier = ""

        logging.info("Button %d%s → %s", event.button_id, modifier, action)
        await self._execute_action(action)
        self._invalidate_cache()

    # ------------------------------------------------------------------ #
    #  Action dispatch                                                    #
    # ------------------------------------------------------------------ #

    async def _execute_action(self, action: str):  # noqa: C901
        if not action or action == "noop":
            return

        if action.startswith("view_"):
            await self._action_set_view(action[5:])

        elif action == "fit":
            await self._action_fit()

        elif action == "zoom_in":
            await self._action_zoom(0.8)
        elif action == "zoom_out":
            await self._action_zoom(1.25)

        elif action == "toggle_lock_rotation":
            self.lock_rotation = not self.lock_rotation
            self._auto_lock_active = False  # user took manual control
            set_lock_led(self.lock_rotation)
            msg = "LOCK ON" if self.lock_rotation else "LOCK OFF"
            logging.info(msg)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.display.show_message, msg)
            await asyncio.sleep(1.2)
            await self._restore_context_display()

        elif action == "toggle_horizon_lock":
            self._horizon_lock = not self._horizon_lock
            msg = "HORIZ ON" if self._horizon_lock else "HORIZ OFF"
            logging.info(msg)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.display.show_message, msg)
            await asyncio.sleep(1.2)
            await self._restore_context_display()

        elif action == "toggle_perspective":
            current = await self.remote_read("view.perspective")
            await self.remote_write("view.perspective", not current)
            await self._signal_motion()

        elif action == "roll_view":
            await self._action_roll_view()

        elif action == "rotate_view_cw":
            await self._action_rotate_view(clockwise=True)
        elif action == "rotate_view_ccw":
            await self._action_rotate_view(clockwise=False)

        elif action.startswith("recall_view_"):
            slot = int(action[-1])
            await self._action_recall_view(slot)

        elif action.startswith("save_view_"):
            slot = int(action[-1])
            await self._action_save_view(slot)

        elif action.startswith("key_"):
            self._last_key_inject_time = time.monotonic()
            self._inject_key(action[4:])

        elif action.startswith("hotkey_"):
            try:
                idx = int(action[7:]) - 1
            except ValueError:
                return
            # Per-context config overrides take priority over the Onshape tree
            ctx_override = self.context_hotkey_map.get(self._active_set)
            if ctx_override is not None:
                if 0 <= idx < len(ctx_override):
                    sub = ctx_override[idx].get("action", "noop")
                    if sub != action:
                        await self._execute_action(sub)
                return
            # Context-aware: use Onshape's current command list when available
            ctx_cmds = self._context_commands.get(self._active_set, [])
            if idx < len(ctx_cmds):
                await self._invoke_onshape_command(ctx_cmds[idx]["id"])
                return
            # Fall back to configured hotkey action
            if 0 <= idx < len(self.hotkeys):
                sub = self.hotkeys[idx].get("action", "noop")
                if sub != action:
                    await self._execute_action(sub)

        elif action.startswith("onshape_"):
            await self._invoke_onshape_command(action[8:])

        elif action == "menu":
            await self._action_cycle_sensitivity()

        elif action == "toggle_camera_mode":
            await self._action_toggle_camera_mode()

        elif action == "toggle_cursor_pivot":
            await self._action_toggle_cursor_pivot()

        else:
            logging.warning("Unknown action: %r", action)

    # ------------------------------------------------------------------ #
    #  Action implementations                                             #
    # ------------------------------------------------------------------ #

    async def _action_set_view(self, view_name: str):
        matrix = get_view_matrix(view_name)
        if matrix is None:
            logging.warning("Unknown view: %r", view_name)
            return
        model_extents = await self.remote_read("model.extents")
        curr_extents = await self.remote_read("view.extents")
        perspective = await self.remote_read("view.perspective")
        try:
            await self.remote_write("view.perspective", False)
        except Exception:
            pass

        if not model_extents or len(model_extents) < 6:
            logging.warning("_action_set_view: model.extents unavailable — aborting")
            return
        A = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
        R = A[:3, :3]
        mn = np.array(model_extents[:3], dtype=np.float64)
        mx = np.array(model_extents[3:6], dtype=np.float64)

        # Centre the view on the model by shifting the camera pan offset.
        center = (mn + mx) / 2.0
        cam_ctr = center @ R  # model centre in camera space
        A[3, :3] = -cam_ctr  # pan so model centre = viewport origin

        # Compute orthographic half-extents that fit the model.
        if curr_extents and len(curr_extents) >= 5 and curr_extents[3] > 1e-9 and curr_extents[4] > 1e-9:
            viewport_ar = curr_extents[3] / curr_extents[4]
        else:
            viewport_ar = self._viewport_ar
        corners = np.array(
            [
                [mn[0], mn[1], mn[2]],
                [mx[0], mn[1], mn[2]],
                [mn[0], mx[1], mn[2]],
                [mx[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]],
                [mx[0], mn[1], mx[2]],
                [mn[0], mx[1], mx[2]],
                [mx[0], mx[1], mx[2]],
            ],
            dtype=np.float64,
        )
        cam = corners @ R
        lo, hi = cam.min(0), cam.max(0)
        hx = (hi[0] - lo[0]) / 2.0 * 1.05
        hy = (hi[1] - lo[1]) / 2.0 * 1.05
        hz = (hi[2] - lo[2]) / 2.0 * 1.05
        if hy < 1e-12:
            hy = hx / viewport_ar
        if hx / hy >= viewport_ar:
            ext_x, ext_y = hx, hx / viewport_ar
        else:
            ext_x, ext_y = hy * viewport_ar, hy
        extents = [-ext_x, -ext_y, -hz, ext_x, ext_y, hz]

        await self.remote_write("motion", True)
        await self.remote_write("view.affine", A.reshape(-1).tolist())
        if not perspective:
            await self.remote_write("view.extents", extents)
        await self.remote_write("motion", False)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.display.show_message, view_name.upper())
        await asyncio.sleep(0.8)
        await self._restore_context_display()

    async def _action_fit(self):
        curr_affine = await self.remote_read("view.affine")
        if not curr_affine or len(curr_affine) < 16:
            return
        A = np.asarray(curr_affine, dtype=np.float64).reshape(4, 4)
        await self._fit_affine(A)

    async def _fit_affine(self, A: np.ndarray):
        """Reposition and scale affine A to fit the model, then apply it."""
        model_extents = await self.remote_read("model.extents")
        perspective = await self.remote_read("view.perspective")
        curr_extents = await self.remote_read("view.extents")

        if not model_extents or len(model_extents) < 6:
            logging.warning("_fit_affine: model.extents unavailable — aborting")
            return
        mn = np.array(model_extents[0:3], dtype=np.float64)
        mx = np.array(model_extents[3:6], dtype=np.float64)
        center = (mn + mx) / 2.0
        R = A[:3, :3]  # rows = camera axes in world space

        if perspective:
            radius = np.linalg.norm(mx - mn) * 0.5
            cam_z_world = R[2, :]
            dist = max(radius / np.tan(np.radians(22.5)), radius * 1.1)
            cam_pos = center + cam_z_world * dist
            new_affine = np.array(A)
            new_affine[3, :3] = cam_pos
            await asyncio.gather(
                self.remote_write("motion", True),
                self.remote_write("view.affine", new_affine.reshape(-1).tolist()),
            )
            return

        # Infer viewport aspect ratio (width/height) from the current extents.
        # Onshape always stores extents with ext_x / ext_y == viewport AR.
        if curr_extents and len(curr_extents) >= 5 and curr_extents[3] > 1e-9 and curr_extents[4] > 1e-9:
            viewport_ar = curr_extents[3] / curr_extents[4]
            self._viewport_ar = viewport_ar  # cache for next time
        else:
            viewport_ar = self._viewport_ar

        # Project 8 model corners into camera space using A.
        corners = np.array(
            [
                [mn[0], mn[1], mn[2]],
                [mx[0], mn[1], mn[2]],
                [mn[0], mx[1], mn[2]],
                [mx[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]],
                [mx[0], mn[1], mx[2]],
                [mn[0], mx[1], mx[2]],
                [mx[0], mx[1], mx[2]],
            ],
            dtype=np.float64,
        )
        cam = corners @ R + A[3, :3]

        # Compute the camera-space bounding box of the model and its centre.
        cam_lo = cam.min(axis=0)
        cam_hi = cam.max(axis=0)
        cam_center = (cam_lo + cam_hi) / 2.0

        # Half-extents measured from the model's camera-space centre,
        # not from the viewport origin, so off-centre models fit correctly.
        cam_x_half = float((cam_hi[0] - cam_lo[0]) / 2.0)
        cam_y_half = float((cam_hi[1] - cam_lo[1]) / 2.0)
        cam_z_half = float((cam_hi[2] - cam_lo[2]) / 2.0)

        # Shift the affine translation so the model centre lands at the
        # viewport origin.  This is a delta on the existing translation,
        # so it works regardless of where the current pan is.
        new_affine = np.array(A, dtype=np.float64)
        new_affine[3, :3] -= cam_center

        # Scale to maintain viewport aspect ratio so the model isn't distorted.
        pad = 1.05
        if cam_y_half < 1e-12:
            cam_y_half = cam_x_half / viewport_ar
        model_ar = cam_x_half / cam_y_half
        if model_ar >= viewport_ar:
            ext_x = cam_x_half * pad
            ext_y = ext_x / viewport_ar
        else:
            ext_y = cam_y_half * pad
            ext_x = ext_y * viewport_ar
        ext_z = cam_z_half * pad

        extents = [-ext_x, -ext_y, -ext_z, ext_x, ext_y, ext_z]

        logging.info("fit AR=%.3f ext_xy=[%.4f, %.4f]", viewport_ar, ext_x, ext_y)
        await asyncio.gather(
            self.remote_write("motion", True),
            self.remote_write("view.affine", new_affine.reshape(-1).tolist()),
            self.remote_write("view.extents", extents),
        )

    async def _action_zoom(self, scale: float):
        perspective = await self.remote_read("view.perspective")
        if not perspective:
            extents = await self.remote_read("view.extents")
            if extents is not None:
                await asyncio.gather(
                    self.remote_write("motion", True),
                    self.remote_write("view.extents", [c * scale for c in extents]),
                )
        else:
            curr_affine = np.asarray(await self.remote_read("view.affine"), dtype=np.float64).reshape(4, 4)
            R = curr_affine[:3, :3].T
            cam_z = R[:, 2]
            model_extents = await self.remote_read("model.extents")
            extent_size = (max(abs(e) for e in model_extents) if model_extents else None) or 1.0
            step = extent_size * (1.0 - scale) * 0.5
            trans = np.eye(4, dtype=np.float64)
            trans[3, :3] = cam_z * step
            new_affine = trans @ curr_affine
            await asyncio.gather(
                self.remote_write("motion", True),
                self.remote_write("view.affine", new_affine.reshape(-1).tolist()),
            )

    async def _action_roll_view(self):
        curr_affine = np.asarray(await self.remote_read("view.affine"), dtype=np.float64).reshape(4, 4)
        R = curr_affine[:3, :3].T
        look = -R[:, 2]
        world_up = np.array([0.0, 0.0, 1.0])  # Onshape is Z-up
        right = np.cross(look, world_up)
        if np.linalg.norm(right) < 1e-6:  # look ∥ Z — fall back to Y
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(look, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, look)
        up /= np.linalg.norm(up)
        R_new = np.stack([right, up, -look], axis=1)
        new_affine = np.array(curr_affine, dtype=np.float64)
        new_affine[:3, :3] = R_new.T
        await asyncio.gather(
            self.remote_write("motion", True),
            self.remote_write("view.affine", new_affine.reshape(-1).tolist()),
        )

    async def _action_rotate_view(self, clockwise: bool):
        """Roll the camera 90° clockwise (CW) or counterclockwise (CCW).

        A CW roll means the image rotates clockwise on screen: the current
        up-direction moves toward the right.  In terms of the world→camera
        rotation matrix A[:3,:3] (rows = camera axes in world space):
          CW:  new_X = old_Y,  new_Y = -old_X   →  R = [[0,1,0],[-1,0,0],[0,0,1]]
          CCW: new_X = -old_Y, new_Y =  old_X   →  R = [[0,-1,0],[1,0,0],[0,0,1]]
        In orthographic mode the X/Y extents are swapped so the zoom is preserved.
        """
        curr_affine = np.asarray(await self.remote_read("view.affine"), dtype=np.float64).reshape(4, 4)
        perspective = await self.remote_read("view.perspective")

        if clockwise:
            R_roll = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        else:
            R_roll = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

        new_affine = np.array(curr_affine)
        new_affine[:3, :3] = R_roll @ curr_affine[:3, :3]

        writes = [self.remote_write("motion", True), self.remote_write("view.affine", new_affine.reshape(-1).tolist())]
        if not perspective:
            extents = await self.remote_read("view.extents")
            if extents is not None and len(extents) >= 6:
                new_extents = [extents[1], extents[0], extents[2], extents[4], extents[3], extents[5]]
                writes.append(self.remote_write("view.extents", new_extents))
        await asyncio.gather(*writes)

    async def _action_save_view(self, slot: int):
        try:
            affine = await self.remote_read("view.affine")
            extents = await self.remote_read("view.extents")
            perspective = await self.remote_read("view.perspective")
        except Exception as exc:
            logging.warning("save_view_%d: could not read view state — %s", slot, exc)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.display.show_message, "ERR SAVE")
            await asyncio.sleep(0.8)
            await self._restore_context_display()
            return
        logging.debug(
            "save_view_%d: affine=%s extents=%s perspective=%s",
            slot,
            affine,
            extents,
            perspective,
        )
        self.saved_views[slot] = {
            "affine": affine,
            "extents": extents,
            "perspective": perspective,
        }
        _persist_saved_views(self.saved_views)
        msg = f"SAVE V{slot}"
        logging.info(msg)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.display.show_message, msg)
        await asyncio.sleep(0.8)
        await self._restore_context_display()

    async def _action_recall_view(self, slot: int):
        view = self.saved_views.get(slot)
        if view is None:
            msg = f"V{slot} EMPTY"
            logging.info("Custom view slot %d is empty", slot)
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.display.show_message, msg)
            await asyncio.sleep(0.8)
            await self._restore_context_display()
            return
        logging.debug(
            "recall_view_%d: writing affine=%s extents=%s perspective=%s",
            slot,
            view["affine"],
            view.get("extents"),
            view.get("perspective"),
        )
        try:
            writes = [
                self.remote_write("motion", True),
                self.remote_write("view.affine", view["affine"]),
            ]
            if not view["perspective"] and view.get("extents") is not None:
                # Adjust extents for the current viewport aspect ratio so the
                # recalled view is correct even if the window was resized since
                # saving.  cy_half encodes zoom level; cx/cy centre encodes pan.
                # Only cx_half is scaled to match the current window AR.
                # Read live extents (not cache) so a window resize between save
                # and recall is accounted for.
                saved_ex = view["extents"]
                live_ex = await self.remote_read("view.extents")
                if live_ex and len(live_ex) >= 5 and (live_ex[4] - live_ex[1]) > 1e-9:
                    curr_ar = (live_ex[3] - live_ex[0]) / (live_ex[4] - live_ex[1])
                else:
                    curr_ar = self._viewport_ar
                cx = (saved_ex[0] + saved_ex[3]) / 2
                cy = (saved_ex[1] + saved_ex[4]) / 2
                hy = (saved_ex[4] - saved_ex[1]) / 2
                hx = hy * curr_ar
                recalled_ex = [cx - hx, cy - hy, saved_ex[2], cx + hx, cy + hy, saved_ex[5]]
                writes.append(self.remote_write("view.extents", recalled_ex))
            await asyncio.gather(*writes)
        except Exception as exc:
            logging.warning("recall_view_%d failed — %s", slot, exc)
        msg = f"VIEW V{slot}"
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.display.show_message, msg)
        await asyncio.sleep(0.8)
        await self._restore_context_display()

    async def _invoke_onshape_command(self, command_id: str):
        """Ask Onshape to execute a command by ID (e.g. 'Part Studio-extrude').

        Tries several WAMP property names in sequence; if all are rejected by
        Onshape, falls back to injecting the corresponding keyboard shortcut
        via uinput (Wayland-compatible).
        """
        logging.info("Invoking Onshape command: %r", command_id)
        leaf_id = command_id.split("-", 1)[-1] if "-" in command_id else command_id

        # WAMP probe: try candidate property names with a 1.5 s timeout so we
        # don't hang if Onshape silently ignores an unknown method.
        wamp_candidates = [
            ("activeCommand", command_id),
            ("activeCommand", leaf_id),
            ("command", leaf_id),  # original attempt, but with leaf only
        ]
        for prop, value in wamp_candidates:
            try:
                await self.wamp_state_handler.client_rpc(
                    self.controller_uri,
                    "self:update",
                    prop,
                    value,
                    timeout=1.5,
                )
                logging.info("Command invoked via self:update/%s=%r", prop, value)
                return
            except asyncio.TimeoutError:
                logging.warning("WAMP self:update/%s=%r → timeout", prop, value)
                break  # no point retrying further if Onshape didn't respond at all
            except ValueError as exc:
                logging.warning("WAMP self:update/%s=%r → %s", prop, value, exc)

        # Keyboard fallback
        from spacenav_ws.keyboard import inject_shortcut

        if inject_shortcut(command_id):
            return
        logging.warning(
            'No working invocation for command %r.\n  Add a keyboard shortcut to ~/.config/spacenav-ws/shortcuts.json:\n    {"%s": "shift+x"}',
            command_id,
            command_id,
        )

    async def _signal_motion(self):
        await self.remote_write("motion", True)

    # ------------------------------------------------------------------ #
    #  Context-lock helpers                                               #
    # ------------------------------------------------------------------ #

    _2D_CONTEXTS = {"Sketch", "Drawing"}

    def _handle_context_lock(self, old_set: str, new_set: str) -> None:
        """Auto-enable/disable rotation lock when entering/leaving 2D contexts."""
        if new_set in self._2D_CONTEXTS and not self._auto_lock_active:
            self.lock_rotation = True
            self._auto_lock_active = True
            set_lock_led(True)
            logging.info("Auto-lock ON  (%s)", new_set)
            self._notification_task = asyncio.create_task(self._context_notification(new_set))
        elif new_set not in self._2D_CONTEXTS and self._auto_lock_active:
            self.lock_rotation = False
            self._auto_lock_active = False
            set_lock_led(False)
            logging.info("Auto-lock OFF (%s → %s)", old_set, new_set)

    async def _context_notification(self, context: str) -> None:
        """Briefly show a context banner, then restore the hotkey grid."""
        labels = {"Sketch": "SKETCH LOCK", "Drawing": "2D MODE"}
        msg = labels.get(context, f"{context.upper()} MODE")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.display.show_message, msg)
        await asyncio.sleep(0.8)
        await self._restore_context_display()

    async def _restore_context_display(self) -> None:
        """(Re-)draw the hotkey grid for the current context."""
        ctx_override = self.context_hotkey_map.get(self._active_set)
        if ctx_override is not None:
            hotkeys = self._enrich_override_hotkeys(ctx_override)
        else:
            display_cmds = self._context_commands.get(self._active_set, [])
            hotkeys = self._commands_to_hotkeys(display_cmds[:12]) if display_cmds else self.hotkeys
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self.display.show_hotkeys, hotkeys, self._sensitivity_level, self._camera_mode, self._cursor_pivot_enabled
        )

    # ------------------------------------------------------------------ #
    #  Sensitivity                                                         #
    # ------------------------------------------------------------------ #

    def _apply_sensitivity(self) -> None:
        """Recompute the three scale fields from base values × current level multiplier."""
        m = self._SENSITIVITY_MULTIPLIERS[self._sensitivity_level - 1]
        self._rotation_scale = self._base_rotation_scale * m
        self._translation_scale = self._base_translation_scale * m
        self._zoom_scale = self._base_zoom_scale * m

    async def _action_cycle_sensitivity(self) -> None:
        """Step to the next sensitivity level (wraps 5 → 1) and update display."""
        self._sensitivity_level = (self._sensitivity_level % 5) + 1
        self._apply_sensitivity()
        logging.warning(
            "Sensitivity level %d/%d  (×%.2f)",
            self._sensitivity_level,
            5,
            self._SENSITIVITY_MULTIPLIERS[self._sensitivity_level - 1],
        )
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.display.show_sensitivity, self._sensitivity_level)
        await asyncio.sleep(1.2)
        display_cmds = self._context_commands.get(self._active_set, [])
        hotkeys = self._commands_to_hotkeys(display_cmds[:12]) if display_cmds else self.hotkeys
        await loop.run_in_executor(
            None, self.display.show_hotkeys, hotkeys, self._sensitivity_level, self._camera_mode, self._cursor_pivot_enabled
        )

    async def _action_toggle_camera_mode(self) -> None:
        """Toggle between object mode (model moves) and camera mode (fly/orbit)."""
        self._camera_mode = not self._camera_mode
        label = "CAMERA" if self._camera_mode else "OBJECT"
        logging.warning("Motion mode: %s", label)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.display.show_message, label)
        await asyncio.sleep(1.0)
        await self._restore_context_display()

    async def _action_toggle_cursor_pivot(self) -> None:
        """Toggle cursor-based rotation pivot on/off."""
        self._cursor_pivot_enabled = not self._cursor_pivot_enabled
        label = "CPIV ON" if self._cursor_pivot_enabled else "CPIV OFF"
        logging.warning("Cursor pivot: %s", label)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.display.show_message, label)
        await asyncio.sleep(1.0)
        await self._restore_context_display()

    # ------------------------------------------------------------------ #
    #  Key injection (uinput primary, xdotool fallback for X11)           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _inject_key(key_name: str) -> None:
        """Inject a single key press.  Uses uinput (Wayland-compatible) first."""
        from spacenav_ws.keyboard import _send_keys

        if _send_keys(key_name):
            return
        # xdotool fallback (X11 only)
        xkey_map = {
            "esc": "Escape",
            "enter": "Return",
            "delete": "Delete",
            "tab": "Tab",
            "space": "space",
            "alt": "alt",
            "shift": "shift",
            "ctrl": "ctrl",
        }
        xkey = xkey_map.get(key_name, key_name)
        env = {**os.environ, **_X11_ENV}
        try:
            result = subprocess.run(
                ["xdotool", "key", "--clearmodifiers", xkey],
                check=False,
                capture_output=True,
                timeout=0.5,
                env=env,
            )
            if result.returncode != 0:
                logging.warning("xdotool key %r failed: %s", xkey, result.stderr.decode().strip())
        except FileNotFoundError:
            logging.warning("xdotool not found — install with: sudo apt install xdotool")
        except Exception:
            logging.debug("xdotool failed", exc_info=True)

    # ------------------------------------------------------------------ #
    #  Motion handler                                                     #
    # ------------------------------------------------------------------ #

    async def _handle_motion(self, event: MotionEvent):
        # Refresh slow-changing state periodically; only view.affine is read every frame.
        if self._cached_model_extents is None or (time.monotonic() - self._cache_time) > self.CACHE_REFRESH_INTERVAL:
            await self._refresh_cache()

        model_extents = self._cached_model_extents
        perspective = self._cached_perspective
        extents = self._cached_extents

        if not model_extents or len(model_extents) < 6:
            return  # cache not yet populated; drop this frame

        affine_raw = await self.remote_read("view.affine")
        if not affine_raw or len(affine_raw) < 16:
            return  # view not yet ready
        curr_affine = np.asarray(affine_raw, dtype=np.float32).reshape(4, 4)

        R_cam = curr_affine[:3, :3].T
        U, _, Vt = np.linalg.svd(R_cam)
        R_cam = U @ Vt

        pitch = 0.0 if self.lock_rotation else event.pitch
        yaw = 0.0 if self.lock_rotation else event.yaw
        roll = 0.0 if (self.lock_rotation or self._horizon_lock) else event.roll

        # Camera mode: invert rotation axes and lateral translation so the puck
        # feels like flying (you move through the scene) rather than holding the
        # object.  Forward/zoom (Y axis) is unchanged in both modes.
        cam_sign = -1.0 if self._camera_mode else 1.0

        # Rodrigues axis-angle rotation — handles simultaneous multi-axis input
        # as a single rotation (no Euler-order artefacts, no scipy needed).
        ang_cam = np.array([pitch, yaw, -roll], dtype=np.float64) * (cam_sign * 0.01 * np.pi / 180.0 * self._rotation_scale)
        R_delta_cam = _rotation_from_axis_angle(ang_cam, float(np.linalg.norm(ang_cam)))
        R_world = R_cam @ R_delta_cam @ R_cam.T

        rot_delta = np.eye(4, dtype=np.float32)
        rot_delta[:3, :3] = R_world

        # Apply rotation around pivot first, then add camera-relative translation.
        # Pan input is in camera space; multiply by R_cam to convert to world space so
        # Pivot: computed once at gesture start, then locked in world space for the
        # entire gesture so the model orbits a fixed point even as the view changes.
        now = time.monotonic()
        new_gesture = (now - self._last_motion_time) > self._GESTURE_GAP_S
        self._last_motion_time = now

        if new_gesture or self._locked_pivot is None:
            # Priority 1: cursor NDC projection (always attempted when cursor WS is live).
            # Priority 2: Onshape native pivot.position (may be stale / write-only).
            # Priority 3: model bounding-box centre.
            pivot_raw = None
            try:
                pivot_raw = await self.remote_read("pivot.position")
            except Exception:
                pass
            if self._cursor_active and self._cursor_pivot_enabled:
                nx, ny = self._cursor_ndc
                self._locked_pivot = self._cursor_pivot(
                    nx, ny, model_extents, curr_affine, extents,
                    perspective=perspective, frustum=self._cached_frustum,
                )
            elif isinstance(pivot_raw, list) and len(pivot_raw) >= 3:
                self._locked_pivot = np.array(pivot_raw[:3], dtype=np.float64)
                self._cursor_debug_pivot_source = "native"
                self._cursor_debug_used_cursor = False
            else:
                min_pt = np.array(model_extents[:3] if model_extents else [0, 0, 0], dtype=np.float64)
                max_pt = np.array(model_extents[3:6] if model_extents else [0, 0, 0], dtype=np.float64)
                self._locked_pivot = (min_pt + max_pt) * 0.5
                self._cursor_debug_pivot_source = "model_center"
                self._cursor_debug_used_cursor = False
            self._cursor_debug_pivot[:] = self._locked_pivot.tolist()
            logging.debug(
                "pivot: src=%s native=%s cursor_active=%s extents=%s perspective=%s ndc=(%.2f,%.2f) used=%s p=[%.3f,%.3f,%.3f]",
                self._cursor_debug_pivot_source,
                pivot_raw is not None,
                self._cursor_active,
                extents is not None,
                perspective,
                *self._cursor_ndc,
                self._cursor_debug_used_cursor,
                *self._locked_pivot,
            )

        pivot = self._locked_pivot
        pivot_pos, pivot_neg = self._get_affine_pivot_matrices(pivot)
        new_affine = curr_affine @ (pivot_neg @ rot_delta @ pivot_pos)

        # Frustum-proportional pan scaling (PR #5):
        # Perspective: use frustum spans at pivot depth so panning maps 1:1 to screen.
        # Orthographic: use view XY spans from cached extents.
        eye = curr_affine[3, :3].astype(np.float64)
        fwd = (-R_cam[:, 2]).astype(np.float64)
        depth = max(float(abs(np.dot(fwd, pivot.astype(np.float64) - eye))), 1e-3)

        if perspective and self._cached_frustum and len(self._cached_frustum) >= 6:
            fl, fr, fb, ft, fn, _ = self._cached_frustum
            near = max(abs(float(fn)), 1e-9)
            span_x = max((float(fr) - float(fl)) * depth / near, 1e-9)
            span_y = max((float(ft) - float(fb)) * depth / near, 1e-9)
        elif not perspective and extents and len(extents) >= 5:
            span_x = max(extents[3] - extents[0], 1e-9)
            span_y = max(extents[4] - extents[1], 1e-9)
        else:
            span_x = span_y = depth  # fallback

        _PAN_RATE = 3.0 / 350.0 * self._translation_scale  # view-spans per second per max-count (matches PR #5)
        cam_trans = (
            np.array(
                [-event.x * span_x * cam_sign, -event.z * span_y * cam_sign, event.y * depth / 6.0],
                dtype=np.float64,
            )
            * _PAN_RATE
        )
        new_affine[3, :3] += (R_cam @ cam_trans).astype(np.float32)

        writes = [
            self.remote_write("motion", True),
            self.remote_write("view.affine", new_affine.reshape(-1).tolist()),
        ]
        if not perspective and extents and len(extents) >= 6:
            # Zoom: Onshape's base-2 law (scale = 2^(-delta/6)); center-preserving.
            # Only XY extents scale — near/far (indices 2 & 5) are left unchanged.
            _ZOOM_RATE = 20.0 / np.log(2.0) / 350.0 * self._zoom_scale
            zoom_scale = 2.0 ** (-event.y * _ZOOM_RATE / 6.0)
            cx = (extents[0] + extents[3]) * 0.5
            cy = (extents[1] + extents[4]) * 0.5
            hx = (extents[3] - extents[0]) * 0.5 * zoom_scale
            hy = (extents[4] - extents[1]) * 0.5 * zoom_scale
            new_extents = [cx - hx, cy - hy, extents[2], cx + hx, cy + hy, extents[5]]
            writes.append(self.remote_write("view.extents", new_extents))
            self._cached_extents = new_extents  # keep cache in sync
        await asyncio.gather(*writes)

    def _cursor_pivot(
        self,
        nx: float,
        ny: float,
        model_extents: list,
        curr_affine: np.ndarray,
        extents: list | None,
        perspective: bool = False,
        frustum: list | None = None,
    ) -> np.ndarray:
        """Thin wrapper around compute_cursor_pivot that records debug state."""
        r = compute_cursor_pivot(nx, ny, model_extents, curr_affine, extents, perspective, frustum)
        self._cursor_debug_dist = r.dist
        self._cursor_debug_viewport_half = r.viewport_half
        self._cursor_debug_used_cursor = r.used_cursor
        self._cursor_debug_pivot_source = r.source
        return r.pivot

    @staticmethod
    def _get_affine_pivot_matrices(pivot: np.ndarray):
        pivot_pos = np.eye(4, dtype=np.float32)
        pivot_pos[3, :3] = pivot.astype(np.float32)
        pivot_neg = np.eye(4, dtype=np.float32)
        pivot_neg[3, :3] = -pivot.astype(np.float32)
        return pivot_pos, pivot_neg


async def create_mouse_controller(
    wamp_state_handler: WampSession,
    spacenav_reader: asyncio.StreamReader,
) -> Controller:
    await wamp_state_handler.wamp.begin()

    msg = await wamp_state_handler.wamp.next_message()
    while isinstance(msg, Prefix):
        await wamp_state_handler.wamp.run_message_handler(msg)
        msg = await wamp_state_handler.wamp.next_message()

    if not isinstance(msg, Call) or msg.proc_uri != "3dx_rpc:create" or len(msg.args) < 2 or msg.args[0] != "3dconnexion:3dmouse":
        raise ValueError(f"WAMP handshake failed: expected 3dmouse create, got {msg!r}")
    mouse = Mouse3d()
    logging.info('Created 3d mouse "%s" for version %s', mouse.id, msg.args[1])
    await wamp_state_handler.wamp.send_message(CallResult(msg.call_id, {"connexion": mouse.id}))

    msg = await wamp_state_handler.wamp.next_message()
    if not isinstance(msg, Call) or msg.proc_uri != "3dx_rpc:create" or len(msg.args) < 3 or msg.args[0] != "3dconnexion:3dcontroller" or msg.args[1] != mouse.id:
        raise ValueError(f"WAMP handshake failed: expected 3dcontroller create, got {msg!r}")
    metadata = msg.args[2]
    controller = Controller(spacenav_reader, mouse, wamp_state_handler, metadata)
    logging.info(
        'Created controller "%s" for mouse "%s", client "%s" v%s',
        controller.id,
        mouse.id,
        metadata.get("name", "unknown"),
        metadata.get("version", "unknown"),
    )
    await wamp_state_handler.wamp.send_message(CallResult(msg.call_id, {"instance": controller.id}))
    return controller
