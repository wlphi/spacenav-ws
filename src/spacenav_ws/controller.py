import asyncio
import json
import logging
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

# Capture X11 display info at import time (while the launching shell's env is available).
_X11_ENV: dict[str, str] = {}
for _k in ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "DBUS_SESSION_BUS_ADDRESS"):
    if _k in os.environ:
        _X11_ENV[_k] = os.environ[_k]

import numpy as np
from scipy.spatial import transform

from spacenav_ws.spacenav import MotionEvent, ButtonEvent, from_message
from spacenav_ws.wamp import WampSession, Prefix, Call, Subscribe, CallResult
from spacenav_ws.views import get_view_matrix
from spacenav_ws.buttons import (
    get_button_map, get_shift_map, get_hotkeys, SHIFT_BUTTON_ID
)
from spacenav_ws.display import EnterpriseDisplay

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
        self.shift_held = False
        self._auto_lock_active: bool = False  # True when lock was applied automatically by context

        self.button_map: dict[int, str] = get_button_map()
        self.shift_map: dict[int, str] = get_shift_map()
        self.hotkeys: list[dict] = get_hotkeys()
        self.saved_views: dict[int, dict] = _load_saved_views()

        # Cached slow-changing state (refreshed every CACHE_REFRESH_INTERVAL seconds)
        self._cached_model_extents: list | None = None
        self._cached_perspective: bool | None = None
        self._cached_extents: list | None = None
        self._cache_time: float = 0.0

        # Context-aware commands sent by Onshape via client_update
        self._active_set: str = ""
        self._context_commands: dict[str, list[dict]] = {}  # activeSet -> [{id, label}, ...]
        self._svg_cache: dict[str, bytes] = {}              # command id -> raw SVG bytes
        self._last_display_key: tuple = ()                  # debounce display updates

        self.display = _display
        self.display.show_hotkeys(self.hotkeys)

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
            display_cmds = self._context_commands.get(self._active_set, [])
            if updated_ids & {c["id"] for c in display_cmds[:12]}:
                self._last_display_key = ()  # force redraw

        if (cmds := args.get("commands")) is not None:
            active_set = cmds.get("activeSet", "")
            tree = cmds.get("tree")
            if tree and not self._context_commands:
                # Log the raw tree once so we can inspect real command IDs
                logging.warning("commands tree (first receive): %s", json.dumps(tree)[:3000])

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
            display_cmds = self._context_commands.get(active_set, [])
            display_key = (active_set, tuple(c["id"] for c in display_cmds[:12]))
            logging.info("display update check: active=%r cmds=%d key_changed=%s",
                         active_set, len(display_cmds), display_key != self._last_display_key)
            if display_key != self._last_display_key:
                self._last_display_key = display_key
                loop = asyncio.get_event_loop()
                if display_cmds:
                    hotkeys = self._commands_to_hotkeys(display_cmds[:12])
                    def _do_show(hk=hotkeys):
                        try:
                            self.display.show_hotkeys(hk)
                        except Exception as exc:
                            logging.warning("show_hotkeys failed: %s", exc)
                    loop.run_in_executor(None, _do_show)
                else:
                    loop.run_in_executor(None, self.display.show_hotkeys, self.hotkeys)

    @property
    def controller_uri(self) -> str:
        return f"wss://127.51.68.120/3dconnexion3dcontroller/{self.id}"

    async def remote_write(self, *args):
        return await self.wamp_state_handler.client_rpc(self.controller_uri, "self:update", *args)

    async def remote_read(self, *args):
        return await self.wamp_state_handler.client_rpc(self.controller_uri, "self:read", *args)

    async def _refresh_cache(self):
        """Fetch slow-changing state from the client concurrently."""
        self._cached_model_extents, self._cached_perspective, self._cached_extents = await asyncio.gather(
            self.remote_read("model.extents"),
            self.remote_read("view.perspective"),
            self.remote_read("view.extents"),
        )
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
                if node.get("type") == 2:
                    result.append({"id": node["id"], "label": node.get("label", "")})
                if "nodes" in node:
                    walk(node["nodes"])
        walk(cat_node.get("nodes", []))
        return result

    def _commands_to_hotkeys(self, commands: list[dict]) -> list[dict]:
        """Convert a flat command list to the hotkey format used by the display."""
        hotkeys = []
        for c in commands:
            hk = {"label": c["label"][:4].upper(), "action": f"onshape_{c['id']}"}
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
            # Drain any queued events so we only process the most recent one
            while len(self.reader._buffer) >= 32:
                mouse_event = bytes(self.reader._buffer[:32])
                del self.reader._buffer[:32]
            nums = struct.unpack("iiiiiiii", mouse_event)
            event = from_message(list(nums))
            if self.client_metadata["name"] in ["Onshape", "WebThreeJS Sample"]:
                await self.update_client(event)
            else:
                logging.warning(
                    "Unknown client! Cannot send mouse events, client_metadata:%s",
                    self.client_metadata,
                )

    # ------------------------------------------------------------------ #
    #  Top-level event router                                             #
    # ------------------------------------------------------------------ #

    async def update_client(self, event: MotionEvent | ButtonEvent):
        if isinstance(event, ButtonEvent):
            await self._handle_button_event(event)
            return
        await self._handle_motion(event)

    async def _handle_button_event(self, event: ButtonEvent):
        # Track Shift modifier on both press and release
        if event.button_id == SHIFT_BUTTON_ID:
            self.shift_held = event.pressed
            return

        if not event.pressed:
            return  # ignore releases for all other buttons

        # Shift-modified action takes priority
        if self.shift_held and event.button_id in self.shift_map:
            action = self.shift_map[event.button_id]
        else:
            action = self.button_map.get(event.button_id, "noop")

        logging.info(
            "Button %d%s → %s",
            event.button_id,
            " [+Shift]" if self.shift_held else "",
            action,
        )
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
            self._auto_lock_active = False   # user took manual control
            msg = "LOCK ON" if self.lock_rotation else "LOCK OFF"
            logging.info(msg)
            self.display.show_message(msg)
            await asyncio.sleep(1.2)
            self._restore_context_display()

        elif action == "toggle_perspective":
            current = await self.remote_read("view.perspective")
            await self.remote_write("view.perspective", not current)
            await self._signal_motion()

        elif action == "roll_view":
            await self._action_roll_view()

        elif action.startswith("recall_view_"):
            slot = int(action[-1])
            await self._action_recall_view(slot)

        elif action.startswith("save_view_"):
            slot = int(action[-1])
            await self._action_save_view(slot)

        elif action.startswith("key_"):
            self._inject_key(action[4:])

        elif action.startswith("hotkey_"):
            try:
                idx = int(action[7:]) - 1
            except ValueError:
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
            pass

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
        perspective = await self.remote_read("view.perspective")
        await self.remote_write("motion", True)
        await self.remote_write("view.affine", matrix)
        if not perspective:
            await self.remote_write("view.extents", model_extents)
        await self.remote_write("motion", False)
        self.display.show_message(view_name.upper())
        await asyncio.sleep(0.8)
        self.display.show_hotkeys(self.hotkeys)

    async def _action_fit(self):
        model_extents = await self.remote_read("model.extents")
        curr_affine = await self.remote_read("view.affine")
        perspective = await self.remote_read("view.perspective")
        curr_extents = await self.remote_read("view.extents")
        # Infer viewport aspect ratio (width/height) from the current extents.
        # Onshape always stores extents with ext_x / ext_y == viewport AR.
        if curr_extents and curr_extents[3] > 1e-9 and curr_extents[4] > 1e-9:
            viewport_ar = curr_extents[3] / curr_extents[4]
            self._viewport_ar = viewport_ar   # cache for next time
        else:
            viewport_ar = getattr(self, "_viewport_ar", 16.0 / 9.0)

        mn = np.array(model_extents[0:3], dtype=np.float64)
        mx = np.array(model_extents[3:6], dtype=np.float64)
        center = (mn + mx) / 2.0
        A = np.asarray(curr_affine, dtype=np.float64).reshape(4, 4)
        R = A[:3, :3]   # rows = camera axes in world space

        if perspective:
            radius = np.linalg.norm(mx - mn) * 0.5
            cam_z_world = R[2, :]
            dist = max(radius / np.tan(np.radians(22.5)), radius * 1.1)
            cam_pos = center + cam_z_world * dist
            new_affine = np.array(A)
            new_affine[3, :3] = cam_pos
            await self.remote_write("motion", True)
            await self.remote_write("view.affine", new_affine.reshape(-1).tolist())
            await self.remote_write("motion", False)
            return

        # Project 8 model corners into camera space using the current affine.
        corners = np.array([
            [mn[0], mn[1], mn[2]], [mx[0], mn[1], mn[2]],
            [mn[0], mx[1], mn[2]], [mx[0], mx[1], mn[2]],
            [mn[0], mn[1], mx[2]], [mx[0], mn[1], mx[2]],
            [mn[0], mx[1], mx[2]], [mx[0], mx[1], mx[2]],
        ], dtype=np.float64)
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
        await self.remote_write("motion", True)
        await self.remote_write("view.affine", new_affine.reshape(-1).tolist())
        await self.remote_write("view.extents", extents)
        await self.remote_write("motion", False)

    async def _action_zoom(self, scale: float):
        perspective = await self.remote_read("view.perspective")
        await self.remote_write("motion", True)
        if not perspective:
            extents = await self.remote_read("view.extents")
            await self.remote_write("view.extents", [c * scale for c in extents])
        else:
            curr_affine = np.asarray(
                await self.remote_read("view.affine"), dtype=np.float64
            ).reshape(4, 4)
            R = curr_affine[:3, :3].T
            cam_z = R[:, 2]
            model_extents = await self.remote_read("model.extents")
            extent_size = max(abs(e) for e in model_extents) or 1.0
            step = extent_size * (1.0 - scale) * 0.5
            trans = np.eye(4, dtype=np.float64)
            trans[3, :3] = cam_z * step
            new_affine = trans @ curr_affine
            await self.remote_write("view.affine", new_affine.reshape(-1).tolist())
        await self.remote_write("motion", False)

    async def _action_roll_view(self):
        curr_affine = np.asarray(
            await self.remote_read("view.affine"), dtype=np.float64
        ).reshape(4, 4)
        R = curr_affine[:3, :3].T
        look = -R[:, 2]
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(world_up, look)
        if np.linalg.norm(right) < 1e-6:
            world_up = np.array([0.0, 0.0, 1.0])
            right = np.cross(world_up, look)
        right /= np.linalg.norm(right)
        up = np.cross(look, right)
        R_new = np.stack([right, up, -look], axis=1)
        new_affine = np.array(curr_affine, dtype=np.float64)
        new_affine[:3, :3] = R_new.T
        await self.remote_write("motion", True)
        await self.remote_write("view.affine", new_affine.reshape(-1).tolist())
        await self.remote_write("motion", False)

    async def _action_save_view(self, slot: int):
        affine = await self.remote_read("view.affine")
        extents = await self.remote_read("view.extents")
        perspective = await self.remote_read("view.perspective")
        self.saved_views[slot] = {
            "affine": affine,
            "extents": extents,
            "perspective": perspective,
        }
        _persist_saved_views(self.saved_views)
        msg = f"SAVE V{slot}"
        logging.info(msg)
        self.display.show_message(msg)
        await asyncio.sleep(0.8)
        self.display.show_hotkeys(self.hotkeys)

    async def _action_recall_view(self, slot: int):
        view = self.saved_views.get(slot)
        if view is None:
            msg = f"V{slot} EMPTY"
            logging.info("Custom view slot %d is empty", slot)
            self.display.show_message(msg)
            await asyncio.sleep(0.8)
            self.display.show_hotkeys(self.hotkeys)
            return
        await self.remote_write("motion", True)
        await self.remote_write("view.affine", view["affine"])
        await self.remote_write("view.perspective", view["perspective"])
        if not view["perspective"]:
            await self.remote_write("view.extents", view["extents"])
        await self.remote_write("motion", False)
        msg = f"VIEW V{slot}"
        self.display.show_message(msg)
        await asyncio.sleep(0.8)
        self.display.show_hotkeys(self.hotkeys)

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
            ("command",       leaf_id),   # original attempt, but with leaf only
        ]
        for prop, value in wamp_candidates:
            try:
                await self.wamp_state_handler.client_rpc(
                    self.controller_uri, "self:update", prop, value,
                    timeout=1.5,
                )
                logging.info("Command invoked via self:update/%s=%r", prop, value)
                return
            except asyncio.TimeoutError:
                logging.warning("WAMP self:update/%s=%r → timeout", prop, value)
                break   # no point retrying further if Onshape didn't respond at all
            except ValueError as exc:
                logging.warning("WAMP self:update/%s=%r → %s", prop, value, exc)

        # Keyboard fallback
        from spacenav_ws.keyboard import inject_shortcut
        if inject_shortcut(command_id):
            return
        logging.warning(
            "No working invocation for command %r.\n"
            "  Add a keyboard shortcut to ~/.config/spacenav-ws/shortcuts.json:\n"
            '    {"%s": "shift+x"}',
            command_id, command_id,
        )

    async def _signal_motion(self):
        await self.remote_write("motion", True)
        await self.remote_write("motion", False)

    # ------------------------------------------------------------------ #
    #  Context-lock helpers                                               #
    # ------------------------------------------------------------------ #

    _2D_CONTEXTS = {"Sketch", "Drawing"}

    def _handle_context_lock(self, old_set: str, new_set: str) -> None:
        """Auto-enable/disable rotation lock when entering/leaving 2D contexts."""
        if new_set in self._2D_CONTEXTS and not self._auto_lock_active:
            self.lock_rotation = True
            self._auto_lock_active = True
            logging.info("Auto-lock ON  (%s)", new_set)
            asyncio.create_task(self._context_notification(new_set))
        elif new_set not in self._2D_CONTEXTS and self._auto_lock_active:
            self.lock_rotation = False
            self._auto_lock_active = False
            logging.info("Auto-lock OFF (%s → %s)", old_set, new_set)

    async def _context_notification(self, context: str) -> None:
        """Briefly show a context banner, then restore the hotkey grid."""
        labels = {"Sketch": "SKETCH LOCK", "Drawing": "2D MODE"}
        msg = labels.get(context, f"{context.upper()} MODE")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.display.show_message, msg)
        await asyncio.sleep(0.8)
        self._restore_context_display()

    def _restore_context_display(self) -> None:
        """(Re-)draw the hotkey grid for the current context."""
        display_cmds = self._context_commands.get(self._active_set, [])
        hotkeys = self._commands_to_hotkeys(display_cmds[:12]) if display_cmds else self.hotkeys
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self.display.show_hotkeys, hotkeys)

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
            "esc": "Escape", "enter": "Return", "delete": "Delete",
            "tab": "Tab", "space": "space",
            "alt": "alt", "shift": "shift", "ctrl": "ctrl",
        }
        xkey = xkey_map.get(key_name, key_name)
        env = {**os.environ, **_X11_ENV}
        try:
            result = subprocess.run(
                ["xdotool", "key", "--clearmodifiers", xkey],
                check=False, capture_output=True, timeout=0.5, env=env,
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

        curr_affine = np.asarray(
            await self.remote_read("view.affine"), dtype=np.float32
        ).reshape(4, 4)

        R_cam = curr_affine[:3, :3].T
        U, _, Vt = np.linalg.svd(R_cam)
        R_cam = U @ Vt

        pitch = 0.0 if self.lock_rotation else event.pitch
        yaw   = 0.0 if self.lock_rotation else event.yaw
        roll  = 0.0 if self.lock_rotation else event.roll

        angles = np.array([pitch, yaw, -roll]) * 0.01
        R_delta_cam = transform.Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        R_world = R_cam @ R_delta_cam @ R_cam.T

        rot_delta = np.eye(4, dtype=np.float32)
        rot_delta[:3, :3] = R_world

        # Apply rotation around model pivot first, then add camera-relative translation.
        # Pan input is in camera space; multiply by R_cam to convert to world space so
        # panning follows the screen axes regardless of camera orientation.
        pivot_pos, pivot_neg = self._get_affine_pivot_matrices(model_extents)
        new_affine = curr_affine @ (pivot_neg @ rot_delta @ pivot_pos)

        extent_scale = sum(extents[3:6]) / 3.0 if extents else 1.0
        cam_trans = np.array([-event.x, -event.z, event.y], dtype=np.float32) * 0.000375 * extent_scale
        new_affine[3, :3] += R_cam @ cam_trans

        writes = [
            self.remote_write("motion", True),
            self.remote_write("view.affine", new_affine.reshape(-1).tolist()),
        ]
        if not perspective:
            scale = 1.0 + event.y * 0.0001
            new_extents = [c * scale for c in extents]
            writes.append(self.remote_write("view.extents", new_extents))
            self._cached_extents = new_extents  # keep cache in sync
        await asyncio.gather(*writes)

    @staticmethod
    def _get_affine_pivot_matrices(model_extents):
        min_pt = np.array(model_extents[0:3], dtype=np.float32)
        max_pt = np.array(model_extents[3:6], dtype=np.float32)
        pivot = (min_pt + max_pt) * 0.5
        pivot_pos = np.eye(4, dtype=np.float32)
        pivot_pos[3, :3] = pivot
        pivot_neg = np.eye(4, dtype=np.float32)
        pivot_neg[3, :3] = -pivot
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

    assert isinstance(msg, Call)
    assert msg.proc_uri == "3dx_rpc:create" and msg.args[0] == "3dconnexion:3dmouse"
    mouse = Mouse3d()
    logging.info('Created 3d mouse "%s" for version %s', mouse.id, msg.args[1])
    await wamp_state_handler.wamp.send_message(CallResult(msg.call_id, {"connexion": mouse.id}))

    msg = await wamp_state_handler.wamp.next_message()
    assert isinstance(msg, Call)
    assert msg.proc_uri == "3dx_rpc:create" and msg.args[0] == "3dconnexion:3dcontroller" and msg.args[1] == mouse.id
    metadata = msg.args[2]
    controller = Controller(spacenav_reader, mouse, wamp_state_handler, metadata)
    logging.info(
        'Created controller "%s" for mouse "%s", client "%s" v%s',
        controller.id, mouse.id, metadata["name"], metadata["version"],
    )
    await wamp_state_handler.wamp.send_message(CallResult(msg.call_id, {"instance": controller.id}))
    return controller
