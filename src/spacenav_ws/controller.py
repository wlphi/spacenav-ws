import asyncio
import json
import logging
import os
import struct
import subprocess
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

        self.button_map: dict[int, str] = get_button_map()
        self.shift_map: dict[int, str] = get_shift_map()
        self.hotkeys: list[dict] = get_hotkeys()
        self.saved_views: dict[int, dict] = _load_saved_views()

        self.display = EnterpriseDisplay()
        self.display.show_hotkeys(self.hotkeys)

    async def subscribe(self, msg: Subscribe):
        logging.info("handling subscribe %s", msg)
        self.subscribed = True
        self.focus = True

    async def client_update(self, controller_id: str, args: dict[str, Any]):
        logging.debug("Got update for '%s': %s", controller_id, args)
        if (focus := args.get("focus")) is not None:
            self.focus = focus

    @property
    def controller_uri(self) -> str:
        return f"wss://127.51.68.120/3dconnexion3dcontroller/{self.id}"

    async def remote_write(self, *args):
        return await self.wamp_state_handler.client_rpc(self.controller_uri, "self:update", *args)

    async def remote_read(self, *args):
        return await self.wamp_state_handler.client_rpc(self.controller_uri, "self:read", *args)

    async def start_mouse_event_stream(self):
        logging.info("Starting the mouse stream")
        while True:
            mouse_event = await self.reader.read(32)
            if self.focus and self.subscribed:
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
            msg = "LOCK ON" if self.lock_rotation else "LOCK OFF"
            logging.info(msg)
            self.display.show_message(msg)
            await asyncio.sleep(1.2)
            self.display.show_hotkeys(self.hotkeys)

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
            if 0 <= idx < len(self.hotkeys):
                sub = self.hotkeys[idx].get("action", "noop")
                if sub != action:
                    await self._execute_action(sub)

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

    async def _signal_motion(self):
        await self.remote_write("motion", True)
        await self.remote_write("motion", False)

    @staticmethod
    def _inject_key(key_name: str):
        key_map = {
            "esc":    "Escape",
            "enter":  "Return",
            "delete": "Delete",
            "tab":    "Tab",
            "space":  "space",
            "alt":    "alt",
            "shift":  "shift",
            "ctrl":   "ctrl",
        }
        xkey = key_map.get(key_name, key_name)
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
        model_extents = await self.remote_read("model.extents")
        perspective = await self.remote_read("view.perspective")
        curr_affine = np.asarray(
            await self.remote_read("view.affine"), dtype=np.float32
        ).reshape(4, 4)

        R_cam = curr_affine[:3, :3].T
        U, _, Vt = np.linalg.svd(R_cam)
        R_cam = U @ Vt

        pitch = 0.0 if self.lock_rotation else event.pitch
        yaw   = 0.0 if self.lock_rotation else event.yaw
        roll  = 0.0 if self.lock_rotation else event.roll

        angles = np.array([pitch, yaw, -roll]) * 0.02
        R_delta_cam = transform.Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
        R_world = R_cam @ R_delta_cam @ R_cam.T

        rot_delta = np.eye(4, dtype=np.float32)
        rot_delta[:3, :3] = R_world
        trans_delta = np.eye(4, dtype=np.float32)
        trans_delta[3, :3] = np.array([-event.x, -event.z, event.y], dtype=np.float32) * 0.0005

        pivot_pos, pivot_neg = self._get_affine_pivot_matrices(model_extents)
        new_affine = trans_delta @ curr_affine @ (pivot_neg @ rot_delta @ pivot_pos)

        if not perspective:
            extents = await self.remote_read("view.extents")
            scale = 1.0 + event.y * 0.0002
            await self.remote_write("motion", True)
            await self.remote_write("view.extents", [c * scale for c in extents])
        else:
            await self.remote_write("motion", True)
        await self.remote_write("view.affine", new_affine.reshape(-1).tolist())

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
