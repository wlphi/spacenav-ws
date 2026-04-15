# spacenav-ws

A Python bridge that lets Onshape on Linux use a 3Dconnexion SpaceMouse. It speaks the same WAMP-based protocol that the official Windows driver uses, so Onshape behaves as if a real 3Dconnexion driver is present.

This is a fork of [rmstorm/spacenav-ws](https://github.com/rmstorm/spacenav-ws). The upstream project established the protocol and basic motion. This fork adds full SpaceMouse Enterprise support: LCD display, all navigation buttons, camera mode, cursor-based pivot, and Wayland-compatible key injection.

---

## Quick start

See [INSTALL.md](INSTALL.md) for the full setup including udev rules, systemd service, and sensitivity tuning.

The short version:

```bash
# 1. Validate spacenavd is running and the device is visible
uv run spacenav-ws read-mouse

# 2. Start the server
uv run spacenav-ws serve

# 3. Open https://127.51.68.120:8181 in your browser and accept the cert
# 4. Install the userscript (see INSTALL.md step 10)
# 5. Open an Onshape document
```

---

## What this fork adds

### SpaceMouse Enterprise LCD

The 640×150 pixel display shows a 6×2 grid of the 12 programmable hotkeys with Onshape-provided icons when available. Status badges and indicators:

- **Blue bar** at the bottom — current sensitivity level (5 segments)
- **CAM** badge (orange, top-right) — camera/fly mode is active
- **CPV** badge (red, top-left) — cursor-based pivot is disabled

The display also shows full-screen banners for mode changes (LOCK ON/OFF, CAMERA/OBJECT, CPIV ON/OFF) and snaps back to the grid automatically.

### Motion controls

All six axes work: pan (X/Z), zoom (Y), and simultaneous pitch/yaw/roll. The motion model uses frustum-proportional panning and Rodrigues axis-angle rotation, consistent with the [PR #5 math](https://github.com/jordens/spacenav-ws) from the jordens rewrite.

**Cursor-based rotation pivot**: when the userscript is active, rotation pivots around the 3D point under the mouse cursor rather than the model center. Falls back to the model bounding-box center when the cursor is off-screen or pivot is disabled.

### Button mapping (SpaceMouse Enterprise)

| Button | Default | Shift | Ctrl |
|--------|---------|-------|------|
| 0–11 | Hotkeys 1–12 | — | — |
| 12 (Menu) | Cycle sensitivity | Camera mode | Toggle cursor pivot |
| 13 (Fit) | Fit to screen | — | — |
| 14 (Top) | Top view | Bottom view | — |
| 15 (Right) | Right view | Left view | — |
| 16 (Front) | Front view | Back view | — |
| 17 (Rotate) | Roll CW 90° | Roll CCW 90° | — |
| 18 | Esc | — | — |
| 19 | Alt | — | — |
| 20 | **Shift modifier** | — | — |
| 21 | **Ctrl modifier** | — | — |
| 22 | Lock rotation | — | — |
| 23 | Enter | — | — |
| 24 | Delete | — | — |
| 25 | Tab | — | — |
| 26 | Space | — | — |
| 27 (V1) | Recall view 1 | Save view 1 | — |
| 28 (V2) | Recall view 2 | Save view 2 | — |
| 29 (V3) | Recall view 3 | Save view 3 | — |
| 30 (ISO) | ISO 1 view | ISO 2 view | — |

The 12 programmable hotkeys (buttons 0–11) display the current Onshape context commands when in a sketch or Part Studio, and fall back to the configured labels otherwise.

Sensitivity cycles through five multipliers: **0.2×, 0.5×, 1.0×, 2.0×, 4.0×**. The default is 1.0×. This is a per-session multiplier applied on top of the base constants in spacenav-ws; the spacenavd device-level sensitivity is independent.

### Key injection

Keys (Esc, Alt, Enter, Delete, Tab, Space) are injected via uinput, which works on Wayland. xdotool is used as a fallback on X11.

### Rotation lock LED

The rotation lock state is reflected on the SpaceMouse Enterprise's status LED via the evdev interface.

---

## Configuration

All configuration lives in `~/.config/spacenav-ws/config.json`. The file is optional; missing keys fall back to defaults.

```json
{
  "motion": {
    "rotation_scale":    1.0,
    "translation_scale": 0.1,
    "zoom_scale":        0.1
  },
  "button_map": {
    "13": "zoom_in"
  },
  "shift_map": {
    "13": "view_iso2"
  },
  "ctrl_map": {
    "12": "toggle_cursor_pivot"
  },
  "hotkeys": [
    {"label": "ISO1", "action": "view_iso1"},
    {"label": "ISO2", "action": "view_iso2"},
    {"label": "TOP",  "action": "view_top"},
    {"label": "FRNT", "action": "view_front"},
    {"label": "RGHT", "action": "view_right"},
    {"label": "FIT",  "action": "fit"},
    {"label": "BACK", "action": "view_back"},
    {"label": "BOTT", "action": "view_bottom"},
    {"label": "LEFT", "action": "view_left"},
    {"label": "",     "action": "noop"},
    {"label": "",     "action": "noop"},
    {"label": "",     "action": "noop"}
  ],
  "cors_origins": ["https://cad.onshape.com"]
}
```

Available actions: `view_front`, `view_back`, `view_top`, `view_bottom`, `view_right`, `view_left`, `view_iso1`, `view_iso2`, `fit`, `zoom_in`, `zoom_out`, `toggle_lock_rotation`, `toggle_camera_mode`, `toggle_cursor_pivot`, `toggle_perspective`, `roll_view`, `rotate_view_cw`, `rotate_view_ccw`, `recall_view_1/2/3`, `save_view_1/2/3`, `key_esc`, `key_enter`, `key_delete`, `key_tab`, `key_space`, `key_alt`, `key_shift`, `key_ctrl`, `hotkey_1` … `hotkey_12`, `noop`.

---

## Architecture decisions

These are the deliberate deviations from the upstream repo and from approaches that were considered but rejected.

### All Onshape logic stays in spacenav-ws, not spacenavd

spacenavd (`/etc/spnavrc`) handles static, device-level transforms: dead zones, sensitivity multipliers, axis inversion. It has no concept of keyboard modifiers, application context, SVG rendering, or WAMP protocol. Everything Onshape-specific — frustum-proportional panning, context-aware rotation lock, button→action dispatch, display rendering — lives in spacenav-ws. A PR to add LCD support directly to spacenavd was rejected by the maintainer; that confirms the boundary.

### jordens PR #5 was not merged wholesale

PR #5 in the upstream repo is a 30-file architectural rewrite. The math improvements were sound: Rodrigues axis-angle rotation, frustum-proportional pan, center-preserving orthographic zoom, native pivot.position read. Those were cherry-picked individually. The full merge was deferred because the architectural overhaul would have required rewriting most of this fork's additions at the same time.

### cursor_state.py was deleted

The upstream had a `cursor_state.py` module holding global cursor state. This was replaced with instance fields on the `Controller` class (`_cursor_ndc`, `_cursor_active`, etc.). Globals that outlive a connection caused stale state across reconnects; putting state on the controller instance ties its lifetime to the WebSocket session.

### view.frustum is not used

The 3Dconnexion protocol defines `view.frustum` as `[left, right, bottom, top, near, far]`. Onshape does not expose it. HAR captures of the official Windows driver session confirmed the property is never present. The frustum path in `_cursor_pivot` is retained as dead code in case a future Onshape update adds it, but `_cached_frustum` is always `None`.

### Pivot priority: cursor before native

`pivot.position` is readable via WAMP but Onshape returns the last-written value, not a live cursor-intersection result. Reading it gives a constant world-space point that does not track cursor movement. The cursor NDC projection (or bounding-box estimate when `view.extents` is unavailable) is therefore attempted first when the cursor WebSocket is connected. The native pivot is a fallback for sessions without the userscript.

### SVG rendering uses resvg, not Cairo

The original display branch used cairosvg. The upstream maintainer rejected PR #134 partly because of the Cairo dependency. This fork uses `resvg` (a standalone CLI, `pacman -S resvg`) which has no Python binding and no C library dependency. The calling code shells out via subprocess.

### spnavcfg is not recommended

The Qt5→Qt6 migration in spnavcfg is incomplete (upstream issue #43). AUR builds are unreliable. Device configuration is done by editing `/etc/spnavrc` directly; spacenavd reloads it on `SIGHUP` without restarting.

---

## Development

```bash
uv run spacenav-ws serve --hot-reload   # auto-restart on file changes
uv run pytest tests/ -v                 # run the test suite
```

The test suite covers the rotation math, cursor pivot geometry, view matrices, button config loading, and display rendering. All tests run offline with no hardware or network.

```
tests/test_math.py     — rotation, cursor pivot, view matrices
tests/test_buttons.py  — button map defaults and config overrides
tests/test_display.py  — rendering smoke tests and layout invariants
```
