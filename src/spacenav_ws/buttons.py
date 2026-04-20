"""Button configuration for SpaceMouse Enterprise.

Button IDs confirmed by live testing on a SpaceMouse Enterprise (2024).

Shift modifier (button 20) is tracked by the controller; buttons with a
Shift variant are listed in ENTERPRISE_SHIFT_MAP.

Config file: ~/.config/spacenav-ws/config.json
------------------------------------------------------------
Override button actions or hotkey labels/actions:

{
    "button_map": {
        "13": "fit",
        "30": "view_iso1"
    },
    "shift_map": {
        "30": "view_iso2"
    },
    "hotkeys": [
        {"label": "ISO1",  "action": "view_iso1"},
        {"label": "ISO2",  "action": "view_iso2"},
        {"label": "TOP",   "action": "view_top"},
        {"label": "FRNT",  "action": "view_front"},
        {"label": "RGHT",  "action": "view_right"},
        {"label": "FIT",   "action": "fit"},
        {"label": "BACK",  "action": "view_back"},
        {"label": "BOTT",  "action": "view_bottom"},
        {"label": "LEFT",  "action": "view_left"},
        {"label": "",      "action": "noop"},
        {"label": "",      "action": "noop"},
        {"label": "",      "action": "noop"}
    ]
}

Available actions
-----------------
view_front, view_back, view_top, view_bottom, view_right, view_left,
view_iso1, view_iso2     -- named view + fit to model
fit                       -- fit current orientation to model extents
zoom_in, zoom_out         -- scale extents ×0.8 / ×1.25
toggle_lock_rotation      -- lock/unlock rotation (suppress pitch/yaw/roll)
toggle_horizon_lock       -- lock/unlock horizon (suppress roll only; pitch/yaw still work)
toggle_camera_mode        -- toggle object mode ↔ camera (fly) mode
toggle_perspective        -- switch perspective ↔ orthographic
roll_view                 -- snap roll to 0 while keeping look direction
rotate_view_cw            -- roll the view 90° clockwise (image rotates CW on screen)
rotate_view_ccw           -- roll the view 90° counterclockwise
recall_view_1/2/3         -- recall a saved custom view
save_view_1/2/3           -- save current view (also Shift+V1/V2/V3)
key_<key>                 -- inject a single key or chord, e.g. key_esc, key_shift+s,
                             key_ctrl+z.  Modifier+key combos use + as separator.
                             Named keys: esc, enter, delete, tab, space, alt, shift, ctrl
onshape_<id>              -- invoke an Onshape command by its tree ID, e.g.
                             onshape_extrude, onshape_mate_FASTENED.
                             Falls back to the keyboard shortcut defined in keyboard.py.
toggle_cursor_pivot       -- enable/disable cursor-based rotation pivot (Ctrl+Menu)
hotkey_1 … hotkey_12      -- dispatch to configured hotkey action
noop                      -- do nothing

Built-in LCD icons  (set automatically — no config needed)
-----------------------------------------------------------
These actions display a custom icon on the LCD grid instead of a text label.
Defined in icons.py as inline SVGs rendered with no brightness adaptation
(they are already designed for the dark display).

    view_iso1    isometric cube, blue top/front face highlighted
    view_iso2    isometric cube, blue right/front face highlighted
    view_top     cube with top face lit
    view_bottom  cube with bottom face lit (top face dimmed)
    view_right   cube with right face lit
    view_left    cube with left face lit
    view_front   cube with front face lit
    view_back    cube with back face dimmed (opposite of front)
    fit          crosshair / fit-to-extents symbol

Onshape context icons  (loaded at runtime from Onshape's command tree)
-----------------------------------------------------------------------
When using onshape_<id> actions in context_hotkeys, the icon Onshape sends
for that command ID is shown automatically — no extra config needed.  The
icon appears after the first time Onshape sends the images update for that
context (usually within a second of entering the context).

Icons are only available for IDs that appear in the Onshape command tree.
To discover available IDs and whether icons are cached, check the logs:

    journalctl --user -u spacenav-ws | grep "svg_cache ids"
"""

import json
import logging
from pathlib import Path

from spacenav_ws.icons import VIEW_ICONS

# ---------------------------------------------------------------------------
# Primary button map  (button_id → action, no modifier held)
# ---------------------------------------------------------------------------
ENTERPRISE_DEFAULT_BUTTON_MAP: dict[int, str] = {
    # 12 programmable hotkeys (top of device, numbered 1–12)
    0: "hotkey_1",
    1: "hotkey_2",
    2: "hotkey_3",
    3: "hotkey_4",
    4: "hotkey_5",
    5: "hotkey_6",
    6: "hotkey_7",
    7: "hotkey_8",
    8: "hotkey_9",
    9: "hotkey_10",
    10: "hotkey_11",
    11: "hotkey_12",
    # Fixed view / navigation buttons
    12: "menu",
    13: "fit",
    14: "view_top",
    15: "view_right",
    16: "view_front",
    17: "rotate_view_cw",
    22: "toggle_lock_rotation",
    30: "view_iso1",
    # Custom view recall (V1/V2/V3)
    27: "recall_view_1",
    28: "recall_view_2",
    29: "recall_view_3",
    # Keyboard injection
    18: "key_esc",
    # 19 is the Alt modifier — handled separately, NOT in this map
    # 20 is the Shift modifier — handled separately, NOT in this map
    # 21 is the Ctrl modifier — handled separately, NOT in this map
    23: "key_enter",
    24: "key_delete",
    25: "key_tab",
    26: "key_space",
}

# ---------------------------------------------------------------------------
# Shift-modified actions  (button_id → action when Shift is held)
# ---------------------------------------------------------------------------
ENTERPRISE_DEFAULT_SHIFT_MAP: dict[int, str] = {
    12: "toggle_camera_mode",  # Shift + Menu → toggle object/camera mode
    14: "view_bottom",  # Shift + Top
    15: "view_left",  # Shift + Right
    16: "view_back",  # Shift + Front
    17: "rotate_view_ccw",  # Shift + Rotate → rotate left
    22: "toggle_horizon_lock",  # Shift + Lock → horizon lock (suppress roll only)
    27: "save_view_1",  # Shift + V1
    28: "save_view_2",  # Shift + V2
    29: "save_view_3",  # Shift + V3
    30: "view_iso2",  # Shift + ISO1
}

# The Shift key button ID — tracked as a held modifier, never dispatched as an action
SHIFT_BUTTON_ID = 20

# The Ctrl key button ID — tracked as a held modifier, never dispatched as an action
CTRL_BUTTON_ID = 21

# The Alt key button ID — tracked as a held modifier, never dispatched as an action
ALT_BUTTON_ID = 19

# ---------------------------------------------------------------------------
# Ctrl-modified actions  (button_id → action when Ctrl is held)
# ---------------------------------------------------------------------------
ENTERPRISE_DEFAULT_CTRL_MAP: dict[int, str] = {
    12: "toggle_cursor_pivot",  # Ctrl + Menu → toggle cursor-based rotation pivot
}

# ---------------------------------------------------------------------------
# Alt-modified actions  (button_id → action when Alt is held)
# ---------------------------------------------------------------------------
ENTERPRISE_DEFAULT_ALT_MAP: dict[int, str] = {
    12: "toggle_invert_pitch",  # Alt + Menu → invert pitch axis
}

# ---------------------------------------------------------------------------
# Ctrl+Alt-modified actions  (button_id → action when both Ctrl and Alt held)
# ---------------------------------------------------------------------------
ENTERPRISE_DEFAULT_CTRL_ALT_MAP: dict[int, str] = {
    12: "toggle_swap_yz",  # Ctrl + Alt + Menu → swap zoom ↔ vertical pan axes
}

# ---------------------------------------------------------------------------
# Default hotkey labels & actions for the 12 programmable keys
# ---------------------------------------------------------------------------
_DEFAULT_HOTKEY_DEFS: list[dict] = [
    {"label": "ISO1", "action": "view_iso1"},
    {"label": "ISO2", "action": "view_iso2"},
    {"label": "TOP", "action": "view_top"},
    {"label": "FRNT", "action": "view_front"},
    {"label": "RGHT", "action": "view_right"},
    {"label": "FIT", "action": "fit"},
    {"label": "BACK", "action": "view_back"},
    {"label": "BOTT", "action": "view_bottom"},
    {"label": "LEFT", "action": "view_left"},
    {"label": "CHFR", "action": "onshape_chamfer"},
    {"label": "DRFT", "action": "onshape_draft"},
    {"label": "", "action": "noop"},
]


def _build_default_hotkeys() -> list[dict]:
    hotkeys = []
    for entry in _DEFAULT_HOTKEY_DEFS:
        hk = dict(entry)
        svg = VIEW_ICONS.get(entry["action"])
        if svg is not None:
            hk["svg"] = svg
            hk["no_adapt"] = True  # icons are designed for dark LCD, skip brightness boost
        hotkeys.append(hk)
    return hotkeys


DEFAULT_HOTKEYS: list[dict] = _build_default_hotkeys()

CONFIG_PATH = Path.home() / ".config" / "spacenav-ws" / "config.json"

_config_cache: dict | None = None


def load_config() -> dict:
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                _config_cache = json.load(f)
            logging.info("Loaded button config from %s", CONFIG_PATH)
        except Exception:
            logging.exception("Failed to parse %s — using defaults", CONFIG_PATH)
            _config_cache = {}
    else:
        logging.info("No config at %s — using defaults", CONFIG_PATH)
        _config_cache = {}
    return _config_cache


def _read_config_from_disk() -> dict:
    """Read config.json directly from disk, bypassing the cache."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except Exception:
            logging.warning("_read_config_from_disk: could not read %s", CONFIG_PATH)
    return {}


def _write_config_to_disk(cfg: dict) -> None:
    """Write cfg to config.json and update the in-memory cache."""
    global _config_cache
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
        _config_cache = cfg
    except Exception:
        logging.warning("_write_config_to_disk: could not write %s", CONFIG_PATH)


def save_device_state(
    *,
    sensitivity_level: int,
    invert_pitch: bool,
    swap_yz: bool,
    lock_rotation: bool,
    horizon_lock: bool,
    camera_mode: bool,
    cursor_pivot: bool,
) -> None:
    """Persist all device toggle state to config.json under the 'state' key."""
    cfg = _read_config_from_disk()
    cfg["state"] = {
        "sensitivity_level": sensitivity_level,
        "invert_pitch":      invert_pitch,
        "swap_yz":           swap_yz,
        "lock_rotation":     lock_rotation,
        "horizon_lock":      horizon_lock,
        "camera_mode":       camera_mode,
        "cursor_pivot":      cursor_pivot,
    }
    _write_config_to_disk(cfg)


def load_device_state() -> dict:
    """Return persisted device state, falling back to safe defaults.

    Reads from the unified 'state' key; also accepts legacy 'axis_flags' and
    'sensitivity_level' top-level keys written by earlier versions.
    """
    cfg = load_config()
    state = cfg.get("state", {})
    # Legacy fallbacks so old config.json files keep working.
    if "sensitivity_level" not in state:
        state["sensitivity_level"] = cfg.get("sensitivity_level", 3)
    if "invert_pitch" not in state:
        state["invert_pitch"] = cfg.get("axis_flags", {}).get("invert_pitch", False)
    if "swap_yz" not in state:
        state["swap_yz"] = cfg.get("axis_flags", {}).get("swap_yz", False)
    return state


def get_button_map() -> dict[int, str]:
    config = load_config()
    m = dict(ENTERPRISE_DEFAULT_BUTTON_MAP)
    for k, v in config.get("button_map", {}).items():
        try:
            m[int(k)] = str(v)
        except ValueError:
            logging.warning("Invalid button_map key: %r", k)
    return m


def get_shift_map() -> dict[int, str]:
    config = load_config()
    m = dict(ENTERPRISE_DEFAULT_SHIFT_MAP)
    for k, v in config.get("shift_map", {}).items():
        try:
            m[int(k)] = str(v)
        except ValueError:
            logging.warning("Invalid shift_map key: %r", k)
    return m


def get_ctrl_map() -> dict[int, str]:
    config = load_config()
    m = dict(ENTERPRISE_DEFAULT_CTRL_MAP)
    for k, v in config.get("ctrl_map", {}).items():
        try:
            m[int(k)] = str(v)
        except ValueError:
            logging.warning("Invalid ctrl_map key: %r", k)
    return m


def get_alt_map() -> dict[int, str]:
    config = load_config()
    m = dict(ENTERPRISE_DEFAULT_ALT_MAP)
    for k, v in config.get("alt_map", {}).items():
        try:
            m[int(k)] = str(v)
        except ValueError:
            logging.warning("Invalid alt_map key: %r", k)
    return m


def get_ctrl_alt_map() -> dict[int, str]:
    config = load_config()
    m = dict(ENTERPRISE_DEFAULT_CTRL_ALT_MAP)
    for k, v in config.get("ctrl_alt_map", {}).items():
        try:
            m[int(k)] = str(v)
        except ValueError:
            logging.warning("Invalid ctrl_alt_map key: %r", k)
    return m


def get_hotkeys() -> list[dict]:
    config = load_config()
    hotkeys = [dict(h) for h in DEFAULT_HOTKEYS]
    for i, entry in enumerate(config.get("hotkeys", [])[:12]):
        hotkeys[i] = {
            "label": str(entry.get("label", ""))[:4].upper(),
            "action": str(entry.get("action", "noop")),
        }
    return hotkeys


def get_context_hotkey_map() -> dict[str, list[dict]]:
    """Return per-context hotkey overrides from config.

    Config format (context name matches Onshape's activeSet string):
    {
        "context_hotkeys": {
            "Assembly": [
                {"label": "INS",  "action": "onshape_insertPart"},
                {"label": "FAST", "action": "onshape_mate_FASTENED"}
            ]
        }
    }

    When a context override is defined it takes priority over the Onshape
    command tree for both display and button dispatch.
    """
    config = load_config()
    result: dict[str, list[dict]] = {}
    for context, entries in config.get("context_hotkeys", {}).items():
        hotkeys = []
        for entry in list(entries)[:12]:
            hotkeys.append({
                "label": str(entry.get("label", ""))[:4].upper(),
                "action": str(entry.get("action", "noop")),
            })
        result[str(context)] = hotkeys
    return result
