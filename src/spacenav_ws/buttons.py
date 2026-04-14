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
toggle_perspective        -- switch perspective ↔ orthographic
roll_view                 -- snap roll to 0 while keeping look direction
rotate_view_cw            -- roll the view 90° clockwise (image rotates CW on screen)
rotate_view_ccw           -- roll the view 90° counterclockwise
recall_view_1/2/3         -- recall a saved custom view
save_view_1/2/3           -- save current view (also Shift+V1/V2/V3)
key_esc, key_enter, key_delete, key_tab, key_space,
key_alt, key_shift, key_ctrl  -- inject key via xdotool
hotkey_1 … hotkey_12      -- dispatch to configured hotkey action
noop                      -- do nothing
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
    19: "key_alt",
    # 20 is the Shift modifier — handled separately, NOT in this map
    21: "key_ctrl",
    23: "key_enter",
    24: "key_delete",
    25: "key_tab",
    26: "key_space",
}

# ---------------------------------------------------------------------------
# Shift-modified actions  (button_id → action when Shift is held)
# ---------------------------------------------------------------------------
ENTERPRISE_DEFAULT_SHIFT_MAP: dict[int, str] = {
    14: "view_bottom",  # Shift + Top
    15: "view_left",  # Shift + Right
    16: "view_back",  # Shift + Front
    17: "rotate_view_ccw",  # Shift + Rotate → rotate left
    27: "save_view_1",  # Shift + V1
    28: "save_view_2",  # Shift + V2
    29: "save_view_3",  # Shift + V3
    30: "view_iso2",  # Shift + ISO1
}

# The Shift key button ID — tracked as a held modifier, never dispatched as an action
SHIFT_BUTTON_ID = 20

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
    {"label": "", "action": "noop"},
    {"label": "", "action": "noop"},
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


def get_hotkeys() -> list[dict]:
    config = load_config()
    hotkeys = [dict(h) for h in DEFAULT_HOTKEYS]
    for i, entry in enumerate(config.get("hotkeys", [])[:12]):
        hotkeys[i] = {
            "label": str(entry.get("label", ""))[:4].upper(),
            "action": str(entry.get("action", "noop")),
        }
    return hotkeys
