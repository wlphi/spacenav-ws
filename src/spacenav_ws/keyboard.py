"""Onshape keyboard shortcut injection via Linux uinput (Wayland-compatible).

This is the fallback command-invocation path used when WAMP property writes are
not accepted by Onshape.  A virtual keyboard device is created through
/dev/uinput (below the display server, works on both X11 and Wayland).

Setup (one-time):
    echo 'KERNEL=="uinput", MODE="0660", GROUP="input"' \\
        | sudo tee /etc/udev/rules.d/99-uinput.rules
    sudo udevadm control --reload-rules && sudo udevadm trigger
    sudo usermod -a -G input $USER   # log out and back in afterwards

Also install: uv add evdev  (already done if spacenav-ws deps include it)

The shortcut map can be extended via ~/.config/spacenav-ws/shortcuts.json:
    {
        "Part Studio-myCustomTool": "shift+t",
        "myLeafId": "ctrl+k"
    }
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Onshape keyboard shortcuts  (command leaf-ID → shortcut string)
# Source: https://cad.onshape.com/help/Content/shortcut_keys.htm
# ---------------------------------------------------------------------------

SHORTCUTS: dict[str, str] = {
    # ── Part Studio ──────────────────────────────────────────────────────
    # These are Onshape's documented shortcuts (cad.onshape.com/help/Content/shortcut_keys.htm)
    "extrude": "shift+e",
    "revolve": "shift+w",
    "newSketch": "shift+s",
    "sketch": "shift+s",
    "fillet": "shift+f",
    # No default Onshape shortcuts for the following — assign them in
    # Onshape Preferences → Keyboard Shortcuts, then mirror here:
    "chamfer": "shift+alt+c",
    "draft": "shift+alt+d",
    "shell": "shift+h",  # user-assigned example
    "mirror": "shift+m",  # user-assigned example
    "pattern": "shift+p",  # user-assigned example
    "sweep": "shift+a",  # user-assigned
    "loft": "shift+k",  # user-assigned
    "thicken": "shift+o",  # user-assigned
    "enclose": "shift+u",  # user-assigned
    "faceBlend": "shift+j",  # user-assigned
    "bodyDraft": "shift+g",  # user-assigned
    # ── Sketch (real IDs captured from Onshape command tree) ─────────────
    "LINESEGMENT": "l",
    "MIDPOINTLINE": "l",  # midpoint line, same tool family
    # Circles — "C" = Center point circle; 3-point perimeter circle has no default
    "CIRCLE_CENTER_RADIUS": "c",  # "Center point circle"
    "CIRCLE": "c",  # generic fallback
    # Rectangles
    "RECTANGLE_CENTER": "r",  # "Center point rectangle"  (Onshape default: R)
    "RECTANGLE_TWO_CORNERS": "g",  # "Corner rectangle"        (Onshape default: G)
    # Arcs
    "ARC_START_END_RADIUS": "a",  # "3 point arc"
    "ARC_TANGENT": "a",  # "Tangent arc" — no default, use A as fallback
    "CENTER_ARC": "a",  # "Center point arc" — no default, use A as fallback
    # Sketch constraints & tools
    "DIMENSION": "d",
    "TRIM": "m",  # Onshape uses M (not T) for trim
    "EXTEND": "x",
    "OFFSET": "o",
    "SPLIT": "x",  # no default; X (extend) as fallback
    "TOGGLE_CONSTRUCTION": "q",  # "Construction"
    "CONSTRUCTION": "q",
    "HORIZONTAL": "h",
    "VERTICAL": "v",
    "EQUAL": "e",
    "PARALLEL": "b",
    "TANGENT": "t",
    "COINCIDENT": "i",
    "PERPENDICULAR": "shift+l",
    "MIDPOINT": "shift+m",
    "TEXT_RECTANGLE_TWO_CORNERS": "shift+t",  # "Text"
    "USE": "u",  # "Use / project" (Onshape default: U)
    "FILLET": "shift+f",  # Sketch fillet (Onshape default: Shift+F)
    # Confirmed Onshape shortcuts for tools with no single-letter default
    "RECTANGLE_ALIGNED": "shift+alt+r",  # Aligned rectangle
    "CIRCLE_PERIMETER": "alt+k",  # 3-point circle
    "SLOT": "j",  # Slot
    "SKETCHMIRROR": "alt+i",  # Mirror
    "SKETCHLPATTERN": "alt+q",  # Linear pattern
    "SKETCHCPATTERN": "alt+g",  # Circular pattern
    "SPLINE": "alt+v",  # Spline
    "ELLIPSE": "shift+b",  # Ellipse
    # ── Assembly ─────────────────────────────────────────────────────────
    # Tree IDs confirmed from live svg_cache logs (active='Assembly')
    "mate_FASTENED": "m",                       # Onshape default
    "insertPartOrAssembly": "i",                # Onshape default (tree ID: Assembly-insertPartOrAssembly)
    "TOGGLE_ASSEMBLY_SNAP_MODE": "shift+s",     # user-assigned
    "TOGGLE_SHOW_MATES_ON_SELECTION_MODE": "j", # Onshape default
    # User-assigned shortcuts:
    "geometryMate_TANGENT": "t",
    "widthMate_WIDTH": "shift+w",               # tree ID: widthMate_WIDTH
    # Remaining mate types — assign in Onshape Preferences then uncomment:
    # "mate_REVOLUTE": "shift+alt+r",
    # "mate_CYLINDRICAL": "shift+alt+y",
    # "mate_SLIDER": "shift+alt+s",
    # "mate_PLANAR": "shift+alt+p",
    # "mate_BALL": "shift+alt+b",
    # "mate_PARALLEL": "shift+alt+l",
    # "mate_PIN_SLOT": "shift+alt+n",
    # ── General ──────────────────────────────────────────────────────────
    "fit": "f",
    "zoomIn": "shift+z",
    "zoomOut": "z",
}

# Load user-defined overrides / additions
_SHORTCUTS_CONFIG = Path.home() / ".config" / "spacenav-ws" / "shortcuts.json"
if _SHORTCUTS_CONFIG.exists():
    try:
        SHORTCUTS.update(json.loads(_SHORTCUTS_CONFIG.read_text()))
        logging.info("keyboard: loaded extra shortcuts from %s", _SHORTCUTS_CONFIG)
    except Exception:
        logging.warning("keyboard: failed to parse %s", _SHORTCUTS_CONFIG)

# ---------------------------------------------------------------------------
# uinput device (lazily opened)
# ---------------------------------------------------------------------------

_uinput = None
_uinput_error_logged = False


def _get_uinput():
    global _uinput, _uinput_error_logged
    if _uinput is not None:
        return _uinput
    try:
        from evdev import UInput

        _uinput = UInput(name="spacenav-ws-kbd")
        logging.info("keyboard: uinput virtual keyboard created (/dev/uinput)")
        return _uinput
    except PermissionError:
        if not _uinput_error_logged:
            logging.warning(
                "keyboard: no access to /dev/uinput — keyboard fallback disabled.\n"
                "  Fix with:\n"
                '    echo \'KERNEL=="uinput", MODE="0660", GROUP="input"\' \\\n'
                "        | sudo tee /etc/udev/rules.d/99-uinput.rules\n"
                "    sudo udevadm control --reload-rules && sudo udevadm trigger\n"
                "    sudo usermod -a -G input $USER   # then log out & in\n"
                "  After that, restart spacenav-ws."
            )
            _uinput_error_logged = True
    except ImportError:
        if not _uinput_error_logged:
            logging.warning("keyboard: 'evdev' not installed — run: uv add evdev")
            _uinput_error_logged = True
    except Exception as exc:
        if not _uinput_error_logged:
            logging.warning("keyboard: uinput init failed — %s", exc)
            _uinput_error_logged = True
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def inject_shortcut(command_id: str) -> bool:
    """Inject the keyboard shortcut for an Onshape command ID.

    Tries first the full ID (e.g. 'Part Studio-extrude'), then the leaf part
    (e.g. 'extrude').  Returns True on success.
    """
    leaf = command_id.split("-", 1)[-1] if "-" in command_id else command_id
    shortcut = SHORTCUTS.get(command_id) or SHORTCUTS.get(leaf)
    if shortcut is None:
        logging.warning("keyboard: no shortcut for %r — add it to %s", command_id, _SHORTCUTS_CONFIG)
        return False
    return _send_keys(shortcut, command_id)


def _send_keys(shortcut: str, label: str = "") -> bool:
    """Parse 'shift+e' style string and inject via uinput."""
    try:
        from evdev import ecodes as E
    except ImportError:
        return False

    ui = _get_uinput()
    if ui is None:
        return False

    parts = shortcut.lower().split("+")
    mods: list[int] = []
    keys: list[int] = []

    mod_map = {
        "shift": E.KEY_LEFTSHIFT,
        "ctrl": E.KEY_LEFTCTRL,
        "control": E.KEY_LEFTCTRL,
        "alt": E.KEY_LEFTALT,
        "meta": E.KEY_LEFTMETA,
    }

    for part in parts:
        if part in mod_map:
            mods.append(mod_map[part])
        else:
            kc = getattr(E, f"KEY_{part.upper()}", None)
            if kc is None:
                logging.warning("keyboard: unknown key code %r in shortcut %r", part, shortcut)
                return False
            keys.append(kc)

    if not keys:
        # Standalone modifier (e.g. "alt", "ctrl") — press and release it alone
        if mods:
            keys, mods = mods, []
        else:
            logging.warning("keyboard: no main key in shortcut %r", shortcut)
            return False

    try:
        for mod in mods:
            ui.write(E.EV_KEY, mod, 1)
        for key in keys:
            ui.write(E.EV_KEY, key, 1)
        ui.syn()
        time.sleep(0.05)
        for key in reversed(keys):
            ui.write(E.EV_KEY, key, 0)
        for mod in reversed(mods):
            ui.write(E.EV_KEY, mod, 0)
        ui.syn()
        logging.warning("keyboard: %r → %r", label or shortcut, shortcut)
        return True
    except Exception as exc:
        logging.warning("keyboard: write failed — %s", exc)
        return False
