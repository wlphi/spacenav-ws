"""SpaceMouse Enterprise OLED display driver.

The Enterprise has a 128×32 pixel monochrome OLED connected over USB HID.
spacenavd holds the 6DOF input interface, but the display lives on a separate
HID interface that we can open independently.

Requires the ``hid`` package (hidapi Python bindings):
    pip install hid               # or: uv add hid

Udev rule (create /etc/udev/rules.d/99-spacemouse.rules):
    SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"
Then: sudo udevadm control --reload-rules && sudo udevadm trigger

Display protocol (SpaceMouse Enterprise, reverse-engineered)
-------------------------------------------------------------
The display is updated by writing HID output reports with report-ID 0x10.
The 512-byte bitmap (128 cols × 32 rows, 1 bit per pixel, row-major) is sent
in 8 strips of 64 bytes (covering 4 rows each), each preceded by a 5-byte
header:

    [0x10, strip_index, 0x00, 0x00, 0x00, ...64 bytes...]

If the protocol byte layout ever proves wrong, set ``DISPLAY_DEBUG = True``
and read the hex output to identify the correct framing.

Bitmap layout (row-major, MSB = leftmost pixel)
------------------------------------------------
    byte at index ``row * 16 + col // 8``
    bit  ``7 - (col % 8)``   → pixel at (col, row)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------
VENDOR_ID = 0x256F    # 3Dconnexion
PRODUCT_ID = 0xC631   # SpaceMouse Enterprise

DISPLAY_W = 128
DISPLAY_H = 32
BITMAP_BYTES = DISPLAY_W * DISPLAY_H // 8   # 512

# HID report parameters
_REPORT_ID = 0x10
_STRIP_ROWS = 4                              # rows per HID write
_STRIP_DATA = DISPLAY_W * _STRIP_ROWS // 8  # 64 bytes of pixels per strip
_N_STRIPS = DISPLAY_H // _STRIP_ROWS        # 8 strips total

# ---------------------------------------------------------------------------
# 5×7 bitmap font (column-major, bit-0 = top row)
# Characters: space, digits 0-9, uppercase A-Z, a few symbols
# ---------------------------------------------------------------------------
_FONT: dict[str, list[int]] = {
    " ":  [0x00, 0x00, 0x00, 0x00, 0x00],
    "!":  [0x00, 0x00, 0x5F, 0x00, 0x00],
    "+":  [0x08, 0x08, 0x3E, 0x08, 0x08],
    "-":  [0x08, 0x08, 0x08, 0x08, 0x08],
    "/":  [0x20, 0x10, 0x08, 0x04, 0x02],
    "0":  [0x3E, 0x51, 0x49, 0x45, 0x3E],
    "1":  [0x00, 0x42, 0x7F, 0x40, 0x00],
    "2":  [0x42, 0x61, 0x51, 0x49, 0x46],
    "3":  [0x21, 0x41, 0x45, 0x4B, 0x31],
    "4":  [0x18, 0x14, 0x12, 0x7F, 0x10],
    "5":  [0x27, 0x45, 0x45, 0x45, 0x39],
    "6":  [0x3C, 0x4A, 0x49, 0x49, 0x30],
    "7":  [0x01, 0x71, 0x09, 0x05, 0x03],
    "8":  [0x36, 0x49, 0x49, 0x49, 0x36],
    "9":  [0x06, 0x49, 0x49, 0x29, 0x1E],
    "A":  [0x7C, 0x12, 0x11, 0x12, 0x7C],
    "B":  [0x7F, 0x49, 0x49, 0x49, 0x36],
    "C":  [0x3E, 0x41, 0x41, 0x41, 0x22],
    "D":  [0x7F, 0x41, 0x41, 0x22, 0x1C],
    "E":  [0x7F, 0x49, 0x49, 0x49, 0x41],
    "F":  [0x7F, 0x09, 0x09, 0x09, 0x01],
    "G":  [0x3E, 0x41, 0x49, 0x49, 0x7A],
    "H":  [0x7F, 0x08, 0x08, 0x08, 0x7F],
    "I":  [0x00, 0x41, 0x7F, 0x41, 0x00],
    "J":  [0x20, 0x40, 0x41, 0x3F, 0x01],
    "K":  [0x7F, 0x08, 0x14, 0x22, 0x41],
    "L":  [0x7F, 0x40, 0x40, 0x40, 0x40],
    "M":  [0x7F, 0x02, 0x04, 0x02, 0x7F],
    "N":  [0x7F, 0x04, 0x08, 0x10, 0x7F],
    "O":  [0x3E, 0x41, 0x41, 0x41, 0x3E],
    "P":  [0x7F, 0x09, 0x09, 0x09, 0x06],
    "Q":  [0x3E, 0x41, 0x51, 0x21, 0x5E],
    "R":  [0x7F, 0x09, 0x19, 0x29, 0x46],
    "S":  [0x26, 0x49, 0x49, 0x49, 0x32],
    "T":  [0x01, 0x01, 0x7F, 0x01, 0x01],
    "U":  [0x3F, 0x40, 0x40, 0x40, 0x3F],
    "V":  [0x1F, 0x20, 0x40, 0x20, 0x1F],
    "W":  [0x3F, 0x40, 0x38, 0x40, 0x3F],
    "X":  [0x63, 0x14, 0x08, 0x14, 0x63],
    "Y":  [0x07, 0x08, 0x70, 0x08, 0x07],
    "Z":  [0x61, 0x51, 0x49, 0x45, 0x43],
}

_CHAR_W = 5  # pixels per glyph column
_CHAR_H = 7  # pixels per glyph row
_CHAR_STEP = 6  # advance = glyph width + 1px gap


# ---------------------------------------------------------------------------
# Low-level bitmap helpers
# ---------------------------------------------------------------------------

def _set_pixel(bmp: bytearray, x: int, y: int) -> None:
    if 0 <= x < DISPLAY_W and 0 <= y < DISPLAY_H:
        bmp[y * (DISPLAY_W >> 3) + (x >> 3)] |= 1 << (7 - (x & 7))


def draw_char(bmp: bytearray, char: str, x: int, y: int) -> None:
    """Render one 5×7 glyph at (x, y) into ``bmp``."""
    cols = _FONT.get(char.upper(), _FONT[" "])
    for cx, col_bits in enumerate(cols):
        for cy in range(_CHAR_H):
            if col_bits & (1 << cy):
                _set_pixel(bmp, x + cx, y + cy)


def draw_string(bmp: bytearray, text: str, x: int, y: int, max_chars: int = 21) -> None:
    """Render a string starting at (x, y); each char advances _CHAR_STEP pixels."""
    for i, ch in enumerate(text[:max_chars]):
        draw_char(bmp, ch, x + i * _CHAR_STEP, y)


# ---------------------------------------------------------------------------
# Hotkey-label display layout
# ---------------------------------------------------------------------------
# 4-column × 3-row grid on a 128×32 display:
#
#   ┌──────┬──────┬──────┬──────┐   row 0  (y 0..9)
#   │  H1  │  H2  │  H3  │  H4  │
#   ├──────┼──────┼──────┼──────┤   row 1  (y 10..20)
#   │  H5  │  H6  │  H7  │  H8  │
#   ├──────┼──────┼──────┼──────┤   row 2  (y 21..31)
#   │  H9  │  H10 │  H11 │  H12 │
#   └──────┴──────┴──────┴──────┘
#
# Cell = 32px wide × 10px tall (7 font + 3 padding/border).
# Labels are truncated to 4 chars and centred horizontally.

_CELL_W = 32
_CELL_H = 11   # row stride (10 usable + 1 divider line below)
_COLS = 4
_ROWS = 3


def render_hotkey_grid(labels: list[str]) -> bytearray:
    """Return a 512-byte 128×32 bitmap with the 12 hotkey labels."""
    bmp = bytearray(BITMAP_BYTES)

    # Horizontal dividers at y=10, y=21
    for row_div in (10, 21):
        for x in range(DISPLAY_W):
            _set_pixel(bmp, x, row_div)

    # Vertical dividers at x=32, 64, 96
    for col_div in (32, 64, 96):
        for y in range(DISPLAY_H):
            _set_pixel(bmp, col_div, y)

    for i, raw_label in enumerate(labels[:12]):
        label = str(raw_label).upper()[:4]
        col = i % _COLS
        row = i // _COLS

        cell_x = col * _CELL_W
        cell_y = row * _CELL_H

        # Usable cell interior: x in [cell_x+1 .. cell_x+30], y in [cell_y .. cell_y+9]
        usable_w = _CELL_W - 1  # 31px (leave 1px for the divider)
        text_w = len(label) * _CHAR_STEP - (1 if label else 0)
        text_x = cell_x + 1 + max(0, (usable_w - text_w) // 2)
        text_y = cell_y + 1  # 1px top padding

        draw_string(bmp, label, text_x, text_y)

    return bmp


def render_centered_message(text: str) -> bytearray:
    """Return a 512-byte bitmap with ``text`` centred on the display."""
    bmp = bytearray(BITMAP_BYTES)
    text = text.upper()[:21]
    text_w = len(text) * _CHAR_STEP - (1 if text else 0)
    x = max(0, (DISPLAY_W - text_w) // 2)
    y = (DISPLAY_H - _CHAR_H) // 2
    draw_string(bmp, text, x, y)
    return bmp


# ---------------------------------------------------------------------------
# HID display driver
# ---------------------------------------------------------------------------

class EnterpriseDisplay:
    """Controls the SpaceMouse Enterprise OLED.

    All public methods are safe to call even when the display is unavailable
    (missing udev rule, ``hid`` package not installed, wrong device, etc.).
    Errors are logged at DEBUG level so they don't spam the console.

    The ``hid`` package is imported lazily so the rest of the bridge works
    without it if display support is not needed.
    """

    def __init__(self) -> None:
        self._dev = None
        self._open()

    # ------------------------------------------------------------------
    # Device lifecycle
    # ------------------------------------------------------------------

    def _open(self) -> None:
        try:
            import hid  # type: ignore[import-untyped]
        except ImportError:
            logging.info(
                "Display: 'hid' package not installed — display disabled.\n"
                "  Install with:  pip install hid   (or: uv add hid)"
            )
            return

        devices = hid.enumerate(VENDOR_ID, PRODUCT_ID)
        if not devices:
            logging.warning(
                "Display: SpaceMouse Enterprise not found (VID=0x%04X PID=0x%04X).\n"
                "  Check USB connection and udev rule.",
                VENDOR_ID, PRODUCT_ID,
            )
            return

        # The Enterprise exposes multiple HID interfaces; spacenavd holds the
        # primary (6DOF+buttons) interface. We try each path until we find one
        # that accepts display reports.  Typically the display lives on the
        # interface with a vendor-specific usage page (0xFF00).
        candidates = sorted(
            devices,
            key=lambda d: (0 if d.get("usage_page", 0) == 0xFF00 else 1),
        )
        for info in candidates:
            try:
                import hid
                dev = hid.device()
                dev.open_path(info["path"])
                self._dev = dev
                logging.info(
                    "Display: opened Enterprise HID interface at %s (usage_page=0x%04X)",
                    info["path"].decode(errors="replace") if isinstance(info["path"], bytes) else info["path"],
                    info.get("usage_page", 0),
                )
                return
            except Exception as exc:
                logging.debug("Display: could not open %s — %s", info.get("path"), exc)

        logging.warning(
            "Display: could not open any HID interface for the Enterprise.\n"
            "  Make sure the udev rule is in place and run:\n"
            "    sudo udevadm control --reload-rules && sudo udevadm trigger"
        )

    def close(self) -> None:
        if self._dev is not None:
            try:
                self._dev.close()
            except Exception:
                pass
            self._dev = None

    @property
    def available(self) -> bool:
        return self._dev is not None

    # ------------------------------------------------------------------
    # Raw bitmap send
    # ------------------------------------------------------------------

    def _send(self, bmp: bytearray) -> bool:
        """Write a 512-byte bitmap to the display.  Returns True on success.

        Protocol: 8 output reports, each covering 4 pixel rows (64 bytes).
        Report format: [report_id=0x10, strip_index, 0x00, 0x00, 0x00, ...64px bytes...]
        """
        if self._dev is None:
            return False
        try:
            for strip in range(_N_STRIPS):
                start = strip * _STRIP_DATA
                chunk = bmp[start: start + _STRIP_DATA]
                # write() prepends the report ID automatically on most platforms
                report = bytes([_REPORT_ID, strip, 0x00, 0x00, 0x00]) + bytes(chunk)
                self._dev.write(report)
            return True
        except Exception as exc:
            logging.debug("Display: write failed — %s", exc)
            # Mark unavailable so subsequent calls don't keep trying
            self._dev = None
            return False

    # ------------------------------------------------------------------
    # High-level update methods
    # ------------------------------------------------------------------

    def show_hotkeys(self, hotkeys: list[dict]) -> None:
        """Display the 12 hotkey labels in a 4×3 grid."""
        labels = [h.get("label", "") for h in hotkeys[:12]]
        self._send(render_hotkey_grid(labels))

    def show_message(self, text: str) -> None:
        """Display a centred status message (max ~21 chars)."""
        self._send(render_centered_message(text))

    def clear(self) -> None:
        """Blank the display."""
        self._send(bytearray(BITMAP_BYTES))
