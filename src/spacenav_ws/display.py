"""SpaceMouse Enterprise LCD display driver.

The Enterprise has a 640×150 pixel colour LCD connected via a vendor-specific
USB interface (interface 0, bulk OUT endpoint 0x01).

Protocol (reverse-engineered by TheHoodedFoot/SpaceLCD):
  1. Render a 640×150 RGB image with Pillow.
  2. Convert pixels to BGR565 (16-bit, little-endian).
  3. Raw-DEFLATE compress the bitmap (no zlib wrapper; wbits=-15).
  4. Prepend a 512-byte header:
       [0x11, 0x0F, len_lo, len_hi, <zeros to 0x1B>,
        len_lo (at 0x1C), len_hi (at 0x1D), <zeros to 0x1FF>]
  5. Send the whole thing to EP 0x01 in 64-byte bulk chunks.

Requires:
  pip install pyusb pillow numpy
  # Also: sudo apt install libusb-1.0-0

Udev rule (create /etc/udev/rules.d/99-spacemouse.rules):
  SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0666"
  SUBSYSTEM=="usb",    ATTRS{idVendor}=="256f", MODE="0666"
Then: sudo udevadm control --reload-rules && sudo udevadm trigger
"""

from __future__ import annotations

import logging
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------
VENDOR_ID  = 0x256F   # 3Dconnexion
PRODUCT_ID = 0xC633   # SpaceMouse Enterprise

DISPLAY_W = 640
DISPLAY_H = 150
BITMAP_BYTES = DISPLAY_W * DISPLAY_H * 2   # 192 000 bytes (BGR565)

_HEADER_SIZE  = 512
_EFFECT_CUT   = 0x11   # instant display update
_USB_EP       = 0x01   # bulk OUT
_USB_IFACE    = 0      # vendor-specific interface
_USB_TIMEOUT  = 1000   # ms
_USB_CHUNK    = 64

# ---------------------------------------------------------------------------
# Font / layout
# ---------------------------------------------------------------------------
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]
_FONT_SIZE_GRID = 22    # for 4-char labels in ~106×75 cells
_FONT_SIZE_MSG  = 40    # for full-screen status messages

_COLS    = 6
_ROWS    = 2
_CELL_W  = DISPLAY_W // _COLS   # ~106
_CELL_H  = DISPLAY_H // _ROWS   # 75


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _get_font(size: int):
    """Return a Pillow ImageFont, trying system fonts then the built-in."""
    try:
        from PIL import ImageFont
        for path in _FONT_PATHS:
            if Path(path).exists():
                return ImageFont.truetype(path, size)
        # Pillow ≥ 10 built-in scalable font
        return ImageFont.load_default(size=size)
    except Exception:
        from PIL import ImageFont
        return ImageFont.load_default()


def _img_to_bgr565(img) -> bytes:
    """Convert a PIL RGB image to raw BGR565 bytes (little-endian)."""
    arr = np.array(img, dtype=np.uint16)    # (H, W, 3)
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    word = ((b >> 3) << 11) | ((g >> 2) << 5) | (r >> 3)
    return word.astype("<u2").tobytes()


def _compress(raw: bytes) -> bytes:
    """Raw-DEFLATE compress (no zlib header/trailer)."""
    c = zlib.compressobj(level=6, wbits=-15, strategy=zlib.Z_FIXED)
    return c.compress(raw) + c.flush()


def _build_packet(bitmap: bytes) -> bytes:
    """Wrap a BGR565 bitmap in the 512-byte header and compress it."""
    compressed = _compress(bitmap)
    if len(compressed) > 0xFFFF:
        raise RuntimeError(f"Display: compressed bitmap too large ({len(compressed)} B)")
    header = bytearray(_HEADER_SIZE)
    header[0x00] = _EFFECT_CUT
    header[0x01] = 0x0F
    header[0x02] = len(compressed) & 0xFF
    header[0x03] = (len(compressed) >> 8) & 0xFF
    header[0x1C] = header[0x02]
    header[0x1D] = header[0x03]
    return bytes(header) + compressed


_ICON_SIZE = 36   # px — icon rendered at this size, centred in top portion of cell
_LABEL_FONT_SIZE = 14


def _svg_to_pil(svg_bytes: bytes, size: int):
    """Render SVG bytes to a square PIL RGBA image using cairosvg."""
    try:
        import cairosvg
        from PIL import Image
        import io
        png = cairosvg.svg2png(bytestring=svg_bytes, output_width=size, output_height=size)
        return Image.open(io.BytesIO(png)).convert("RGBA")
    except Exception:
        return None


def render_hotkey_grid(hotkeys: list[dict]) -> bytes:
    """Render a 6×2 grid of icon+label cells and return a BGR565 packet."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (DISPLAY_W, DISPLAY_H), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    label_font = _get_font(_LABEL_FONT_SIZE)

    # Grid lines
    for col in range(1, _COLS):
        x = col * _CELL_W
        draw.line([(x, 0), (x, DISPLAY_H - 1)], fill=(60, 60, 60), width=1)
    for row in range(1, _ROWS):
        y = row * _CELL_H
        draw.line([(0, y), (DISPLAY_W - 1, y)], fill=(60, 60, 60), width=1)

    for i, hk in enumerate(hotkeys[:12]):
        label = str(hk.get("label", "")).upper()
        col = i % _COLS
        row = i // _COLS
        cell_x = col * _CELL_W
        cell_y = row * _CELL_H
        cx = cell_x + _CELL_W // 2

        svg_data = hk.get("svg")  # base64-decoded SVG bytes, set by controller
        if svg_data:
            icon = _svg_to_pil(svg_data, _ICON_SIZE)
            if icon:
                # Tint white/light pixels to match foreground colour
                icon_rgb = Image.new("RGB", icon.size, (220, 220, 220))
                icon_rgb.putalpha(icon.split()[3])  # keep alpha
                ix = cx - _ICON_SIZE // 2
                iy = cell_y + 4
                img.paste(icon_rgb, (ix, iy), mask=icon.split()[3])

        if label:
            bbox = label_font.getbbox(label)
            tw = bbox[2] - bbox[0]
            lx = cx - tw // 2 - bbox[0]
            ly = cell_y + _CELL_H - _LABEL_FONT_SIZE - 5
            draw.text((lx, ly), label, fill=(255, 255, 255), font=label_font)

    return _build_packet(_img_to_bgr565(img))


def render_message(text: str) -> bytes:
    """Render a centred status message and return a BGR565 packet."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (DISPLAY_W, DISPLAY_H), color=(0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = _get_font(_FONT_SIZE_MSG)
    text = text.upper()
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (DISPLAY_W - tw) // 2 - bbox[0]
    y = (DISPLAY_H - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return _build_packet(_img_to_bgr565(img))


# ---------------------------------------------------------------------------
# USB display driver
# ---------------------------------------------------------------------------

class EnterpriseDisplay:
    """Controls the SpaceMouse Enterprise colour LCD via USB bulk transfer.

    Uses pyusb to communicate directly with interface 0 (vendor-specific).
    Gracefully no-ops when pyusb/Pillow are missing or the device is absent.
    """

    def __init__(self) -> None:
        self._handle = None
        self._open()

    def _open(self) -> None:
        try:
            import usb.core
            import usb.util
        except ImportError:
            logging.info(
                "Display: 'pyusb' not installed — display disabled.\n"
                "  Install with: uv add pyusb"
            )
            return

        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            logging.info(
                "Display: 'pillow' not installed — display disabled.\n"
                "  Install with: uv add pillow"
            )
            return

        dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
        if dev is None:
            logging.warning(
                "Display: SpaceMouse Enterprise not found (VID=0x%04X PID=0x%04X).\n"
                "  Check USB connection and udev rule.",
                VENDOR_ID, PRODUCT_ID,
            )
            return

        try:
            # Interface 0 is vendor-specific with no kernel driver — claim it.
            # We deliberately do NOT touch interface 1 (HID / spacenavd).
            dev.set_configuration()
        except Exception:
            pass  # configuration may already be set

        try:
            if dev.is_kernel_driver_active(_USB_IFACE):
                dev.detach_kernel_driver(_USB_IFACE)
            usb.util.claim_interface(dev, _USB_IFACE)
            self._handle = dev
            logging.info(
                "Display: opened SpaceMouse Enterprise LCD (interface %d, EP 0x%02X)",
                _USB_IFACE, _USB_EP,
            )
        except Exception as exc:
            logging.warning("Display: could not claim USB interface — %s", exc)

    def close(self) -> None:
        if self._handle is not None:
            try:
                import usb.util
                usb.util.release_interface(self._handle, _USB_IFACE)
                usb.util.dispose_resources(self._handle)
            except Exception:
                pass
            self._handle = None

    @property
    def available(self) -> bool:
        return self._handle is not None

    # ------------------------------------------------------------------
    # Raw send
    # ------------------------------------------------------------------

    def _send(self, packet: bytes) -> bool:
        """Send a pre-built display packet via USB bulk transfer."""
        if self._handle is None:
            return False
        try:
            data = bytearray(packet)
            offset = 0
            while offset < len(data):
                chunk = data[offset: offset + _USB_CHUNK]
                self._handle.write(_USB_EP, chunk, _USB_TIMEOUT)
                offset += _USB_CHUNK
            return True
        except Exception as exc:
            logging.debug("Display: write failed — %s", exc)
            self._handle = None
            return False

    # ------------------------------------------------------------------
    # High-level update methods
    # ------------------------------------------------------------------

    def show_hotkeys(self, hotkeys: list[dict]) -> None:
        """Display the 12 hotkey labels in a 4×3 grid."""
        try:
            self._send(render_hotkey_grid(hotkeys))
        except Exception as exc:
            logging.debug("Display: show_hotkeys failed — %s", exc)

    def show_message(self, text: str) -> None:
        """Display a centred status message."""
        try:
            self._send(render_message(text))
        except Exception as exc:
            logging.debug("Display: show_message failed — %s", exc)

    def clear(self) -> None:
        """Blank the display (send black bitmap)."""
        from PIL import Image
        img = Image.new("RGB", (DISPLAY_W, DISPLAY_H), color=(0, 0, 0))
        try:
            self._send(_build_packet(_img_to_bgr565(img)))
        except Exception as exc:
            logging.debug("Display: clear failed — %s", exc)
