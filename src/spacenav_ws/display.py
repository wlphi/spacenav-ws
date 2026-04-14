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

import numpy as np

# ---------------------------------------------------------------------------
# Hardware constants
# ---------------------------------------------------------------------------
VENDOR_ID = 0x256F  # 3Dconnexion
PRODUCT_ID = 0xC633  # SpaceMouse Enterprise

DISPLAY_W = 640
DISPLAY_H = 150
BITMAP_BYTES = DISPLAY_W * DISPLAY_H * 2  # 192 000 bytes (BGR565)

_HEADER_SIZE = 512
_EFFECT_CUT = 0x11  # instant display update
_USB_EP = 0x01  # bulk OUT
_USB_IFACE = 0  # vendor-specific interface
_USB_TIMEOUT = 1000  # ms
_USB_CHUNK = 64

# ---------------------------------------------------------------------------
# Font / layout
# ---------------------------------------------------------------------------
_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
]
_FONT_SIZE_GRID = 22  # for 4-char labels in ~106×75 cells
_FONT_SIZE_MSG = 40  # for full-screen status messages

_COLS = 6
_ROWS = 2
_CELL_W = DISPLAY_W // _COLS  # ~106
_CELL_H = DISPLAY_H // _ROWS  # 75


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
    arr = np.array(img, dtype=np.uint16)  # (H, W, 3)
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


_ICON_SIZE = 44  # px — larger icon for better detail
_LABEL_FONT_SIZE = 13  # slightly smaller to give more room to icon

# ── Colour palette (dark theme) ──────────────────────────────────────────────
_C_BG = (10, 12, 17)  # overall background — near-black
_C_CELL = (22, 26, 36)  # per-cell dark-navy fill
_C_GRID = (48, 54, 70)  # grid dividers — muted slate-blue
_C_LABEL = (185, 195, 215)  # label text — cool off-white
_C_ACCENT = (55, 70, 100)  # top-edge accent stripe per cell


def _svg_to_pil(svg_bytes: bytes, size: int):
    """Render SVG bytes to a square PIL RGBA image using resvg (system CLI).

    Install on Arch/Manjaro:  sudo pacman -S resvg
    """
    import io
    import shutil
    import subprocess

    from PIL import Image

    if shutil.which("resvg") is None:
        logging.debug("resvg not found — icons disabled. Install with: sudo pacman -S resvg")
        return None
    try:
        result = subprocess.run(
            ["resvg", "--width", str(size), "--height", str(size), "-", "-c"],
            input=svg_bytes,
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            logging.debug("resvg failed: %s", result.stderr.decode(errors="replace").strip())
            return None
        return Image.open(io.BytesIO(result.stdout)).convert("RGBA")
    except Exception as exc:
        logging.debug("_svg_to_pil failed: %s", exc)
        return None


def _adapt_icon(icon):
    """Make a light-background SVG icon legible on a dark display.

    Onshape icons are designed for white UIs: they have coloured fills and
    dark/black outlines on a transparent background.  On our black LCD the
    outlines would vanish.  This function:
      1. Boosts dark pixels toward white (outlines become visible).
      2. Slightly desaturates colours for a muted, cohesive palette.
      3. Leaves already-bright / saturated pixels mostly unchanged.
    """
    arr = np.array(icon, dtype=np.float32) / 255.0
    r, g, b, a = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2], arr[:, :, 3]

    # Perceptual luminance
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Brightness boost: smoothly lifts dark pixels toward bright.
    # boost=1 → fully white; boost=0 → original colour unchanged.
    # Curve: pixels below lum≈0.45 get lifted; above that, untouched.
    boost = np.clip(1.0 - lum / 0.45, 0.0, 1.0) ** 1.6

    r2 = r + boost * (1.0 - r)
    g2 = g + boost * (1.0 - g)
    b2 = b + boost * (1.0 - b)

    # Slight desaturation (blend 20 % toward grey) → "muted" palette
    grey = 0.299 * r2 + 0.587 * g2 + 0.114 * b2
    sat = 0.80  # keep 80 % of colour, 20 % grey
    r2 = r2 * sat + grey * (1.0 - sat)
    g2 = g2 * sat + grey * (1.0 - sat)
    b2 = b2 * sat + grey * (1.0 - sat)

    out = np.stack([r2, g2, b2, a], axis=2)
    out = (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
    from PIL import Image

    return Image.fromarray(out, "RGBA")


def render_hotkey_grid(hotkeys: list[dict]) -> bytes:
    """Render a 6×2 grid of icon+label cells and return a BGR565 packet."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (DISPLAY_W, DISPLAY_H), color=_C_BG)
    draw = ImageDraw.Draw(img)

    # ── Per-cell background + top-edge accent ──────────────────────────
    for row in range(_ROWS):
        for col in range(_COLS):
            x0 = col * _CELL_W + 1
            y0 = row * _CELL_H + 1
            x1 = (col + 1) * _CELL_W - 1
            y1 = (row + 1) * _CELL_H - 1
            draw.rectangle([x0, y0, x1, y1], fill=_C_CELL)
            # thin accent stripe at top of each cell
            draw.line([(x0, y0), (x1, y0)], fill=_C_ACCENT, width=2)

    # ── Grid dividers ───────────────────────────────────────────────────
    for col in range(1, _COLS):
        x = col * _CELL_W
        draw.line([(x, 0), (x, DISPLAY_H - 1)], fill=_C_GRID, width=1)
    for row in range(1, _ROWS):
        y = row * _CELL_H
        draw.line([(0, y), (DISPLAY_W - 1, y)], fill=_C_GRID, width=1)

    label_font = _get_font(_LABEL_FONT_SIZE)

    # Icon area height (above label)
    _label_h = _LABEL_FONT_SIZE + 6  # label text + bottom padding
    _icon_area = _CELL_H - _label_h  # pixels available for the icon

    for i, hk in enumerate(hotkeys[:12]):
        label = str(hk.get("label", "")).upper()
        col = i % _COLS
        row = i // _COLS
        cell_x = col * _CELL_W
        cell_y = row * _CELL_H
        cx = cell_x + _CELL_W // 2

        svg_data = hk.get("svg")
        if svg_data:
            icon = _svg_to_pil(svg_data, _ICON_SIZE)
            if icon:
                if not hk.get("no_adapt"):
                    # Onshape icons are designed for white backgrounds; adapt them
                    # for the dark LCD by boosting dark outlines toward white.
                    icon = _adapt_icon(icon)
                # Centre icon horizontally; vertically within icon area
                ix = cx - _ICON_SIZE // 2
                iy = cell_y + (_icon_area - _ICON_SIZE) // 2 + 3
                img.paste(icon.convert("RGB"), (ix, iy), mask=icon.split()[3])

        if label:
            bbox = label_font.getbbox(label)
            tw = bbox[2] - bbox[0]
            lx = cx - tw // 2 - bbox[0]
            ly = cell_y + _CELL_H - _label_h + 1
            draw.text((lx, ly), label, fill=_C_LABEL, font=label_font)

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
# Rotation-lock LED control
# ---------------------------------------------------------------------------


def _find_enterprise_event_path() -> str | None:
    """Return the /dev/input/eventN path for the SpaceMouse Enterprise.

    Searches /sys/class/input for the device by name, avoiding any dependency
    on the node number (which changes across reboots / re-plugs).
    """
    import glob as _glob

    for uevent in _glob.glob("/sys/class/input/event*/device/uevent"):
        try:
            text = Path(uevent).read_text()
            if "SpaceMouse Enterprise" in text:
                node = uevent.split("/")[4]  # "eventN"
                return f"/dev/input/{node}"
        except Exception:
            pass
    return None


def set_lock_led(on: bool) -> None:
    """Light or extinguish the rotation-lock LED on the SpaceMouse Enterprise.

    Sends EV_LED / LED_MISC via the evdev event node — the same mechanism
    used by spacenavd internally.  Writing LED events to the event node is
    permitted for members of the 'input' group without conflicting with
    spacenavd's exclusive read-grab.

    No-op when the device is absent or the caller lacks read-write permission
    on the event node (requires 'input' group membership).
    """
    import struct
    import time as _time

    EV_LED = 0x11
    LED_SUSPEND = 0x06  # SpaceMouse Enterprise rotation-lock LED

    path = _find_enterprise_event_path()
    if path is None:
        logging.debug("set_lock_led: SpaceMouse Enterprise event node not found")
        return
    try:
        # input_event: { struct timeval tv; __u16 type; __u16 code; __s32 value; }
        # On 64-bit Linux timeval = two int64 fields (sec, usec).
        import os as _os
        t = int(_time.time())
        ev = struct.pack("qqHHi", t, 0, EV_LED, LED_SUSPEND, 1 if on else 0)
        fd = _os.open(path, _os.O_WRONLY | _os.O_NONBLOCK)
        try:
            _os.write(fd, ev)
        finally:
            _os.close(fd)
        logging.debug("set_lock_led(%s) via %s", on, path)
    except PermissionError:
        logging.warning(
            "set_lock_led: no write permission for %s — ensure the user is in the 'input' group and has re-logged in",
            path,
        )
    except Exception as exc:
        logging.debug("set_lock_led(%s) failed — %s", on, exc)


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
            logging.info("Display: 'pyusb' not installed — display disabled.\n  Install with: uv add pyusb")
            return

        try:
            from PIL import Image  # noqa: F401
        except ImportError:
            logging.info("Display: 'pillow' not installed — display disabled.\n  Install with: uv add pillow")
            return

        dev = usb.core.find(idVendor=VENDOR_ID, idProduct=PRODUCT_ID)
        if dev is None:
            logging.warning(
                "Display: SpaceMouse Enterprise not found (VID=0x%04X PID=0x%04X).\n  Check USB connection and udev rule.",
                VENDOR_ID,
                PRODUCT_ID,
            )
            return

        try:
            # Interface 0 is vendor-specific with no kernel driver — claim it.
            # We deliberately do NOT touch interface 1 (HID / spacenavd).
            dev.set_configuration()
        except Exception:
            pass  # configuration may already be set

        # Retry a few times: a previous server killed with SIGKILL may leave
        # the interface marked busy until the OS releases the file descriptors.
        import time

        last_exc: Exception | None = None
        for attempt in range(1, 5):
            try:
                if dev.is_kernel_driver_active(_USB_IFACE):
                    dev.detach_kernel_driver(_USB_IFACE)
                usb.util.claim_interface(dev, _USB_IFACE)
                self._handle = dev
                logging.info(
                    "Display: opened SpaceMouse Enterprise LCD (interface %d, EP 0x%02X)",
                    _USB_IFACE,
                    _USB_EP,
                )
                self.clear()  # blank any leftover content from previous session
                break
            except Exception as exc:
                last_exc = exc
                if attempt < 4:
                    logging.info("Display: claim attempt %d failed, retrying — %s", attempt, exc)
                    time.sleep(0.4)
        else:
            logging.warning("Display: could not claim USB interface — %s", last_exc)

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
        """Send a pre-built display packet via USB bulk transfer.

        The entire packet is sent in a single write() call.  pyusb/libusb
        splits it into 64-byte USB packets internally — the device sees one
        complete transfer and processes its header + compressed bitmap together.
        """
        if self._handle is None:
            return False
        try:
            self._handle.write(_USB_EP, packet, _USB_TIMEOUT)
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
