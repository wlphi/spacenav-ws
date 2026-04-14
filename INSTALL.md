# Installation (Linux / local development)

This guide covers installing **spacenav-ws** from source on Linux, including all system-level prerequisites.

## 1. Install spacenavd

Install and enable the open-source SpaceMouse driver:

```bash
# Arch / Manjaro
sudo pacman -S spacenavd
sudo systemctl enable --now spacenavd
```

Verify it is running and the socket is present:

```bash
systemctl is-active spacenavd        # → active
ls /var/run/spnav.sock               # → should exist
```

## 2. Install system dependencies

```bash
# SVG renderer for the SpaceMouse Enterprise LCD display
sudo pacman -S resvg
```

## 3. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 4. Clone the repo

```bash
git clone https://github.com/rmstorm/spacenav-ws.git
cd spacenav-ws
```

## 5. Pin Python and install dependencies

The project requires Python ≤ 3.13 (a transitive dependency, `watchfiles`, does not yet support 3.14+).

```bash
uv python pin 3.13
uv sync
```

## 6. Udev rules

Two udev rules are needed — one for USB access to the SpaceMouse (required for the LCD display on SpaceMouse Enterprise), and one for creating virtual input devices via uinput.

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", MODE="0666"' \
    | sudo tee /etc/udev/rules.d/99-spacemouse.rules

echo 'KERNEL=="uinput", MODE="0660", GROUP="input"' \
    | sudo tee /etc/udev/rules.d/99-uinput.rules

sudo udevadm control --reload-rules && sudo udevadm trigger
```

Add your user to the `input` group (required for uinput / virtual keyboard injection):

```bash
sudo usermod -aG input $USER
```

Log out and back in for the group change to take effect, then replug the SpaceMouse.

## 7. Verify

```bash
uv run spacenav-ws read-mouse
# → no errors; wiggle the mouse to see motion events printed
```

## 8. Run the server

```bash
uv run spacenav-ws serve
```

Open [https://127.51.68.120:8181](https://127.51.68.120:8181) in your browser and add a certificate exception for the self-signed cert.

## 9. Autostart with systemd (optional)

To run spacenav-ws automatically at login:

```bash
cp additional/spacenav-ws.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now spacenav-ws.service
```

The service file defaults to `uvx spacenav-ws serve` (PyPI install). If you are running from a local clone, edit the service file and swap to the commented-out `uv run` line, adjusting the project path.

Check status:

```bash
systemctl --user status spacenav-ws.service
journalctl --user -u spacenav-ws.service -f
```

## 10. Install the userscript

Install [Tampermonkey](https://addons.mozilla.org/en-US/firefox/addon/tampermonkey/) (Firefox) or [Violentmonkey](https://violentmonkey.github.io/), then install the script via [Greasy Fork](https://greasyfork.org/en/scripts/533516-onshape-3d-mouse-on-linux-in-page-patch).

Open an Onshape document — the SpaceMouse should now work.

## 11. Tuning sensitivity

Sensitivity is controlled at two independent layers:

### Device level — `/etc/spnavrc`

spacenavd applies these settings globally, before any application sees the motion data. Edit the file as root, then reload without restarting: `sudo kill -HUP $(pidof spacenavd)`.

```ini
# /etc/spnavrc — relevant sensitivity settings

# Master multiplier for all 6 axes (default: 1.0)
sensitivity = 1.0

# Independent multipliers for translation and rotation
sensitivity-translation = 1.0
sensitivity-rotation     = 0.8

# Per-axis fine-tuning (cumulative with the above)
# sensitivity-translation-x = 1.0
# sensitivity-translation-y = 1.0
# sensitivity-translation-z = 1.0
# sensitivity-rotation-x    = 1.0
# sensitivity-rotation-y    = 1.0
# sensitivity-rotation-z    = 1.0

# Noise floor — raw units below this threshold are zeroed (suppresses jitter)
dead-zone = 5

# Axis inversion — list axes to invert, or "none"
# invert-trans = none
# invert-rot   = y z
```

See `/usr/share/doc/spacenavd/example-spnavrc` (or the upstream repo) for the full reference.

### Onshape level — `~/.config/spacenav-ws/config.json`

These multipliers scale the Onshape-specific motion constants in spacenav-ws (rotation speed in radians, pan in view-spans, zoom rate). They only affect the SpaceMouse-in-Onshape experience; spacenavd settings are device-wide and affect all applications.

```json
{
  "motion": {
    "rotation_scale":    1.0,
    "translation_scale": 1.0,
    "zoom_scale":        1.0
  }
}
```

`1.0` is the default. Set `rotation_scale` to `0.5` to halve rotation speed, `2.0` to double it, etc. Restart `spacenav-ws` after editing.
