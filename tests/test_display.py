"""Smoke tests for display rendering functions (display.py).

These tests verify that every rendering path produces a valid packet without
crashing.  They do not open any USB devices — all functions under test are
pure image-processing pipelines that return bytes.
"""

import pytest

from spacenav_ws.buttons import DEFAULT_HOTKEYS
from spacenav_ws.display import (
    DISPLAY_H,
    DISPLAY_W,
    _GRID_H,
    _CELL_H,
    _CELL_W,
    _COLS,
    _ROWS,
    _SENS_BAR_H,
    render_hotkey_grid,
    render_sensitivity_screen,
)

_HEADER_SIZE = 512  # bytes — fixed header prepended to every packet


# ===========================================================================
# Layout constants
# ===========================================================================


class TestLayoutConstants:
    def test_cells_fit_within_display_width(self):
        """Cells may not fill the full width (integer division), but must not exceed it."""
        assert _COLS * _CELL_W <= DISPLAY_W

    def test_cells_fill_grid_height(self):
        assert _ROWS * _CELL_H == _GRID_H

    def test_grid_plus_bar_fits_display(self):
        """Grid + gap + sensitivity bar must not exceed display height."""
        assert _GRID_H + 1 + _SENS_BAR_H <= DISPLAY_H

    def test_sensitivity_bar_does_not_overlap_grid(self):
        """Bottom of the grid cells must be strictly above the bar."""
        grid_bottom = _GRID_H - 1          # last pixel row used by cells
        bar_top = DISPLAY_H - _SENS_BAR_H  # first pixel row of bar
        assert grid_bottom < bar_top


# ===========================================================================
# render_hotkey_grid
# ===========================================================================


class TestRenderHotkeyGrid:
    def test_returns_bytes(self):
        assert isinstance(render_hotkey_grid(DEFAULT_HOTKEYS), bytes)

    def test_packet_larger_than_header(self):
        assert len(render_hotkey_grid(DEFAULT_HOTKEYS)) > _HEADER_SIZE

    def test_empty_hotkey_list(self):
        """Grid with no hotkeys should still render without error."""
        pkt = render_hotkey_grid([])
        assert isinstance(pkt, bytes)
        assert len(pkt) > _HEADER_SIZE

    def test_sensitivity_level_zero(self):
        pkt = render_hotkey_grid(DEFAULT_HOTKEYS, sensitivity_level=0)
        assert isinstance(pkt, bytes)

    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
    def test_sensitivity_levels(self, level):
        pkt = render_hotkey_grid(DEFAULT_HOTKEYS, sensitivity_level=level)
        assert isinstance(pkt, bytes)
        assert len(pkt) > _HEADER_SIZE

    def test_camera_mode_badge(self):
        pkt = render_hotkey_grid(DEFAULT_HOTKEYS, camera_mode=True)
        assert isinstance(pkt, bytes)

    def test_cpv_badge_when_disabled(self):
        pkt = render_hotkey_grid(DEFAULT_HOTKEYS, cursor_pivot_enabled=False)
        assert isinstance(pkt, bytes)

    def test_all_badges_and_bar_together(self):
        pkt = render_hotkey_grid(
            DEFAULT_HOTKEYS,
            sensitivity_level=3,
            camera_mode=True,
            cursor_pivot_enabled=False,
        )
        assert isinstance(pkt, bytes)
        assert len(pkt) > _HEADER_SIZE

    def test_hotkeys_with_svg_icons(self):
        """DEFAULT_HOTKEYS includes SVG icons; rendering must not crash."""
        icon_hotkeys = [h for h in DEFAULT_HOTKEYS if h.get("svg")]
        if icon_hotkeys:
            pkt = render_hotkey_grid(icon_hotkeys)
            assert isinstance(pkt, bytes)

    def test_hotkeys_without_labels(self):
        hotkeys = [{"label": "", "action": "noop"}] * 12
        pkt = render_hotkey_grid(hotkeys)
        assert isinstance(pkt, bytes)

    def test_truncates_to_12_hotkeys(self):
        """Passing more than 12 hotkeys must not crash."""
        many = [{"label": f"K{i}", "action": "noop"} for i in range(20)]
        pkt = render_hotkey_grid(many)
        assert isinstance(pkt, bytes)


# ===========================================================================
# render_sensitivity_screen
# ===========================================================================


class TestRenderSensitivityScreen:
    @pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
    def test_all_levels_return_bytes(self, level):
        pkt = render_sensitivity_screen(level)
        assert isinstance(pkt, bytes)
        assert len(pkt) > _HEADER_SIZE
