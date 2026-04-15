"""Tests for button configuration loading and override logic (buttons.py).

No hardware is required — all functions under test are pure config-file logic.
"""

import json

import pytest

import spacenav_ws.buttons as btn_mod
from spacenav_ws.buttons import (
    CTRL_BUTTON_ID,
    DEFAULT_HOTKEYS,
    ENTERPRISE_DEFAULT_BUTTON_MAP,
    ENTERPRISE_DEFAULT_CTRL_MAP,
    ENTERPRISE_DEFAULT_SHIFT_MAP,
    SHIFT_BUTTON_ID,
    get_button_map,
    get_context_hotkey_map,
    get_ctrl_map,
    get_hotkeys,
    get_shift_map,
)


# ===========================================================================
# Default map invariants
# ===========================================================================


class TestDefaultMaps:
    def test_shift_button_not_in_button_map(self):
        """Shift (button 20) is a modifier and must not appear as a dispatchable action."""
        assert SHIFT_BUTTON_ID not in ENTERPRISE_DEFAULT_BUTTON_MAP

    def test_ctrl_button_not_in_button_map(self):
        """Ctrl (button 21) is a modifier and must not appear as a dispatchable action."""
        assert CTRL_BUTTON_ID not in ENTERPRISE_DEFAULT_BUTTON_MAP

    def test_hotkey_buttons_0_to_11_mapped(self):
        """Buttons 0-11 must map to hotkey_1 … hotkey_12."""
        for i in range(12):
            assert ENTERPRISE_DEFAULT_BUTTON_MAP[i] == f"hotkey_{i + 1}"

    def test_shift_menu_toggles_camera_mode(self):
        assert ENTERPRISE_DEFAULT_SHIFT_MAP[12] == "toggle_camera_mode"

    def test_shift_lock_toggles_horizon_lock(self):
        """Shift + Lock (button 22) must trigger horizon-lock, not rotation-lock."""
        assert ENTERPRISE_DEFAULT_SHIFT_MAP[22] == "toggle_horizon_lock"

    def test_ctrl_menu_toggles_cursor_pivot(self):
        assert ENTERPRISE_DEFAULT_CTRL_MAP[12] == "toggle_cursor_pivot"

    def test_default_hotkeys_length(self):
        assert len(DEFAULT_HOTKEYS) == 12

    def test_default_hotkey_labels_are_strings(self):
        for hk in DEFAULT_HOTKEYS:
            assert isinstance(hk.get("label", ""), str)

    def test_default_hotkey_actions_are_strings(self):
        for hk in DEFAULT_HOTKEYS:
            assert isinstance(hk.get("action", ""), str)


# ===========================================================================
# Config file overrides
# ===========================================================================


@pytest.fixture(autouse=True)
def clear_config_cache(monkeypatch):
    """Ensure each test starts with a clean config cache."""
    monkeypatch.setattr(btn_mod, "_config_cache", None)


class TestButtonMapOverride:
    def test_override_replaces_single_button(self, tmp_path, monkeypatch):
        config = {"button_map": {"13": "zoom_in"}}
        _write_config(tmp_path, config, monkeypatch)
        assert get_button_map()[13] == "zoom_in"

    def test_override_preserves_unmentioned_buttons(self, tmp_path, monkeypatch):
        config = {"button_map": {"13": "zoom_in"}}
        _write_config(tmp_path, config, monkeypatch)
        m = get_button_map()
        # Button 14 (view_top) should be unchanged
        assert m[14] == "view_top"

    def test_invalid_key_is_silently_ignored(self, tmp_path, monkeypatch):
        config = {"button_map": {"not_a_number": "fit"}}
        _write_config(tmp_path, config, monkeypatch)
        m = get_button_map()
        assert isinstance(m, dict)

    def test_missing_config_file_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setattr(btn_mod, "CONFIG_PATH", tmp_path / "nonexistent.json")
        assert get_button_map() == ENTERPRISE_DEFAULT_BUTTON_MAP


class TestShiftMapOverride:
    def test_override_adds_new_entry(self, tmp_path, monkeypatch):
        config = {"shift_map": {"13": "view_iso2"}}
        _write_config(tmp_path, config, monkeypatch)
        assert get_shift_map()[13] == "view_iso2"

    def test_missing_config_file_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setattr(btn_mod, "CONFIG_PATH", tmp_path / "nonexistent.json")
        assert get_shift_map() == ENTERPRISE_DEFAULT_SHIFT_MAP


class TestCtrlMapOverride:
    def test_override_adds_new_entry(self, tmp_path, monkeypatch):
        config = {"ctrl_map": {"13": "fit"}}
        _write_config(tmp_path, config, monkeypatch)
        assert get_ctrl_map()[13] == "fit"

    def test_missing_config_file_returns_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setattr(btn_mod, "CONFIG_PATH", tmp_path / "nonexistent.json")
        assert get_ctrl_map() == ENTERPRISE_DEFAULT_CTRL_MAP


class TestHotkeyOverride:
    def test_override_first_hotkey(self, tmp_path, monkeypatch):
        config = {"hotkeys": [{"label": "MYKEY", "action": "fit"}]}
        _write_config(tmp_path, config, monkeypatch)
        hotkeys = get_hotkeys()
        assert hotkeys[0]["label"] == "MYKE"  # truncated to 4 chars and uppercased
        assert hotkeys[0]["action"] == "fit"

    def test_label_truncated_to_4_chars(self, tmp_path, monkeypatch):
        config = {"hotkeys": [{"label": "TOOLONG", "action": "noop"}]}
        _write_config(tmp_path, config, monkeypatch)
        assert len(get_hotkeys()[0]["label"]) <= 4

    def test_label_is_uppercased(self, tmp_path, monkeypatch):
        config = {"hotkeys": [{"label": "fit", "action": "fit"}]}
        _write_config(tmp_path, config, monkeypatch)
        assert get_hotkeys()[0]["label"] == "FIT"

    def test_override_only_affects_specified_indices(self, tmp_path, monkeypatch):
        config = {"hotkeys": [{"label": "NEW", "action": "fit"}]}
        _write_config(tmp_path, config, monkeypatch)
        hotkeys = get_hotkeys()
        # Slot 0 overridden, slots 1-11 should keep defaults
        assert hotkeys[0]["label"] == "NEW"
        assert hotkeys[1] == DEFAULT_HOTKEYS[1]

    def test_override_limited_to_12(self, tmp_path, monkeypatch):
        """More than 12 hotkey entries in config must be silently capped."""
        config = {"hotkeys": [{"label": f"K{i}", "action": "noop"} for i in range(20)]}
        _write_config(tmp_path, config, monkeypatch)
        assert len(get_hotkeys()) == 12

    def test_invalid_json_falls_back_to_defaults(self, tmp_path, monkeypatch):
        bad_file = tmp_path / "config.json"
        bad_file.write_text("{not valid json")
        monkeypatch.setattr(btn_mod, "CONFIG_PATH", bad_file)
        assert get_button_map() == ENTERPRISE_DEFAULT_BUTTON_MAP


class TestContextHotkeyMap:
    def test_returns_empty_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr(btn_mod, "CONFIG_PATH", tmp_path / "nonexistent.json")
        assert get_context_hotkey_map() == {}

    def test_returns_empty_when_key_absent(self, tmp_path, monkeypatch):
        config = {"button_map": {"13": "fit"}}
        _write_config(tmp_path, config, monkeypatch)
        assert get_context_hotkey_map() == {}

    def test_parses_single_context(self, tmp_path, monkeypatch):
        config = {
            "context_hotkeys": {
                "Assembly": [
                    {"label": "INS", "action": "onshape_Assembly-insertPartOrAssembly"},
                    {"label": "FAST", "action": "onshape_mate_FASTENED"},
                ]
            }
        }
        _write_config(tmp_path, config, monkeypatch)
        result = get_context_hotkey_map()
        assert "Assembly" in result
        hotkeys = result["Assembly"]
        assert hotkeys[0]["label"] == "INS"
        assert hotkeys[0]["action"] == "onshape_Assembly-insertPartOrAssembly"
        assert hotkeys[1]["label"] == "FAST"

    def test_label_truncated_and_uppercased(self, tmp_path, monkeypatch):
        config = {
            "context_hotkeys": {
                "Part Studio": [{"label": "toolong", "action": "noop"}]
            }
        }
        _write_config(tmp_path, config, monkeypatch)
        hotkeys = get_context_hotkey_map()["Part Studio"]
        assert hotkeys[0]["label"] == "TOOL"

    def test_capped_at_12_entries(self, tmp_path, monkeypatch):
        config = {
            "context_hotkeys": {
                "Assembly": [{"label": f"H{i}", "action": "noop"} for i in range(20)]
            }
        }
        _write_config(tmp_path, config, monkeypatch)
        assert len(get_context_hotkey_map()["Assembly"]) == 12

    def test_multiple_contexts_independent(self, tmp_path, monkeypatch):
        config = {
            "context_hotkeys": {
                "Assembly": [{"label": "A1", "action": "onshape_mate_FASTENED"}],
                "Part Studio": [{"label": "P1", "action": "onshape_extrude"}],
            }
        }
        _write_config(tmp_path, config, monkeypatch)
        result = get_context_hotkey_map()
        assert result["Assembly"][0]["action"] == "onshape_mate_FASTENED"
        assert result["Part Studio"][0]["action"] == "onshape_extrude"

    def test_noop_entries_preserved(self, tmp_path, monkeypatch):
        config = {
            "context_hotkeys": {
                "Assembly": [{"label": "", "action": "noop"}]
            }
        }
        _write_config(tmp_path, config, monkeypatch)
        hotkeys = get_context_hotkey_map()["Assembly"]
        assert hotkeys[0]["label"] == ""
        assert hotkeys[0]["action"] == "noop"

    def test_missing_label_defaults_to_empty_string(self, tmp_path, monkeypatch):
        config = {
            "context_hotkeys": {
                "Assembly": [{"action": "fit"}]
            }
        }
        _write_config(tmp_path, config, monkeypatch)
        hotkeys = get_context_hotkey_map()["Assembly"]
        assert hotkeys[0]["label"] == ""

    def test_missing_action_defaults_to_noop(self, tmp_path, monkeypatch):
        config = {
            "context_hotkeys": {
                "Assembly": [{"label": "X"}]
            }
        }
        _write_config(tmp_path, config, monkeypatch)
        hotkeys = get_context_hotkey_map()["Assembly"]
        assert hotkeys[0]["action"] == "noop"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(tmp_path, config: dict, monkeypatch) -> None:
    """Write config to a temp file and point CONFIG_PATH at it."""
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps(config))
    monkeypatch.setattr(btn_mod, "CONFIG_PATH", cfg_file)
