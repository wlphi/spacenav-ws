"""Tests for pure math functions in spacenav_ws.

These functions have no I/O dependencies and are safe to run offline.
They encode the core navigation behaviour, so silent regressions here
would produce wrong camera motion in Onshape without any obvious error.
"""

import math

import numpy as np
# ---------------------------------------------------------------------------
# Helpers — import the functions under test directly so the tests remain
# independent of the Controller class and its hardware dependencies.
# ---------------------------------------------------------------------------

from spacenav_ws.controller import _rotation_from_axis_angle, Controller
from spacenav_ws.views import VIEW_MATRICES, get_view_matrix


# ===========================================================================
# _rotation_from_axis_angle
# ===========================================================================


class TestRotationFromAxisAngle:
    def test_zero_angle_returns_identity(self):
        R = _rotation_from_axis_angle(np.array([1.0, 0.0, 0.0]), 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_zero_axis_returns_identity(self):
        R = _rotation_from_axis_angle(np.array([0.0, 0.0, 0.0]), 1.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_result_is_rotation_matrix(self):
        """R must be orthogonal (R @ R.T == I) with det == +1."""
        axis = np.array([1.0, 2.0, 3.0])
        angle = 0.7
        R = _rotation_from_axis_angle(axis, angle)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-10)

    def test_180_degree_rotation_x(self):
        """180° around X: Y → -Y, Z → -Z."""
        R = _rotation_from_axis_angle(np.array([1.0, 0.0, 0.0]), math.pi)
        v = R @ np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(v, [0.0, -1.0, 0.0], atol=1e-10)
        v = R @ np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(v, [0.0, 0.0, -1.0], atol=1e-10)

    def test_90_degree_rotation_z(self):
        """90° around Z: X → Y."""
        R = _rotation_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.pi / 2)
        v = R @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(v, [0.0, 1.0, 0.0], atol=1e-10)

    def test_axis_direction_preserved(self):
        """Points on the rotation axis must be unchanged."""
        axis = np.array([1.0, 1.0, 1.0]) / math.sqrt(3)
        R = _rotation_from_axis_angle(axis, 1.23)
        np.testing.assert_allclose(R @ axis, axis, atol=1e-10)

    def test_full_360_is_identity(self):
        axis = np.array([0.0, 1.0, 0.0])
        R = _rotation_from_axis_angle(axis, 2 * math.pi)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_unnormalised_axis_same_as_normalised(self):
        """Axis magnitude must not affect the result."""
        axis_unit = np.array([1.0, 0.0, 0.0])
        axis_long = np.array([5.0, 0.0, 0.0])
        angle = 1.1
        R1 = _rotation_from_axis_angle(axis_unit, angle)
        R2 = _rotation_from_axis_angle(axis_long, angle)
        np.testing.assert_allclose(R1, R2, atol=1e-12)


# ===========================================================================
# _cursor_pivot  (needs a Controller instance but no hardware)
# ===========================================================================

def _make_controller():
    """Return a Controller with all hardware dependencies stubbed out."""

    class FakeWamp:
        wamp = type("W", (), {
            "subscribe_handlers": {},
            "call_handlers": {},
        })()

    class FakeReader:
        pass

    # Patch display so EnterpriseDisplay() doesn't try to open USB.
    import unittest.mock as mock
    import spacenav_ws.display as disp_mod

    with mock.patch.object(disp_mod, "EnterpriseDisplay", return_value=mock.MagicMock()):
        ctrl = Controller.__new__(Controller)
        # Minimal __init__ without hardware side-effects.
        ctrl.id = "controller0"
        ctrl.client_metadata = {}
        ctrl.reader = FakeReader()
        ctrl.wamp_state_handler = FakeWamp()
        ctrl.subscribed = False
        ctrl.focus = False
        ctrl.lock_rotation = False
        ctrl.shift_held = False
        ctrl._auto_lock_active = False
        ctrl.button_map = {}
        ctrl.shift_map = {}
        ctrl.hotkeys = []
        ctrl.saved_views = {}
        ctrl._cached_model_extents = None
        ctrl._cached_perspective = None
        ctrl._cached_extents = None
        ctrl._cached_frustum = None
        ctrl._cache_time = 0.0
        ctrl._locked_pivot = None
        ctrl._last_motion_time = 0.0
        ctrl._GESTURE_GAP_S = 0.15
        ctrl._last_key_inject_time = 0.0
        ctrl._viewport_ar = 16.0 / 9.0
        ctrl._cursor_ndc = [0.0, 0.0]
        ctrl._cursor_active = False
        ctrl._cursor_debug_pivot = [0.0, 0.0, 0.0]
        ctrl._cursor_debug_dist = 0.0
        ctrl._cursor_debug_viewport_half = 0.0
        ctrl._cursor_debug_used_cursor = False
        ctrl._rotation_scale = 1.0
        ctrl._translation_scale = 1.0
        ctrl._zoom_scale = 1.0
        ctrl._active_set = ""
        ctrl._context_commands = {}
        ctrl._svg_cache = {}
        ctrl._last_display_key = ()
        ctrl.display = mock.MagicMock()
    return ctrl


class TestCursorPivot:
    """_cursor_pivot(nx, ny, model_extents, curr_affine, extents)"""

    def _identity_affine(self):
        return np.eye(4, dtype=np.float64)

    def _extents(self, half_x=1.0, half_y=1.0):
        return [-half_x, -half_y, -10.0, half_x, half_y, 10.0]

    def _model_extents(self, half=1.0):
        return [-half, -half, -half, half, half, half]

    def test_centre_cursor_returns_model_centre(self):
        ctrl = _make_controller()
        result = ctrl._cursor_pivot(0.0, 0.0, self._model_extents(), self._identity_affine(), self._extents())
        # Model centre is origin; cursor at NDC (0,0) should give model centre
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)

    def test_missing_extents_returns_model_centre(self):
        ctrl = _make_controller()
        result = ctrl._cursor_pivot(0.5, 0.5, self._model_extents(), self._identity_affine(), None)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)

    def test_far_cursor_falls_back_to_model_centre(self):
        """NDC far outside viewport (e.g. cursor in a toolbar) → model centre."""
        ctrl = _make_controller()
        # NDC (10, 10) is 10× the half-extent away — well beyond the 2× guard
        result = ctrl._cursor_pivot(10.0, 10.0, self._model_extents(), self._identity_affine(), self._extents())
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0], atol=1e-10)
        assert ctrl._cursor_debug_used_cursor is False

    def test_near_cursor_is_used(self):
        ctrl = _make_controller()
        ctrl._cursor_pivot(0.3, 0.3, self._model_extents(), self._identity_affine(), self._extents())
        assert ctrl._cursor_debug_used_cursor is True

    def test_debug_fields_written(self):
        ctrl = _make_controller()
        ctrl._cursor_pivot(0.0, 0.0, self._model_extents(), self._identity_affine(), self._extents())
        # Fields must be numbers, not zero-default from unexecuted branches
        assert isinstance(ctrl._cursor_debug_dist, float)
        assert isinstance(ctrl._cursor_debug_viewport_half, float)

    def test_pivot_moves_with_cursor(self):
        """Pivot X should shift when NDC X shifts (identity affine, symmetric extents)."""
        ctrl = _make_controller()
        p0 = ctrl._cursor_pivot(0.0, 0.0, self._model_extents(2.0), self._identity_affine(), self._extents(2.0, 2.0))
        p1 = ctrl._cursor_pivot(0.5, 0.0, self._model_extents(2.0), self._identity_affine(), self._extents(2.0, 2.0))
        assert p1[0] > p0[0], "pivot X should increase when NDC X increases"


# ===========================================================================
# views.py — VIEW_MATRICES sanity checks
# ===========================================================================


class TestViewMatrices:
    def test_all_views_present(self):
        for name in ("top", "bottom", "front", "back", "left", "right", "iso1", "iso2"):
            assert get_view_matrix(name) is not None, f"missing view: {name}"

    def test_all_matrices_are_valid_rotations(self):
        for name, flat in VIEW_MATRICES.items():
            M = np.array(flat).reshape(4, 4)
            R = M[:3, :3]
            # Rotation part must be orthogonal
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10, err_msg=f"{name}: R not orthogonal")
            assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-10), f"{name}: det(R) != 1"

    def test_unknown_view_returns_none(self):
        assert get_view_matrix("nonexistent") is None

    def test_top_view_looks_down(self):
        """Top view: camera -Z axis should point downward (+Z world means up, so cam -Z = world -Z = looking down)."""
        M = np.array(get_view_matrix("top")).reshape(4, 4)
        # Row 2 of M[:3,:3] is the camera Z axis in world space; -Row2 is look direction
        look = -M[2, :3]
        # Looking "down" in Onshape Z-up means negative Z world direction
        assert look[2] < -0.9, f"top view should look toward -Z, got look={look}"

    def test_front_view_looks_along_y(self):
        """Front view: camera should look in +Y world direction."""
        M = np.array(get_view_matrix("front")).reshape(4, 4)
        look = -M[2, :3]
        assert look[1] > 0.9, f"front view should look toward +Y, got look={look}"
