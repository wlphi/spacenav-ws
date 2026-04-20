"""Tests for pure math functions in spacenav_ws.

These functions have no I/O dependencies and are safe to run offline.
They encode the core navigation behaviour, so silent regressions here
would produce wrong camera motion in Onshape without any obvious error.
"""

import math

import numpy as np
import pytest

from spacenav_ws.controller import (
    CursorPivotResult,
    _rotation_from_axis_angle,
    compute_cursor_pivot,
    Controller,
)
from spacenav_ws.spacenav import ButtonEvent, MotionEvent, from_message
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
# compute_cursor_pivot  — pure function, no hardware stub needed
# ===========================================================================


def _identity_affine():
    return np.eye(4, dtype=np.float64)


def _extents(half_x=1.0, half_y=1.0):
    return [-half_x, -half_y, -10.0, half_x, half_y, 10.0]


def _model_extents(half=1.0):
    return [-half, -half, -half, half, half, half]


class TestComputeCursorPivot:
    """compute_cursor_pivot is a pure function; no Controller stub required."""

    def test_returns_cursor_pivot_result(self):
        r = compute_cursor_pivot(0.0, 0.0, _model_extents(), _identity_affine(), _extents())
        assert isinstance(r, CursorPivotResult)

    def test_centre_cursor_returns_model_centre(self):
        r = compute_cursor_pivot(0.0, 0.0, _model_extents(), _identity_affine(), _extents())
        np.testing.assert_allclose(r.pivot, [0.0, 0.0, 0.0], atol=1e-10)

    def test_missing_extents_uses_bbox_estimate(self):
        """When view.extents is None the bbox-estimate path is taken.

        With identity affine and a unit cube model the bbox-projected half-extents
        are 1.0 * 1.5 = 1.5 in both X and Y.  Cursor (0.5, 0.5) → camera-space
        offset (0.75, 0.75) → result moves away from model centre.
        """
        r = compute_cursor_pivot(0.5, 0.5, _model_extents(), _identity_affine(), None)
        assert r.used_cursor is True
        assert r.source == "bbox_estimate"
        np.testing.assert_allclose(r.pivot, [0.75, 0.75, 0.0], atol=1e-6)

    def test_far_cursor_falls_back_to_model_centre(self):
        """NDC far outside viewport (cursor in toolbar) → model centre."""
        r = compute_cursor_pivot(10.0, 10.0, _model_extents(), _identity_affine(), _extents())
        np.testing.assert_allclose(r.pivot, [0.0, 0.0, 0.0], atol=1e-10)
        assert r.used_cursor is False
        assert r.source == "model_center_oob"

    def test_near_cursor_is_used(self):
        r = compute_cursor_pivot(0.3, 0.3, _model_extents(), _identity_affine(), _extents())
        assert r.used_cursor is True

    def test_debug_fields_are_floats(self):
        r = compute_cursor_pivot(0.0, 0.0, _model_extents(), _identity_affine(), _extents())
        assert isinstance(r.dist, float)
        assert isinstance(r.viewport_half, float)

    def test_pivot_moves_with_cursor(self):
        """Pivot X should shift when NDC X shifts (identity affine, symmetric extents)."""
        r0 = compute_cursor_pivot(0.0, 0.0, _model_extents(2.0), _identity_affine(), _extents(2.0, 2.0))
        r1 = compute_cursor_pivot(0.5, 0.0, _model_extents(2.0), _identity_affine(), _extents(2.0, 2.0))
        assert r1.pivot[0] > r0.pivot[0], "pivot X should increase when NDC X increases"

    def test_source_is_cursor_with_extents(self):
        r = compute_cursor_pivot(0.3, 0.0, _model_extents(), _identity_affine(), _extents())
        assert r.source == "cursor"

    def test_viewport_half_equals_max_half_extent(self):
        """viewport_half must be max(cx_half, cy_half) from extents."""
        r = compute_cursor_pivot(0.0, 0.0, _model_extents(), _identity_affine(), _extents(half_x=2.0, half_y=1.0))
        assert math.isclose(r.viewport_half, 2.0, rel_tol=1e-6)


# ===========================================================================
# Controller._get_affine_pivot_matrices  (static method, no stub needed)
# ===========================================================================


class TestGetAffinePivotMatrices:
    def test_pivot_at_origin_returns_identity_pair(self):
        pivot_pos, pivot_neg = Controller._get_affine_pivot_matrices(np.zeros(3))
        np.testing.assert_allclose(pivot_pos, np.eye(4), atol=1e-10)
        np.testing.assert_allclose(pivot_neg, np.eye(4), atol=1e-10)

    def test_pivot_pos_neg_are_inverses(self):
        """pivot_pos @ pivot_neg must equal identity."""
        pivot = np.array([1.0, 2.0, 3.0])
        pivot_pos, pivot_neg = Controller._get_affine_pivot_matrices(pivot)
        np.testing.assert_allclose(pivot_pos @ pivot_neg, np.eye(4), atol=1e-6)

    def test_pivot_translation_stored_in_row3(self):
        pivot = np.array([1.0, 2.0, 3.0])
        pivot_pos, pivot_neg = Controller._get_affine_pivot_matrices(pivot)
        np.testing.assert_allclose(pivot_pos[3, :3], pivot, atol=1e-10)
        np.testing.assert_allclose(pivot_neg[3, :3], -pivot, atol=1e-10)


# ===========================================================================
# spacenav.py — from_message event parsing
# ===========================================================================


class TestFromMessage:
    def test_motion_event_type_zero(self):
        msg = [0, 10, -5, 3, 1, 2, -3, 16]
        event = from_message(msg)
        assert isinstance(event, MotionEvent)
        assert event.type == "mtn"

    def test_motion_event_axes_mapped_correctly(self):
        """from_message maps (type, x, z, y, pitch, yaw, roll, period)."""
        msg = [0, 1, 2, 3, 4, 5, 6, 16]
        event = from_message(msg)
        assert event.x == 1
        assert event.z == 2   # note: raw index 2 → .z
        assert event.y == 3   # raw index 3 → .y
        assert event.pitch == 4
        assert event.yaw == 5
        assert event.roll == 6
        assert event.period == 16

    def test_button_press_type_one(self):
        msg = [1, 7, 0, 0, 0, 0, 0, 0]
        event = from_message(msg)
        assert isinstance(event, ButtonEvent)
        assert event.button_id == 7
        assert event.pressed is True
        assert event.type == "btn"

    def test_button_release_type_two(self):
        msg = [2, 7, 0, 0, 0, 0, 0, 0]
        event = from_message(msg)
        assert isinstance(event, ButtonEvent)
        assert event.button_id == 7
        assert event.pressed is False

    def test_motion_zero_all_axes(self):
        msg = [0, 0, 0, 0, 0, 0, 0, 0]
        event = from_message(msg)
        assert isinstance(event, MotionEvent)
        assert event.x == 0 and event.y == 0 and event.z == 0

    def test_motion_negative_values(self):
        msg = [0, -100, -200, -300, -400, -500, -600, 32]
        event = from_message(msg)
        assert event.x == -100
        assert event.roll == -600


# ===========================================================================
# display.py — pure image processing
# ===========================================================================


class TestAdaptIcon:
    """_adapt_icon lifts dark pixels toward white and desaturates colour."""

    def _solid_rgba(self, r, g, b, a=255, size=4):
        from PIL import Image
        img = Image.new("RGBA", (size, size), (r, g, b, a))
        return img

    def test_returns_rgba_image(self):
        from spacenav_ws.display import _adapt_icon
        from PIL import Image
        result = _adapt_icon(self._solid_rgba(0, 0, 0))
        assert isinstance(result, Image.Image)
        assert result.mode == "RGBA"

    def test_black_pixels_become_bright(self):
        """Fully black pixels should be boosted to near-white."""
        from spacenav_ws.display import _adapt_icon
        result = _adapt_icon(self._solid_rgba(0, 0, 0))
        arr = np.array(result)
        assert arr[0, 0, 0] > 200, "red channel should be boosted toward 255"
        assert arr[0, 0, 1] > 200, "green channel should be boosted toward 255"

    def test_white_pixels_unchanged(self):
        """Fully white pixels are above the boost threshold and should stay white."""
        from spacenav_ws.display import _adapt_icon
        result = _adapt_icon(self._solid_rgba(255, 255, 255))
        arr = np.array(result)
        assert arr[0, 0, 0] > 250
        assert arr[0, 0, 1] > 250

    def test_alpha_channel_preserved(self):
        """Alpha is not touched by the adapt function."""
        from spacenav_ws.display import _adapt_icon
        result = _adapt_icon(self._solid_rgba(128, 128, 128, 64))
        arr = np.array(result)
        assert arr[0, 0, 3] == 64

    def test_output_same_size_as_input(self):
        from spacenav_ws.display import _adapt_icon
        result = _adapt_icon(self._solid_rgba(100, 50, 200, size=8))
        assert result.size == (8, 8)


class TestImgToBgr565:
    """_img_to_bgr565 converts RGB PIL image to little-endian BGR565 bytes."""

    def test_output_length(self):
        from PIL import Image
        from spacenav_ws.display import _img_to_bgr565, DISPLAY_W, DISPLAY_H
        img = Image.new("RGB", (DISPLAY_W, DISPLAY_H), (0, 0, 0))
        data = _img_to_bgr565(img)
        assert len(data) == DISPLAY_W * DISPLAY_H * 2  # 2 bytes per pixel

    def test_pure_red_encodes_correctly(self):
        """Pure red (255,0,0) in RGB565 = 0b00000_000000_11111 = 0x001F (little-endian)."""
        from PIL import Image
        from spacenav_ws.display import _img_to_bgr565
        img = Image.new("RGB", (1, 1), (255, 0, 0))
        data = _img_to_bgr565(img)
        word = int.from_bytes(data[:2], "little")
        # In BGR565: bits [15:11]=blue(0), [10:5]=green(0), [4:0]=red(31)
        assert word & 0x001F == 31, "red channel should occupy low 5 bits"
        assert (word >> 5) & 0x3F == 0, "green should be 0"
        assert (word >> 11) & 0x1F == 0, "blue should be 0"

    def test_pure_blue_encodes_correctly(self):
        """Pure blue (0,0,255) → high 5 bits set in BGR565."""
        from PIL import Image
        from spacenav_ws.display import _img_to_bgr565
        img = Image.new("RGB", (1, 1), (0, 0, 255))
        data = _img_to_bgr565(img)
        word = int.from_bytes(data[:2], "little")
        assert (word >> 11) & 0x1F == 31, "blue channel should occupy high 5 bits"
        assert word & 0x001F == 0, "red should be 0"


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
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10, err_msg=f"{name}: R not orthogonal")
            assert math.isclose(np.linalg.det(R), 1.0, abs_tol=1e-10), f"{name}: det(R) != 1"

    def test_unknown_view_returns_none(self):
        assert get_view_matrix("nonexistent") is None

    def test_top_view_looks_down(self):
        """Top view: camera -Z axis should point downward (+Z world means up, so cam -Z = world -Z = looking down)."""
        M = np.array(get_view_matrix("top")).reshape(4, 4)
        look = -M[2, :3]
        assert look[2] < -0.9, f"top view should look toward -Z, got look={look}"

    def test_front_view_looks_along_y(self):
        """Front view: camera should look in +Y world direction."""
        M = np.array(get_view_matrix("front")).reshape(4, 4)
        look = -M[2, :3]
        assert look[1] > 0.9, f"front view should look toward +Y, got look={look}"

    def test_all_views_have_zero_translation(self):
        """Base view matrices must have zero translation (centering is applied at runtime)."""
        for name, flat in VIEW_MATRICES.items():
            M = np.array(flat).reshape(4, 4)
            np.testing.assert_allclose(
                M[3, :3], [0.0, 0.0, 0.0], atol=1e-12,
                err_msg=f"{name}: base matrix should have zero translation",
            )

    @pytest.mark.parametrize("name,expected_look", [
        ("top",    ( 0,  0, -1)),
        ("bottom", ( 0,  0,  1)),
        ("front",  ( 0,  1,  0)),
        ("back",   ( 0, -1,  0)),
        ("right",  (-1,  0,  0)),
        ("left",   ( 1,  0,  0)),
    ])
    def test_axis_aligned_look_directions(self, name, expected_look):
        """Each axis-aligned view must look exactly along its expected world axis."""
        M = np.array(get_view_matrix(name)).reshape(4, 4)
        look = -M[2, :3]          # cam_z = M[2,:3], look direction = -cam_z
        np.testing.assert_allclose(look, expected_look, atol=1e-10,
                                   err_msg=f"{name}: unexpected look direction")

    def test_iso_views_look_toward_origin(self):
        """ISO views must have all three look-direction components non-zero."""
        for name in ("iso1", "iso2"):
            M = np.array(get_view_matrix(name)).reshape(4, 4)
            look = -M[2, :3]
            assert all(abs(c) > 0.1 for c in look), \
                f"{name}: expected diagonal look direction, got {look}"

    def test_up_direction_is_z_for_ortho_views(self):
        """For all non-top/bottom views the camera Y axis (up in image) must be +Z world."""
        for name in ("front", "back", "left", "right"):
            M = np.array(get_view_matrix(name)).reshape(4, 4)
            cam_y = M[1, :3]   # camera up direction in world space
            np.testing.assert_allclose(cam_y, [0.0, 0.0, 1.0], atol=1e-10,
                                       err_msg=f"{name}: camera up should be +Z")


# ===========================================================================
# _action_set_view centering math
# ===========================================================================


class TestViewCentering:
    """Verify the centering formula used in _action_set_view.

    Contract: given model extents [mn, mx] and a view rotation R, the
    translation T = -(centre @ R) must map the model centre to the
    camera-space origin.  This is purely algebraic — no I/O required.
    """

    def _centred_affine(self, view_name: str, mn, mx) -> np.ndarray:
        A = np.array(get_view_matrix(view_name), dtype=np.float64).reshape(4, 4)
        R = A[:3, :3]
        centre = (np.asarray(mn) + np.asarray(mx)) / 2.0
        A[3, :3] = -(centre @ R)
        return A

    @pytest.mark.parametrize("view_name", list(VIEW_MATRICES))
    def test_model_centre_maps_to_camera_origin(self, view_name):
        """After centering, model centre must land at (0, 0, 0) in camera space."""
        mn = np.array([1.0, 2.0, 3.0])
        mx = np.array([5.0, 6.0, 7.0])
        A = self._centred_affine(view_name, mn, mx)
        R, T = A[:3, :3], A[3, :3]
        centre = (mn + mx) / 2.0
        centre_cam = centre @ R + T
        np.testing.assert_allclose(centre_cam, [0.0, 0.0, 0.0], atol=1e-12,
                                   err_msg=f"{view_name}: centre not at camera origin")

    @pytest.mark.parametrize("view_name", list(VIEW_MATRICES))
    def test_model_corners_are_symmetric_in_camera_space(self, view_name):
        """After centering, the 8 model corners must be symmetric about the
        camera-space origin (i.e. the bounding box is centred at zero)."""
        mn = np.array([1.0, 2.0, 3.0])
        mx = np.array([5.0, 8.0, 11.0])
        A = self._centred_affine(view_name, mn, mx)
        R, T = A[:3, :3], A[3, :3]
        # All 8 corners of the AABB
        corners = np.array([[x, y, z]
                             for x in (mn[0], mx[0])
                             for y in (mn[1], mx[1])
                             for z in (mn[2], mx[2])], dtype=np.float64)
        cam = corners @ R + T
        cam_centre = cam.mean(axis=0)
        np.testing.assert_allclose(cam_centre, [0.0, 0.0, 0.0], atol=1e-12,
                                   err_msg=f"{view_name}: corners not symmetric")

    def test_centering_does_not_alter_rotation(self):
        """Centering must only modify the translation row, not the rotation."""
        for name in VIEW_MATRICES:
            base = np.array(get_view_matrix(name), dtype=np.float64).reshape(4, 4)
            centred = self._centred_affine(name, [0, 0, 0], [4, 4, 4])
            np.testing.assert_array_equal(base[:3, :3], centred[:3, :3])

    def test_centering_with_origin_centred_model(self):
        """Model centred at world origin → T must be zero for all views."""
        mn = np.array([-2.0, -3.0, -1.0])
        mx = np.array([ 2.0,  3.0,  1.0])
        for name in VIEW_MATRICES:
            A = self._centred_affine(name, mn, mx)
            np.testing.assert_allclose(A[3, :3], [0.0, 0.0, 0.0], atol=1e-12,
                                       err_msg=f"{name}: origin-centred model should give T=0")

    @pytest.mark.parametrize("offset", [
        np.array([  100.0,    0.0,    0.0]),   # far along +X
        np.array([ -500.0, -300.0, -200.0]),   # deep negative quadrant
        np.array([1000.0,  2000.0, 3000.0]),   # very large positive offset
        np.array([  0.001,   0.001,  0.001]),  # tiny near-origin offset
    ])
    def test_centering_works_for_off_origin_model(self, offset):
        """model.extents can describe a bounding box anywhere in world space.
        The centering formula must work correctly regardless of position."""
        half = np.array([1.0, 2.0, 0.5])
        mn = offset - half
        mx = offset + half
        expected_centre = offset          # (mn + mx) / 2

        for name in VIEW_MATRICES:
            A = self._centred_affine(name, mn, mx)
            R, T = A[:3, :3], A[3, :3]
            # Model centre must land at camera-space origin
            centre_cam = expected_centre @ R + T
            np.testing.assert_allclose(
                centre_cam, [0.0, 0.0, 0.0], atol=1e-10,
                err_msg=f"{name}: off-origin model centre not at camera origin"
            )
            # Centroid of all 8 corners must also be at camera-space origin
            corners = np.array([[mn[0]+dx*(mx[0]-mn[0]),
                                  mn[1]+dy*(mx[1]-mn[1]),
                                  mn[2]+dz*(mx[2]-mn[2])]
                                 for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)],
                                dtype=np.float64)
            cam = corners @ R + T
            np.testing.assert_allclose(
                cam.mean(axis=0), [0.0, 0.0, 0.0], atol=1e-10,
                err_msg=f"{name}: corner centroid not at camera origin for offset {offset}"
            )
