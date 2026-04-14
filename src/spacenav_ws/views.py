"""Standard Onshape view affine matrices for SpaceMouse navigation.

Onshape uses a **Z-up, Y-forward** coordinate system:
  +X = right
  +Y = into the screen (forward / depth)
  +Z = up

This was determined empirically: the identity matrix produces a Top view
(camera looking straight down -Z), and Rx(-90°) produces the Front view
(camera looking in the +Y direction).

All matrices are flat 16-element row-major 4×4 lists suitable for writing
directly to Onshape's ``view.affine`` property.
"""

from __future__ import annotations
import numpy as np


def _look_at(look: tuple | list, up: tuple | list = (0.0, 0.0, 1.0)) -> list[float]:
    """Build a 4×4 view matrix from a look direction and an up hint.

    The camera looks in ``-Z`` camera-space, so:
        cam_Z_world = -normalise(look)
        cam_X_world = normalise(look × up)
        cam_Y_world = cam_Z_world × cam_X_world
    """
    d = np.asarray(look, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64)
    d = d / np.linalg.norm(d)

    cam_x = np.cross(d, u)
    if np.linalg.norm(cam_x) < 1e-9:  # look ‖ up — use fallback
        u = np.array([0.0, 1.0, 0.0])
        cam_x = np.cross(d, u)
    cam_x /= np.linalg.norm(cam_x)

    cam_z = -d  # camera -Z = look direction
    cam_y = np.cross(cam_z, cam_x)
    cam_y /= np.linalg.norm(cam_y)

    R = np.array([cam_x, cam_y, cam_z])  # rows = camera axes in world space
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = R
    return m.reshape(-1).tolist()


# ---------------------------------------------------------------------------
# Standard named views (Onshape Z-up, Y-forward world)
# ---------------------------------------------------------------------------
# Empirically confirmed on SpaceMouse Enterprise:
#   identity matrix  →  Top view   (camera at +Z looking straight down)
#   Rx(-90°)         →  Front view (camera at -Y looking in +Y direction)

_Z_UP = (0.0, 0.0, 1.0)

VIEW_MATRICES: dict[str, list[float]] = {
    # ── primary orthographic views ──────────────────────────────────────
    "top": _look_at((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),  # look down  (-Z), image-up = +Y
    "bottom": _look_at((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),  # look up    (+Z), image-up = -Y (front faces down)
    "front": _look_at((0.0, 1.0, 0.0), _Z_UP),  # look forward (+Y), up = +Z
    "back": _look_at((0.0, -1.0, 0.0), _Z_UP),  # look backward(-Y), up = +Z
    "right": _look_at((-1.0, 0.0, 0.0), _Z_UP),  # from +X side, look in -X
    "left": _look_at((1.0, 0.0, 0.0), _Z_UP),  # from -X side, look in +X
    # ── isometric views ─────────────────────────────────────────────────
    # ISO1: front-right-top corner  (look from +X, -Y, +Z corner toward origin)
    "iso1": _look_at((-1.0, 1.0, -1.0), _Z_UP),
    # ISO2: front-left-top corner   (look from -X, -Y, +Z corner toward origin)
    "iso2": _look_at((1.0, 1.0, -1.0), _Z_UP),
}


def get_view_matrix(name: str) -> list[float] | None:
    """Return the flat 16-element affine matrix for a named view, or None."""
    return VIEW_MATRICES.get(name)
