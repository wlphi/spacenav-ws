"""Built-in SVG icons for the default hotkey grid.

Each icon is a 44×44 SVG designed for direct rendering on the dark LCD
(no brightness-boost adaptation needed — use ``"no_adapt": True`` in the
hotkey dict so ``render_hotkey_grid`` skips ``_adapt_icon``).

Cube geometry
-------------
Isometric projection, 44×44 viewBox, cube side = 13 px:

              T (22,10)
             / \\
            /   \\
    TL(10.7,16.5) - C(22,23) - TR(33.3,16.5)
            \\   /           \\   /
             \\ /             \\ /
    BL(10.7,29.5)  ---   BR(33.3,29.5)
              \\ /
               B (22,36)

Three visible faces:
  • TOP   (diamond):  T – TR – C – TL       → Z+ face
  • RIGHT (rhombus):  TR – BR – B – C       → X+ face
  • FRONT (rhombus):  TL – C – B – BL       → Y− face  (front in Onshape Y-forward)
"""

# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------
_T = "22,10"
_TR = "33.3,16.5"
_TL = "10.7,16.5"
_C = "22,23"
_BR = "33.3,29.5"
_BL = "10.7,29.5"
_B = "22,36"

_STROKE = "#4a6070"
_SW = "0.8"


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_LIT = "#5c8cc8"  # primary highlight  — medium blue
_OPP = "#3c9080"  # opposite-direction — teal
_DIM_T = "#344258"  # unlit top face
_DIM_R = "#3c4e62"  # unlit right face
_DIM_F = "#2e3c4e"  # unlit front face (darkest)


# ---------------------------------------------------------------------------
# Primitive builders
# ---------------------------------------------------------------------------


def _cube(top: str, right: str, front: str) -> bytes:
    """Return a 44×44 isometric-cube SVG with the given face fill colours."""
    s = f'stroke="{_STROKE}" stroke-width="{_SW}" stroke-linejoin="round"'
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 44 44">'
        f'<polygon points="{_T} {_TR} {_C} {_TL}" fill="{top}" {s}/>'
        f'<polygon points="{_TR} {_BR} {_B} {_C}" fill="{right}" {s}/>'
        f'<polygon points="{_TL} {_C} {_B} {_BL}" fill="{front}" {s}/>'
        "</svg>"
    ).encode()


def _fit() -> bytes:
    """Fit-to-view icon: four L-shaped corner markers."""
    p, c = 7, 9  # padding, corner-arm length
    x1, y1 = p, p
    x2, y2 = 44 - p, 44 - p
    d = (
        f"M{x1},{y1 + c} L{x1},{y1} L{x1 + c},{y1} "
        f"M{x2 - c},{y1} L{x2},{y1} L{x2},{y1 + c} "
        f"M{x2},{y2 - c} L{x2},{y2} L{x2 - c},{y2} "
        f"M{x1 + c},{y2} L{x1},{y2} L{x1},{y2 - c}"
    )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 44 44">'
        f'<path d="{d}" fill="none" stroke="{_LIT}"'
        ' stroke-width="2.5" stroke-linecap="round"/>'
        "</svg>"
    ).encode()


# ---------------------------------------------------------------------------
# Icon table
# ---------------------------------------------------------------------------
# Colour convention:
#   blue  (_LIT) = camera is looking AT this face
#   teal  (_OPP) = camera is looking from the OPPOSITE side of this face
#   dark          = face not relevant to this view

VIEW_ICONS: dict[str, bytes] = {
    # Isometric — all three faces lit with stepped depth shading
    "view_iso1": _cube("#4e82b8", "#3d6ea0", "#2e5888"),
    "view_iso2": _cube("#3d6ea0", "#4e82b8", "#2e5888"),
    # Orthographic — one face highlighted in blue; opposite uses teal
    "view_top": _cube(_LIT, _DIM_R, _DIM_F),
    "view_bottom": _cube(_OPP, _DIM_R, _DIM_F),
    "view_right": _cube(_DIM_T, _LIT, _DIM_F),
    "view_left": _cube(_DIM_T, _OPP, _DIM_F),
    "view_front": _cube(_DIM_T, _DIM_R, _LIT),
    "view_back": _cube(_DIM_T, _DIM_R, _OPP),
    # Fit to extents
    "fit": _fit(),
}
