"""Shared cursor position state updated by the /cursor WebSocket.

The browser sends normalised device coordinates (NDC) in the range [-1, 1]
on every mousemove.  The controller reads them to compute a cursor-based
rotation pivot instead of always rotating around the model bounding-box
centre.

NDC convention (matches WebGL / clip-space):
  (-1, -1) = bottom-left of the 3-D viewport
  ( 1,  1) = top-right
  ( 0,  0) = centre
"""
from __future__ import annotations

# Mutable list so callers can update in-place without rebinding names.
ndc: list[float] = [0.0, 0.0]   # [nx, ny]

# Set to True once the /cursor WebSocket has delivered at least one position.
# Until then, _cursor_pivot falls back to model-bbox centre so that the
# rotation feels the same as before the feature was added.
active: bool = False
