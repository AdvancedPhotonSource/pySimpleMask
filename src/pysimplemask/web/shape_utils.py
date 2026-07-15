# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Convert Plotly relayoutData shapes to a boolean numpy keep-mask."""

from __future__ import annotations

import re

import numpy as np

from pysimplemask.core.rasterize import (
    RoiPolygon,
    circle_vertices,
    rectangle_vertices,
    rasterize,
)


def _parse_svg_path(path: str) -> np.ndarray:
    """Parse SVG path ``"M x,y L x,y ... Z"`` into ``(N, 2)`` (row, col) array.

    Plotly uses (x=col, y=row) convention; we convert to (row, col).
    """
    tokens = re.findall(r"[ML]\s*([-\d.]+),([-\d.]+)", path)
    if not tokens:
        return np.empty((0, 2), dtype=float)
    return np.array([(float(y), float(x)) for x, y in tokens], dtype=float)


def plotly_shapes_to_mask(
    shapes: list[dict],
    detector_shape: tuple[int, int],
    mode: str = "exclusive",
) -> np.ndarray:
    """Convert Plotly ``relayoutData["shapes"]`` entries to a boolean keep-mask.

    Args:
        shapes: List of shape dicts from Plotly.  Each has ``"type"``
            (``"rect"``, ``"circle"``, or ``"path"``) and coordinate fields.
            Plotly convention: ``x`` = column, ``y`` = row.
        detector_shape: ``(H, W)`` of the detector array.
        mode: ``"exclusive"`` masks pixels inside ROIs (keep = ``False``
            inside); ``"inclusive"`` keeps only pixels inside ROIs (keep =
            ``False`` outside).

    Returns:
        Boolean ``np.ndarray`` of shape ``detector_shape``; ``True`` = keep.
    """
    rois: list[RoiPolygon] = []

    for shape in shapes:
        stype = shape.get("type", "")
        if stype == "rect":
            x0, y0 = float(shape["x0"]), float(shape["y0"])
            x1, y1 = float(shape["x1"]), float(shape["y1"])
            # Plotly: x=col, y=row; rectangle_vertices: center=(row,col), size=(h,w)
            cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
            h, w = abs(y1 - y0), abs(x1 - x0)
            verts = rectangle_vertices(center=(cy, cx), size=(h, w))
            rois.append(RoiPolygon(verts, mode=mode))

        elif stype == "circle":
            x0, y0 = float(shape["x0"]), float(shape["y0"])
            x1, y1 = float(shape["x1"]), float(shape["y1"])
            cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
            radius = abs(x1 - x0) / 2.0
            verts = circle_vertices(center=(cy, cx), radius=radius)
            rois.append(RoiPolygon(verts, mode=mode))

        elif stype == "path":
            verts = _parse_svg_path(shape.get("path", ""))
            if len(verts) >= 3:
                rois.append(RoiPolygon(verts, mode=mode))

    if not rois:
        return np.ones(detector_shape, dtype=bool)

    return rasterize(detector_shape, rois)
