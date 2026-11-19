# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Extract pyqtgraph ROI geometry into Qt-free core polygons."""

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QPainterPath

from pysimplemask.core.rasterize import RoiPolygon, rasterize


def roi_outline_vertices(roi, image_item):
    """Outline(s) of one pyqtgraph ROI in image-item coordinates.

    Returns a list of ``(N, 2)`` float arrays of ``(row, col)`` vertices, one
    per closed sub-path (a plain shape gives exactly one); sub-paths with
    fewer than 3 vertices are dropped.
    """
    # EllipseROI.shape() returns a coarse 24-point polygon (a Qt hit-test
    # workaround); build a true ellipse path for a smooth outline instead.
    if isinstance(roi, pg.EllipseROI):
        path = QPainterPath()
        path.addEllipse(roi.boundingRect())
    else:
        path = roi.shape()
    path = roi.mapToItem(image_item, path)

    outlines = []
    for polygon in path.toSubpathPolygons():
        verts = np.array([[pt.y(), pt.x()] for pt in polygon], dtype=float)
        if verts.shape[0] >= 3:
            outlines.append(verts)
    return outlines


def roi_region(roi, image_item, shape):
    """Boolean array of ``shape`` that is True inside one pyqtgraph ROI (e.g.
    the Outlier tab's rectangle), rasterized by the Qt-free core."""
    polygons = [
        RoiPolygon(verts, "inclusive") for verts in roi_outline_vertices(roi, image_item)
    ]
    if not polygons:
        # rasterize() keeps everything when no inclusive ROI is present.
        return np.zeros(shape, dtype=bool)
    return rasterize(shape, polygons)


def extract_roi_geometry(roi_dict, image_item, group_index_lookup=None):
    """Return a list of RoiPolygon for every 'roi_*' item in ``roi_dict``.

    roi_dict: mapping of key -> pyqtgraph ROI (each having a ``.sl_mode`` attribute).
    image_item: the pyqtgraph ImageItem the ROIs are drawn over.
    group_index_lookup: optional {roi_key: 1-based group_index}, e.g. from the
        Draw tab's tracked-inclusive-ROI table. Keys not present default to 0.

    Each ROI's outline is mapped into image-item coordinates and sampled into a
    polygon of ``(row, col)`` vertices, which the core rasterizer fills.
    Items under other keys (the center marker, the Outlier tab's region
    rectangle) are not draw ROIs and are skipped.
    """
    group_index_lookup = group_index_lookup or {}
    rois = []
    for key, roi in roi_dict.items():
        if not key.startswith("roi_"):
            continue
        group_index = group_index_lookup.get(key, 0)
        for verts in roi_outline_vertices(roi, image_item):
            rois.append(RoiPolygon(verts, roi.sl_mode, group_index))
    return rois
