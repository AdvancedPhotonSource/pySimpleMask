# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Extract pyqtgraph ROI geometry into Qt-free core polygons."""

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QPainterPath

from pysimplemask.core.rasterize import RoiPolygon


def extract_roi_geometry(roi_dict, image_item):
    """Return a list of RoiPolygon for every 'roi_*' item in ``roi_dict``.

    roi_dict: mapping of key -> pyqtgraph ROI (each having a ``.sl_mode`` attribute).
    image_item: the pyqtgraph ImageItem the ROIs are drawn over.

    Each ROI's outline is mapped into image-item coordinates and sampled into a
    polygon of ``(row, col)`` vertices, which the core rasterizer fills.
    """
    rois = []
    for key, roi in roi_dict.items():
        if not key.startswith("roi_"):
            continue

        # EllipseROI.shape() returns a coarse 24-point polygon (a Qt hit-test
        # workaround); build a true ellipse path for a smooth outline instead.
        if isinstance(roi, pg.EllipseROI):
            path = QPainterPath()
            path.addEllipse(roi.boundingRect())
        else:
            path = roi.shape()
        path = roi.mapToItem(image_item, path)

        for polygon in path.toSubpathPolygons():
            verts = np.array([[pt.y(), pt.x()] for pt in polygon], dtype=float)
            if verts.shape[0] >= 3:
                rois.append(RoiPolygon(verts, roi.sl_mode))
    return rois
