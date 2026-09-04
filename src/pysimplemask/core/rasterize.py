# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Rasterize ROI geometry to a boolean keep-mask (Qt-free, skimage-based)."""

from dataclasses import dataclass

import numpy as np
from skimage.draw import polygon2mask


@dataclass
class RoiPolygon:
    """A polygon ROI in image space.

    vertices: (N, 2) array of (row, col) points.
    mode: "inclusive" (region to keep) or "exclusive" (region to remove).
    group_index: 1-based group label for an inclusive ROI tracked in the GUI's
        draw-group table; 0 means untracked (the default for every ROI type
        and for exclusive ROIs, which are never grouped).
    """

    vertices: np.ndarray
    mode: str = "exclusive"
    group_index: int = 0


def circle_vertices(center, radius, n=180):
    """Vertices of a circle. center=(row, col)."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rows = center[0] + radius * np.sin(theta)
    cols = center[1] + radius * np.cos(theta)
    return np.column_stack([rows, cols])


def ellipse_vertices(center, axes, angle_deg=0.0, n=180):
    """Vertices of a (possibly rotated) ellipse. center=(row, col), axes=(a_row, a_col)."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = axes[0] * np.sin(theta)
    c = axes[1] * np.cos(theta)
    a = np.deg2rad(angle_deg)
    rr = center[0] + r * np.cos(a) - c * np.sin(a)
    cc = center[1] + r * np.sin(a) + c * np.cos(a)
    return np.column_stack([rr, cc])


def rectangle_vertices(center, size, angle_deg=0.0):
    """Four corners of a (possibly rotated) rectangle. center=(row, col), size=(h, w)."""
    h, w = size[0] / 2.0, size[1] / 2.0
    corners = np.array([[-h, -w], [-h, w], [h, w], [h, -w]])
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rotated = corners @ rot.T
    return rotated + np.asarray(center)


def line_vertices(p0, p1, width):
    """A rectangle polygon of given width along the segment p0->p1. Points are (row, col)."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    direction = p1 - p0
    length = np.hypot(*direction)
    if length == 0:
        normal = np.array([0.0, 0.0])
    else:
        normal = np.array([-direction[1], direction[0]]) / length * (width / 2.0)
    return np.array([p0 + normal, p1 + normal, p1 - normal, p0 - normal])


def rasterize(shape, rois):
    """Combine ROI polygons into a boolean keep-mask of the given image shape.

    keep = (NOT any exclusive) AND (any inclusive OR no inclusive present).
    """
    exclusive = np.zeros(shape, dtype=bool)
    inclusive = np.zeros(shape, dtype=bool)
    has_inclusive = False

    for roi in rois:
        filled = polygon2mask(shape, np.asarray(roi.vertices, dtype=float))
        if roi.mode == "inclusive":
            has_inclusive = True
            inclusive |= filled
        else:
            exclusive |= filled

    keep_inclusive = inclusive if has_inclusive else np.ones(shape, dtype=bool)
    return np.logical_and(~exclusive, keep_inclusive)


def group_index_map(shape, rois):
    """Label each pixel with the group_index of the inclusive ROI covering it.

    Exclusive ROIs and ungrouped inclusive ROIs (group_index == 0) are ignored.
    On overlap between grouped ROIs, last-write-wins (same convention as
    MaskParameter.group_index_map in core/mask.py).
    """
    gmap = np.zeros(shape, dtype=np.uint32)
    for roi in rois:
        if roi.mode != "inclusive" or not roi.group_index:
            continue
        filled = polygon2mask(shape, np.asarray(roi.vertices, dtype=float))
        gmap[filled] = roi.group_index
    return gmap


def group_masks(shape, rois):
    """Per-group raw filled masks, keyed by group_index, for inclusive ROIs
    with group_index != 0 — kept separate (unlike group_index_map) so overlap
    between groups can be detected before last-write-wins collapses them.
    ROIs sharing the same group_index are OR-ed together.
    """
    masks = {}
    for roi in rois:
        if roi.mode != "inclusive" or not roi.group_index:
            continue
        filled = polygon2mask(shape, np.asarray(roi.vertices, dtype=float))
        if roi.group_index in masks:
            masks[roi.group_index] |= filled
        else:
            masks[roi.group_index] = filled
    return masks
