# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import numpy as np

from pysimplemask.core.rasterize import (
    RoiPolygon,
    circle_vertices,
    rasterize,
    rectangle_vertices,
)


def test_no_rois_keeps_everything():
    keep = rasterize((10, 10), [])
    assert keep.all()


def test_exclusive_rectangle_removes_region():
    verts = rectangle_vertices(center=(5, 5), size=(4, 4), angle_deg=0.0)
    keep = rasterize((10, 10), [RoiPolygon(verts, "exclusive")])
    assert not keep[5, 5]
    assert keep[0, 0]


def test_inclusive_only_keeps_inside():
    verts = rectangle_vertices(center=(5, 5), size=(4, 4), angle_deg=0.0)
    keep = rasterize((10, 10), [RoiPolygon(verts, "inclusive")])
    assert keep[5, 5]
    assert not keep[0, 0]


def test_circle_vertices_form_disk():
    verts = circle_vertices(center=(10, 10), radius=5, n=180)
    keep = rasterize((20, 20), [RoiPolygon(verts, "inclusive")])
    assert keep[10, 10]
    assert not keep[0, 0]
    assert abs(keep.sum() - np.pi * 25) < 25
