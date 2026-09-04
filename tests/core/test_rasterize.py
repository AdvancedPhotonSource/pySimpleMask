# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import numpy as np

from pysimplemask.core.rasterize import (
    RoiPolygon,
    circle_vertices,
    group_index_map,
    group_masks,
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


def test_roi_polygon_group_index_defaults_to_zero():
    verts = rectangle_vertices(center=(5, 5), size=(4, 4))
    assert RoiPolygon(verts, "inclusive").group_index == 0


def test_group_index_map_labels_inclusive_region():
    verts = rectangle_vertices(center=(5, 5), size=(4, 4), angle_deg=0.0)
    roi = RoiPolygon(verts, "inclusive", group_index=3)
    gmap = group_index_map((10, 10), [roi])
    assert gmap[5, 5] == 3
    assert gmap[0, 0] == 0


def test_group_index_map_ignores_exclusive_rois():
    verts = rectangle_vertices(center=(5, 5), size=(4, 4), angle_deg=0.0)
    roi = RoiPolygon(verts, "exclusive", group_index=1)
    gmap = group_index_map((10, 10), [roi])
    assert gmap.max() == 0


def test_group_index_map_ignores_ungrouped_inclusive_rois():
    """An inclusive ROI never assigned a group_index (still 0) must not show
    up in the map, so untracked draws don't get mistaken for group 0."""
    verts = rectangle_vertices(center=(5, 5), size=(4, 4), angle_deg=0.0)
    roi = RoiPolygon(verts, "inclusive")  # group_index defaults to 0
    gmap = group_index_map((10, 10), [roi])
    assert gmap.max() == 0


def test_group_index_map_last_write_wins_on_overlap():
    verts_a = rectangle_vertices(center=(5, 5), size=(6, 6), angle_deg=0.0)
    verts_b = rectangle_vertices(center=(6, 6), size=(6, 6), angle_deg=0.0)
    roi_a = RoiPolygon(verts_a, "inclusive", group_index=1)
    roi_b = RoiPolygon(verts_b, "inclusive", group_index=2)
    gmap = group_index_map((10, 10), [roi_a, roi_b])
    assert gmap[6, 6] == 2  # overlap region: roi_b (drawn later) wins
    assert gmap[2, 2] == 1  # roi_a-only region (roi_a: rows/cols 2..8, roi_b: 3..9)


def test_group_masks_keys_by_group_index_and_preserves_overlap():
    """Unlike group_index_map, group_masks keeps each group's own raw filled
    region so overlap between groups can be detected before last-write-wins."""
    verts_a = rectangle_vertices(center=(5, 5), size=(6, 6), angle_deg=0.0)
    verts_b = rectangle_vertices(center=(6, 6), size=(6, 6), angle_deg=0.0)
    roi_a = RoiPolygon(verts_a, "inclusive", group_index=1)
    roi_b = RoiPolygon(verts_b, "inclusive", group_index=2)
    masks = group_masks((10, 10), [roi_a, roi_b])

    assert set(masks) == {1, 2}
    assert masks[1][6, 6] and masks[2][6, 6]  # both claim the overlap region
    assert masks[1][2, 2] and not masks[2][2, 2]


def test_group_masks_ignores_exclusive_and_ungrouped_rois():
    verts = rectangle_vertices(center=(5, 5), size=(4, 4), angle_deg=0.0)
    exclusive = RoiPolygon(verts, "exclusive", group_index=1)
    ungrouped = RoiPolygon(verts, "inclusive")
    masks = group_masks((10, 10), [exclusive, ungrouped])
    assert masks == {}
