# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Unit tests for shape_utils — Plotly shape → numpy mask conversion."""

from pysimplemask.web.shape_utils import plotly_shapes_to_mask

SHAPE = (20, 20)


def test_rect_exclusive_masks_inside():
    shapes = [{"type": "rect", "x0": 5.0, "y0": 5.0, "x1": 10.0, "y1": 10.0}]
    mask = plotly_shapes_to_mask(shapes, SHAPE, mode="exclusive")
    assert mask.shape == SHAPE
    assert not mask[7, 7]   # inside rect → masked out
    assert mask[0, 0]       # outside → kept


def test_rect_inclusive_keeps_inside():
    shapes = [{"type": "rect", "x0": 5.0, "y0": 5.0, "x1": 15.0, "y1": 15.0}]
    mask = plotly_shapes_to_mask(shapes, SHAPE, mode="inclusive")
    assert mask[10, 10]     # inside → kept
    assert not mask[0, 0]   # outside → masked


def test_circle_exclusive_masks_inside():
    # bbox (6,6)→(14,14) → center=(10,10), radius=4
    shapes = [{"type": "circle", "x0": 6.0, "y0": 6.0, "x1": 14.0, "y1": 14.0}]
    mask = plotly_shapes_to_mask(shapes, SHAPE, mode="exclusive")
    assert not mask[10, 10]  # center → masked out
    assert mask[0, 0]        # corner → kept


def test_polygon_path_parses():
    # Triangle in (x=col, y=row): (5,5),(15,5),(10,15) → rows 5-15, cols 5-15
    path = "M 5,5 L 15,5 L 10,15 Z"
    shapes = [{"type": "path", "path": path}]
    mask = plotly_shapes_to_mask(shapes, SHAPE, mode="exclusive")
    assert mask.shape == SHAPE
    assert not mask[8, 9]   # inside triangle → masked


def test_multiple_shapes_both_regions_masked():
    shapes = [
        {"type": "rect", "x0": 0.0, "y0": 0.0, "x1": 4.0, "y1": 4.0},
        {"type": "rect", "x0": 15.0, "y0": 15.0, "x1": 19.0, "y1": 19.0},
    ]
    mask = plotly_shapes_to_mask(shapes, SHAPE, mode="exclusive")
    assert not mask[2, 2]    # inside first rect
    assert not mask[17, 17]  # inside second rect
    assert mask[10, 10]      # between rects → kept


def test_empty_shapes_returns_all_true():
    mask = plotly_shapes_to_mask([], SHAPE)
    assert mask.all()


def test_out_of_bounds_shape_no_crash():
    shapes = [{"type": "rect", "x0": -5.0, "y0": -5.0, "x1": 30.0, "y1": 30.0}]
    mask = plotly_shapes_to_mask(shapes, SHAPE, mode="exclusive")
    assert mask.shape == SHAPE  # no exception, valid shape returned
