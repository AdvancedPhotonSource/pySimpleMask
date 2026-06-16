"""Tests for adjacent-box outlier removal."""

import numpy as np

from pysimplemask.core.outlier_removal import outlier_removal_adjacent_boxes


def _uniform_image(shape=(64, 64), value=100.0):
    return np.full(shape, value, dtype=np.float32)


def test_returns_correct_shapes():
    img = _uniform_image()
    mask = np.ones(img.shape, dtype=bool)
    saxs1d, bad_pixels = outlier_removal_adjacent_boxes(img, mask, box_size=16)
    # 64/16 = 4 boxes per dim -> 16 boxes total; saxs1d has 5 rows
    assert saxs1d.ndim == 2
    assert saxs1d.shape[0] == 5
    assert bad_pixels.shape[0] == 2


def test_uniform_image_no_outliers():
    img = _uniform_image((64, 64), value=50.0)
    mask = np.ones(img.shape, dtype=bool)
    _, bad = outlier_removal_adjacent_boxes(img, mask, box_size=16)
    assert bad.shape[1] == 0


def test_single_hot_pixel_detected():
    img = _uniform_image((64, 64), value=10.0)
    mask = np.ones(img.shape, dtype=bool)
    img[5, 5] = 10000.0  # hot pixel in top-left box (box_size=16)
    _, bad = outlier_removal_adjacent_boxes(
        img, mask, box_size=16, method="percentile", cutoff=3.0
    )
    assert bad.shape[1] >= 1
    # The hot pixel coords must appear
    locs = set(zip(bad[0].tolist(), bad[1].tolist()))
    assert (5, 5) in locs


def test_masked_pixels_ignored():
    img = _uniform_image((64, 64), value=10.0)
    mask = np.ones(img.shape, dtype=bool)
    mask[0:16, 0:16] = False  # mask out first box entirely
    saxs1d, _ = outlier_removal_adjacent_boxes(img, mask, box_size=16)
    # 15 boxes remain (16 total minus the fully masked one)
    assert saxs1d.shape[1] == 15


def test_x_axis_is_sorted_box_index():
    rng = np.random.default_rng(42)
    img = rng.uniform(1, 100, size=(64, 64)).astype(np.float32)
    mask = np.ones(img.shape, dtype=bool)
    saxs1d, _ = outlier_removal_adjacent_boxes(img, mask, box_size=16)
    # x values must be strictly increasing (sorted rank)
    assert np.all(np.diff(saxs1d[0]) > 0)


def test_mad_method_works():
    img = _uniform_image((64, 64), value=20.0)
    mask = np.ones(img.shape, dtype=bool)
    img[10, 10] = 9999.0
    _, bad = outlier_removal_adjacent_boxes(
        img, mask, box_size=16, method="mad", cutoff=3.0
    )
    assert bad.shape[1] >= 1


def test_non_divisible_shape_uses_floor():
    # 70 // 16 = 4 boxes per dim -> 16 boxes
    img = _uniform_image((70, 70), value=5.0)
    mask = np.ones(img.shape, dtype=bool)
    saxs1d, _ = outlier_removal_adjacent_boxes(img, mask, box_size=16)
    assert saxs1d.shape[1] == 16  # floor(70/16)^2 = 4^2
