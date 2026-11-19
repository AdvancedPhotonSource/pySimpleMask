# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Tests for adjacent-box outlier removal."""

import numpy as np
import pytest

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
    assert saxs1d.shape[1] == 16
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


def test_boxes_sorted_by_mean():
    rng = np.random.default_rng(42)
    img = rng.uniform(1, 100, size=(64, 64)).astype(np.float32)
    mask = np.ones(img.shape, dtype=bool)
    saxs1d, _ = outlier_removal_adjacent_boxes(img, mask, box_size=16)
    # row 0: integer rank (0,1,2,…) — strictly increasing by construction
    assert np.all(np.diff(saxs1d[0]) > 0)
    # row 4: raw_avg must be non-increasing (brightest box first)
    assert np.all(np.diff(saxs1d[4]) <= 0)


def test_mad_method_works():
    # Use a box with real spread so MAD is non-zero and can flag the hot pixel.
    rng = np.random.default_rng(7)
    img = rng.uniform(10, 30, size=(64, 64)).astype(np.float32)
    mask = np.ones(img.shape, dtype=bool)
    img[10, 10] = 9999.0  # hot pixel in top-left box
    _, bad = outlier_removal_adjacent_boxes(
        img, mask, box_size=16, method="mad", cutoff=3.0
    )
    assert bad.shape[1] >= 1
    locs = set(zip(bad[0].tolist(), bad[1].tolist()))
    assert (10, 10) in locs


def test_non_divisible_shape_uses_floor():
    # 70 // 16 = 4 boxes per dim -> 16 boxes
    img = _uniform_image((70, 70), value=5.0)
    mask = np.ones(img.shape, dtype=bool)
    saxs1d, _ = outlier_removal_adjacent_boxes(img, mask, box_size=16)
    assert saxs1d.shape[1] == 16  # floor(70/16)^2 = 4^2


# ── region (the GUI's Outlier-tab rectangle) ─────────────────────────────────

def _locs(bad):
    return set(zip(bad[0].tolist(), bad[1].tolist()))


def test_region_restricts_detection_to_region():
    img = _uniform_image((64, 64), value=10.0)
    mask = np.ones(img.shape, dtype=bool)
    img[5, 5] = 10000.0     # hot pixel outside the region
    img[40, 40] = 10000.0   # hot pixel inside the region
    region = np.zeros(img.shape, dtype=bool)
    region[32:64, 32:64] = True
    _, bad = outlier_removal_adjacent_boxes(
        img, mask, box_size=16, method="percentile", cutoff=3.0, region=region
    )
    locs = _locs(bad)
    assert (40, 40) in locs
    assert (5, 5) not in locs
    assert all(region[r, c] for r, c in locs)


def test_region_tiles_anchor_at_region_corner_and_cover_remainder():
    img = _uniform_image((64, 64), value=10.0)
    mask = np.ones(img.shape, dtype=bool)
    region = np.zeros(img.shape, dtype=bool)
    region[10:50, 7:30] = True   # 40 rows x 23 cols, box 16 -> 3 x 2 tiles, edges clipped
    saxs1d, _ = outlier_removal_adjacent_boxes(img, mask, box_size=16, region=region)
    assert saxs1d.shape[1] == 6


def test_region_smaller_than_box_is_still_processed():
    img = _uniform_image((64, 64), value=10.0)
    mask = np.ones(img.shape, dtype=bool)
    img[20, 20] = 10000.0
    region = np.zeros(img.shape, dtype=bool)
    region[18:24, 18:24] = True   # 6 x 6 region, box 16: one clipped tile
    saxs1d, bad = outlier_removal_adjacent_boxes(
        img, mask, box_size=16, method="percentile", cutoff=3.0, region=region
    )
    assert saxs1d.shape[1] == 1
    assert (20, 20) in _locs(bad)


def test_region_respects_mask_and_validates_shape():
    img = _uniform_image((64, 64), value=10.0)
    mask = np.ones(img.shape, dtype=bool)
    img[40, 40] = 10000.0
    region = np.zeros(img.shape, dtype=bool)
    region[32:64, 32:64] = True
    mask[40, 40] = False   # already masked -> never reported
    _, bad = outlier_removal_adjacent_boxes(
        img, mask, box_size=16, method="percentile", cutoff=3.0, region=region
    )
    assert (40, 40) not in _locs(bad)

    with pytest.raises(ValueError):
        outlier_removal_adjacent_boxes(img, mask, region=np.ones((32, 32), dtype=bool))

    saxs1d, bad = outlier_removal_adjacent_boxes(
        img, mask, region=np.zeros(img.shape, dtype=bool)
    )
    assert saxs1d.shape == (5, 0) and bad.shape == (2, 0)


def test_region_none_matches_whole_image_call():
    rng = np.random.default_rng(7)
    img = rng.uniform(1, 100, size=(64, 48)).astype(np.float32)
    img[3, 3] = 1e5
    mask = np.ones(img.shape, dtype=bool)
    a = outlier_removal_adjacent_boxes(img, mask, box_size=16)
    b = outlier_removal_adjacent_boxes(img, mask, box_size=16, region=None)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
