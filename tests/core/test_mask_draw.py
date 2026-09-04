# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import numpy as np

from pysimplemask.core.mask import MaskDraw


def test_mask_draw_evaluate_sets_zero_loc_from_arr():
    md = MaskDraw(shape=(4, 4))
    arr = np.zeros((4, 4), dtype=bool)
    arr[1, 1] = True
    md.evaluate(arr=arr)
    assert not md.get_mask()[1, 1]
    assert md.get_mask()[0, 0]


def test_mask_draw_defaults_to_no_groups():
    md = MaskDraw(shape=(4, 4))
    md.evaluate(arr=np.zeros((4, 4), dtype=bool))
    assert md.group_index_map is None
    assert md.num_groups == 0


def test_mask_draw_stores_group_index_map_and_num_groups():
    md = MaskDraw(shape=(4, 4))
    gmap = np.zeros((4, 4), dtype=np.uint32)
    gmap[0:2, 0:2] = 1
    gmap[2:4, 2:4] = 2
    md.evaluate(arr=np.zeros((4, 4), dtype=bool), group_index_map=gmap)
    np.testing.assert_array_equal(md.group_index_map, gmap)
    assert md.num_groups == 2


def test_mask_draw_find_overlaps_none_for_disjoint_groups():
    md = MaskDraw(shape=(4, 4))
    row_masks = {
        1: np.array([[True, True, False, False]] * 4),
        2: np.array([[False, False, True, True]] * 4),
    }
    md.evaluate(arr=np.zeros((4, 4), dtype=bool), row_masks=row_masks)
    assert md.find_overlaps() == []


def test_mask_draw_find_overlaps_detects_overlapping_groups():
    md = MaskDraw(shape=(4, 4))
    row_masks = {
        1: np.array([[True, True, True, False]] * 4),
        2: np.array([[False, True, True, True]] * 4),
    }
    md.evaluate(arr=np.zeros((4, 4), dtype=bool), row_masks=row_masks)
    # overlap columns {1, 2}, 4 rows each -> 8 pixels
    assert md.find_overlaps() == [(1, 2, 8)]


def test_mask_draw_find_overlaps_respects_mask_argument():
    md = MaskDraw(shape=(4, 4))
    row_masks = {
        1: np.array([[True, True, True, False]] * 4),
        2: np.array([[False, True, True, True]] * 4),
    }
    md.evaluate(arr=np.zeros((4, 4), dtype=bool), row_masks=row_masks)
    external_mask = np.zeros((4, 4), dtype=bool)  # excludes everything
    assert md.find_overlaps(mask=external_mask) == []


def test_mask_draw_describe_constraint():
    md = MaskDraw(shape=(4, 4))
    desc = md.describe_constraint(2)
    assert "2" in desc
