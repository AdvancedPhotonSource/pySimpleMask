# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Tests for the RigakuExtend 64-bit sparse binary loader."""

import numpy as np
import pytest

from pysimplemask.core.reader.formats.rigaku_extend import (
    MODULE_SHAPE,
    RigakuExtendDataset,
    convert_sparse,
    get_number_of_frames_from_binfile,
)


def _word(frame, index, count):
    return (
        (np.uint64(frame) << np.uint64(32))
        | (np.uint64(index) << np.uint64(12))
        | np.uint64(count)
    )


def test_convert_sparse_roundtrip():
    raw = np.array([_word(0, 5, 7), _word(3, 100, 11)], dtype=np.uint64)
    index, frame, count = convert_sparse(raw)
    assert list(index) == [5, 100]
    assert list(frame) == [0, 3]
    assert list(count) == [7, 11]


def test_convert_sparse_count_keeps_full_12_bits():
    # RigakuExtend count is 12 bits (max 4095); must not be truncated to 8 bits.
    raw = np.array([_word(0, 0, 4095)], dtype=np.uint64)
    index, frame, count = convert_sparse(raw)
    assert int(count[0]) == 4095


def test_convert_sparse_large_frame_index():
    # Frame occupies the full top 32 bits, well beyond the 24-bit Rigaku field.
    raw = np.array([_word(2**24 + 10, 0, 1)], dtype=np.uint64)
    index, frame, count = convert_sparse(raw)
    assert int(frame[0]) == 2**24 + 10


def test_module_shape_is_512x448():
    assert MODULE_SHAPE == (512, 448)


def test_get_scattering_mean(make_rigaku_extend):
    path = make_rigaku_extend(
        [(0, 0, 0, 5), (0, 1, 2, 10), (1, 0, 0, 3), (2, 0, 0, 1)]
    )
    ds = RigakuExtendDataset(path)
    img = ds.get_scattering()
    assert img.shape == (512, 448)
    assert img.dtype == np.float32
    assert np.isclose(img[0, 0], (5 + 3 + 1) / 3)
    assert np.isclose(img[1, 2], 10 / 3)


def test_get_scattering_frame_subset(make_rigaku_extend):
    path = make_rigaku_extend(
        [(0, 0, 0, 5), (0, 1, 2, 10), (1, 0, 0, 3), (2, 0, 0, 1)]
    )
    ds = RigakuExtendDataset(path)
    img = ds.get_scattering(num_frames=2, begin_idx=0)
    assert np.isclose(img[0, 0], (5 + 3) / 2)
    assert np.isclose(img[1, 2], 10 / 2)


def test_get_number_of_frames_from_binfile(make_rigaku_extend):
    path = make_rigaku_extend([(0, 0, 0, 1), (4, 0, 0, 1)])
    assert get_number_of_frames_from_binfile(path) == 5


# ---------------------------------------------------------------------------
# get_frames_mean: browsing the frames a few at a time
# ---------------------------------------------------------------------------

# frames 2 and 5 hold no event, and the pixel (0, 0) is hit twice in frame 3
_BROWSE_EVENTS = [
    (0, 0, 0, 5),
    (0, 1, 2, 10),
    (1, 0, 0, 3),
    (3, 0, 0, 2),
    (3, 7, 9, 4),
    (3, 0, 0, 6),
    (4, 1, 2, 1),
    (6, 511, 447, 9),
]


def test_frames_can_be_browsed_100_at_a_time_by_default(make_rigaku_extend):
    ds = RigakuExtendDataset(make_rigaku_extend(_BROWSE_EVENTS))
    assert ds.num_browsable_frames == 7
    assert ds.default_frame_average == 100


def test_get_frames_mean(make_rigaku_extend):
    ds = RigakuExtendDataset(make_rigaku_extend(_BROWSE_EVENTS))

    img = ds.get_frames_mean(0, 1)
    assert img.shape == (512, 448)
    assert img.dtype == np.float32
    assert img[0, 0] == 5 and img[1, 2] == 10
    assert img.sum() == 15

    img = ds.get_frames_mean(3, 5)
    assert img[0, 0] == (2 + 6) / 2
    assert img[7, 9] == 4 / 2
    assert img[1, 2] == 1 / 2
    assert img.sum() == 13 / 2

    assert not ds.get_frames_mean(2, 3).any()  # a frame without events
    assert ds.get_frames_mean(6, 7)[511, 447] == 9  # the last frame and pixel


@pytest.mark.parametrize("begin_idx, end_idx", [(0, 7), (0, 3), (1, 2), (2, 6), (5, 7)])
def test_get_frames_mean_matches_get_scattering(make_rigaku_extend, begin_idx, end_idx):
    ds = RigakuExtendDataset(make_rigaku_extend(_BROWSE_EVENTS))
    expected = ds.get_scattering(num_frames=end_idx - begin_idx, begin_idx=begin_idx)
    assert np.array_equal(ds.get_frames_mean(begin_idx, end_idx), expected)


@pytest.mark.parametrize("begin_idx, end_idx", [(-1, 2), (3, 3), (4, 2), (5, 8), (7, 8)])
def test_get_frames_mean_rejects_range_outside_frames(
    make_rigaku_extend, begin_idx, end_idx
):
    ds = RigakuExtendDataset(make_rigaku_extend(_BROWSE_EVENTS))
    with pytest.raises(ValueError):
        ds.get_frames_mean(begin_idx, end_idx)
