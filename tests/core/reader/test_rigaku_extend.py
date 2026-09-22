# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Tests for the RigakuExtend 64-bit sparse binary loader."""

import numpy as np

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
