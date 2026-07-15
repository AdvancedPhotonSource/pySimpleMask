# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Tests for the Rigaku 64-bit sparse binary loaders."""

import numpy as np

from pysimplemask.core.reader.formats.rigaku import (
    Rigaku3MDataset,
    RigakuDataset,
    convert_sparse,
    get_number_of_frames_from_binfile,
)


def _word(frame, index, count):
    return (
        (np.uint64(frame) << np.uint64(40))
        | (np.uint64(index) << np.uint64(16))
        | np.uint64(count)
    )


def test_convert_sparse_roundtrip():
    raw = np.array([_word(0, 5, 7), _word(3, 100, 11)], dtype=np.uint64)
    index, frame, count = convert_sparse(raw)
    assert list(index) == [5, 100]
    assert list(frame) == [0, 3]
    assert list(count) == [7, 11]


def test_convert_sparse_count_keeps_full_12_bits():
    # Regression guard: count must keep the full 12-bit range (max 4095),
    # not be truncated to 8 bits (255) as in the original buggy code.
    raw = np.array([_word(0, 0, 4095)], dtype=np.uint64)
    index, frame, count = convert_sparse(raw)
    assert int(count[0]) == 4095


def test_get_scattering_mean(make_rigaku):
    path = make_rigaku(
        [(0, 0, 0, 5), (0, 1, 2, 10), (1, 0, 0, 3), (2, 0, 0, 1)]
    )
    ds = RigakuDataset(path)
    img = ds.get_scattering()
    assert img.shape == (512, 1024)
    assert img.dtype == np.float32
    assert np.isclose(img[0, 0], (5 + 3 + 1) / 3)
    assert np.isclose(img[1, 2], 10 / 3)


def test_get_scattering_frame_subset(make_rigaku):
    path = make_rigaku(
        [(0, 0, 0, 5), (0, 1, 2, 10), (1, 0, 0, 3), (2, 0, 0, 1)]
    )
    ds = RigakuDataset(path)
    img = ds.get_scattering(num_frames=2, begin_idx=0)
    assert np.isclose(img[0, 0], (5 + 3) / 2)
    assert np.isclose(img[1, 2], 10 / 2)


def test_get_scattering_module_index_offset_fold(make_rigaku):
    # With ncols=1024, row=1024 col=0 -> index 1024*1024, col=5 -> 1024*1024+5.
    # Both >= 1024*1024, so the loader folds them to local indices 0 and 5.
    path = make_rigaku([(0, 1024, 0, 7), (0, 1024, 5, 3)])
    ds = RigakuDataset(path)
    img = ds.get_scattering()
    assert np.isclose(img[0, 0], 7)
    assert np.isclose(img[0, 5], 3)


def test_get_number_of_frames_from_binfile(make_rigaku):
    path = make_rigaku([(0, 0, 0, 1), (4, 0, 0, 1)])
    assert get_number_of_frames_from_binfile(path) == 5


def test_rigaku_3m_stitching_module_order(make_rigaku_3m):
    events_by_suffix = {
        0: [(0, 0, 0, 10)],
        1: [(0, 0, 0, 11)],
        2: [(0, 0, 0, 12)],
        3: [(0, 0, 0, 13)],
        4: [(0, 0, 0, 14)],
        5: [(0, 0, 0, 15)],
    }
    path = make_rigaku_3m(events_by_suffix)
    ds = Rigaku3MDataset(path)
    assert ds.det_size == (1676, 2100)

    img = ds.get_scattering()
    assert img.shape == (1676, 2100)

    # Block (r,c) origin: v0 = r*(512+70), h0 = c*(1024+52).
    # Module order is (5, 0, 4, 1, 3, 2) row-major across the 3x2 layout.
    expected = {
        (0, 0): 15,  # suffix 5
        (0, 1): 10,  # suffix 0
        (1, 0): 14,  # suffix 4
        (1, 1): 11,  # suffix 1
        (2, 0): 13,  # suffix 3
        (2, 1): 12,  # suffix 2
    }
    for (r, c), value in expected.items():
        assert np.isclose(img[r * 582, c * 1076], value), (r, c, value)
