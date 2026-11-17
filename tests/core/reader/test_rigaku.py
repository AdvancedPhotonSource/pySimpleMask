# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Tests for the Rigaku 64-bit sparse binary loaders."""

import gc
import threading

import numpy as np
import pytest

from pysimplemask.core.reader.formats import rigaku
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


def test_frames_past_the_end_of_the_file_are_empty(make_rigaku):
    """``total_frames`` makes the dataset longer than its file, as for the
    shorter module files of a 3M dataset."""
    path = make_rigaku([(0, 0, 0, 5), (2, 0, 0, 4)])
    ds = RigakuDataset(path, total_frames=10)
    assert ds.num_frames_total == 10
    assert ds.get_scattering(num_frames=0)[0, 0] == np.float32((5 + 4) / 10)
    assert not ds.get_scattering(num_frames=0, begin_idx=3).any()
    assert not ds.get_frames_mean(5, 8).any()
    # a file that is longer than ``total_frames`` keeps its own length
    assert RigakuDataset(path, total_frames=2).num_frames_total == 3


def test_empty_file_has_no_frames(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    ds = RigakuDataset(str(path))
    assert ds.num_frames_total == 0
    assert ds.num_browsable_frames == 0
    with pytest.raises(ValueError):
        ds.get_scattering()


def test_incomplete_last_word_is_ignored(make_rigaku):
    path = make_rigaku([(0, 0, 0, 5), (1, 1, 2, 10)])
    with open(path, "ab") as f:
        f.write(b"\x01\x02\x03")
    ds = RigakuDataset(path)
    assert ds.num_frames_total == 2
    img = ds.get_scattering(num_frames=0)
    assert img[0, 0] == 5 / 2 and img[1, 2] == 10 / 2
    assert img.sum() == 15 / 2


@pytest.mark.parametrize(
    "events",
    [
        [(0, 0, 0, 1), (1, 600, 0, 1)],  # a pixel past the last row
        [(0, 1024, 0, 1), (1, 0, 5, 1)],  # a module with offset indices, and one without
        [(0, 0, 5, 1), (1, 1024, 0, 1)],  # ... the other way round
    ],
)
def test_event_outside_the_detector_is_an_error(make_rigaku, events):
    ds = RigakuDataset(make_rigaku(events))
    with pytest.raises(ValueError):
        ds.get_scattering(num_frames=0)
    with pytest.raises(ValueError):
        ds.get_frames_mean(0, 2)
    # the frame without it reads fine
    assert ds.get_frames_mean(0, 1).sum() == 1


# ---------------------------------------------------------------------------
# a long range of frames: read in chunks, by threads
# ---------------------------------------------------------------------------

_NUM_FRAMES = 60


def _random_events(seed=0):
    """Up to 5 events in each of 60 frames, a few of which have none; several
    events of a frame may hit the same pixel."""
    rng = np.random.default_rng(seed)
    events = []
    for frame in range(_NUM_FRAMES):
        for _ in range(int(rng.integers(0, 6))):
            row, col = int(rng.integers(0, 4)), int(rng.integers(0, 6))
            events.append((frame, row, col, int(rng.integers(1, 4096))))
    return events


def _expected_mean(events, begin_idx, end_idx):
    img = np.zeros((512, 1024))
    for frame, row, col, count in events:
        if begin_idx <= frame < end_idx:
            img[row, col] += count
    return (img / (end_idx - begin_idx)).astype(np.float32)


@pytest.mark.parametrize("num_workers", [None, 1, 2, 5])
def test_long_range_is_read_in_chunks_by_threads(make_rigaku, monkeypatch, num_workers):
    monkeypatch.setattr(rigaku, "_CHUNK_EVENTS", 7)
    events = _random_events()
    assert len(events) > 100
    ds = RigakuDataset(make_rigaku(events))
    for begin_idx, end_idx in [(0, 60), (0, 31), (17, 60), (20, 45), (33, 34)]:
        img = ds.get_scattering(
            num_frames=end_idx - begin_idx, begin_idx=begin_idx, num_processes=num_workers
        )
        assert np.array_equal(img, _expected_mean(events, begin_idx, end_idx))
        assert np.array_equal(ds.get_frames_mean(begin_idx, end_idx), img)


@pytest.mark.parametrize("num_workers, num_spans", [(1, 1), (2, 2), (5, 5), (1000, 15)])
def test_each_thread_reads_one_stretch_of_the_events(
    make_rigaku, monkeypatch, num_workers, num_spans
):
    """The events are split evenly between the threads, which are no more
    than there are chunks to read (15, for 100 events in chunks of 7)."""
    monkeypatch.setattr(rigaku, "_CHUNK_EVENTS", 7)
    events = _random_events()[:100]
    ds = RigakuDataset(make_rigaku(events))
    spans = []
    sum_events = ds._sum_events
    monkeypatch.setattr(
        ds, "_sum_events", lambda lo, hi: spans.append((lo, hi)) or sum_events(lo, hi)
    )
    ds.get_scattering(num_frames=0, num_processes=num_workers)
    spans.sort()
    assert len(spans) == num_spans
    assert spans[0][0] == 0 and spans[-1][1] == 100
    assert all(a[1] == b[0] for a, b in zip(spans[:-1], spans[1:]))
    assert max(hi - lo for lo, hi in spans) - min(hi - lo for lo, hi in spans) <= 1


def test_short_range_is_read_in_the_calling_thread(make_rigaku, monkeypatch):
    """Browsing the frames must not start threads for the few events of a
    range that fits one chunk."""
    events = _random_events()
    ds = RigakuDataset(make_rigaku(events))
    readers = set()
    sum_events = ds._sum_events
    monkeypatch.setattr(
        ds,
        "_sum_events",
        lambda lo, hi: readers.add(threading.get_ident()) or sum_events(lo, hi),
    )
    ds.get_frames_mean(10, 20)
    ds.get_scattering(num_frames=0)
    assert readers == {threading.get_ident()}


def test_threads_leave_garbage_collection_to_the_calling_thread(make_rigaku, monkeypatch):
    """As for the HDF direct read: a collection that a worker thread trips
    would destroy the caller's garbage, a GUI's Qt objects among it, from
    that thread."""
    monkeypatch.setattr(rigaku, "_CHUNK_EVENTS", 7)
    events = _random_events()
    ds = RigakuDataset(make_rigaku(events))
    caller = threading.get_ident()
    collecting_threads = set()

    def record(phase, info):
        collecting_threads.add(threading.get_ident())

    threshold = gc.get_threshold()
    gc.callbacks.append(record)
    gc.set_threshold(1, 1, 1)  # collect at every opportunity
    try:
        img = ds.get_scattering(num_frames=0, num_processes=4)
    finally:
        gc.set_threshold(*threshold)
        gc.callbacks.remove(record)

    assert np.array_equal(img, _expected_mean(events, 0, _NUM_FRAMES))
    assert collecting_threads <= {caller}
    assert gc.isenabled()


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
    (6, 511, 1023, 9),
]


def test_frames_can_be_browsed_100_at_a_time_by_default(make_rigaku):
    ds = RigakuDataset(make_rigaku(_BROWSE_EVENTS))
    assert ds.num_browsable_frames == 7
    assert ds.default_frame_average == 100


def test_get_frames_mean(make_rigaku):
    ds = RigakuDataset(make_rigaku(_BROWSE_EVENTS))

    img = ds.get_frames_mean(0, 1)
    assert img.shape == (512, 1024)
    assert img.dtype == np.float32
    assert img[0, 0] == 5 and img[1, 2] == 10
    assert img.sum() == 15

    img = ds.get_frames_mean(3, 5)
    assert img[0, 0] == (2 + 6) / 2
    assert img[7, 9] == 4 / 2
    assert img[1, 2] == 1 / 2
    assert img.sum() == 13 / 2

    assert not ds.get_frames_mean(2, 3).any()  # a frame without events
    assert ds.get_frames_mean(6, 7)[511, 1023] == 9  # the last frame and pixel


@pytest.mark.parametrize("begin_idx, end_idx", [(0, 7), (0, 3), (1, 2), (2, 6), (5, 7)])
def test_get_frames_mean_matches_get_scattering(make_rigaku, begin_idx, end_idx):
    ds = RigakuDataset(make_rigaku(_BROWSE_EVENTS))
    expected = ds.get_scattering(num_frames=end_idx - begin_idx, begin_idx=begin_idx)
    assert np.array_equal(ds.get_frames_mean(begin_idx, end_idx), expected)


@pytest.mark.parametrize("begin_idx, end_idx", [(-1, 2), (3, 3), (4, 2), (5, 8), (7, 8)])
def test_get_frames_mean_rejects_range_outside_frames(make_rigaku, begin_idx, end_idx):
    ds = RigakuDataset(make_rigaku(_BROWSE_EVENTS))
    with pytest.raises(ValueError):
        ds.get_frames_mean(begin_idx, end_idx)


def test_rigaku_3m_get_frames_mean(make_rigaku_3m):
    # the module files differ in length: suffix 2 holds the most frames
    events_by_suffix = {
        0: [(0, 0, 0, 10), (1, 0, 0, 20)],
        1: [(0, 0, 0, 11)],
        2: [(0, 0, 0, 12), (2, 3, 4, 6), (3, 3, 4, 2)],
        3: [(1, 0, 0, 13)],
        4: [(0, 0, 0, 14), (2, 0, 0, 4)],
        5: [(2, 0, 0, 15)],
    }
    ds = Rigaku3MDataset(make_rigaku_3m(events_by_suffix))
    assert ds.num_browsable_frames == 4
    assert ds.default_frame_average == 100

    img = ds.get_frames_mean(2, 4)
    assert img.shape == (1676, 2100)
    assert img.dtype == np.float32
    # Block (r,c) origin: v0 = r*582, h0 = c*1076; module order (5, 0, 4, 1, 3, 2).
    assert img[0, 0] == 15 / 2  # suffix 5
    assert img[582, 0] == 4 / 2  # suffix 4
    assert img[2 * 582 + 3, 1076 + 4] == (6 + 2) / 2  # suffix 2
    assert img.sum() == (15 + 4 + 6 + 2) / 2

    for begin_idx, end_idx in [(0, 4), (0, 1), (1, 3), (3, 4)]:
        expected = ds.get_scattering(
            num_frames=end_idx - begin_idx, begin_idx=begin_idx
        )
        assert np.array_equal(ds.get_frames_mean(begin_idx, end_idx), expected)

    with pytest.raises(ValueError):
        ds.get_frames_mean(3, 5)


def test_rigaku_3m_reads_each_module_in_chunks_by_threads(make_rigaku_3m, monkeypatch):
    monkeypatch.setattr(rigaku, "_CHUNK_EVENTS", 7)
    events = _random_events()
    # three module files with the offset indices of a real dataset
    offset = [(frame, row + 1024, col, count) for frame, row, col, count in events]
    events_by_suffix = {
        0: events,
        1: offset[: len(offset) // 2],
        2: [(0, 0, 0, 1)],
        3: offset,
        4: events[len(events) // 3 :],
        5: [(59, 3, 5, 2)],
    }
    ds = Rigaku3MDataset(make_rigaku_3m(events_by_suffix))
    plain = {
        0: events,
        1: events[: len(events) // 2],
        2: [(0, 0, 0, 1)],
        3: events,
        4: events[len(events) // 3 :],
        5: [(59, 3, 5, 2)],
    }
    for begin_idx, end_idx in [(0, 60), (11, 47)]:
        expected = ds._stitch(
            [_expected_mean(plain[n], begin_idx, end_idx) for n in ds._MODULE_ORDER]
        )
        for num_workers in (None, 1, 3):
            img = ds.get_scattering(
                num_frames=end_idx - begin_idx,
                begin_idx=begin_idx,
                num_processes=num_workers,
            )
            assert np.array_equal(img, expected)
        assert np.array_equal(ds.get_frames_mean(begin_idx, end_idx), expected)
