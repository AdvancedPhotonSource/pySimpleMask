# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Tests for the HDF5 scattering loader and frame-averaging helper.

Covers :class:`pysimplemask.core.reader.formats.hdf.HdfDataset` and
:func:`pysimplemask.core.reader.io_utils.average_frames_parallel`.
"""

import gc
import threading
import types

import h5py
import hdf5plugin
import numpy as np
import pytest

from pysimplemask.core.reader import io_utils
from pysimplemask.core.reader.formats.hdf import HdfDataset
from pysimplemask.core.reader.io_utils import (
    _chunk_table,
    _probe_direct,
    average_frames_parallel,
    resolve_frame_range,
)


# ---------------------------------------------------------------------------
# det_size
# ---------------------------------------------------------------------------


def test_det_size_3d_stack(make_hdf):
    """A 3-D stack (n, R, C) reports det_size == (R, C)."""
    frames = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    path = make_hdf(frames)
    ds = HdfDataset(path)
    assert ds.det_size == (3, 4)


def test_det_size_2d_dataset_via_make_hdf(make_hdf):
    """A 2-D dataset (R, C) reports det_size == (R, C)."""
    frames = np.arange(5 * 6).reshape(5, 6)
    path = make_hdf(frames)
    ds = HdfDataset(path)
    assert ds.det_size == (5, 6)


def test_det_size_2d_dataset_via_h5py(tmp_path):
    """det_size for a 2-D dataset built directly with h5py."""
    path = str(tmp_path / "twod.h5")
    arr = np.arange(7 * 8).reshape(7, 8)
    with h5py.File(path, "w") as h:
        h["/entry/data/data"] = arr
    ds = HdfDataset(path)
    assert ds.det_size == (7, 8)


# ---------------------------------------------------------------------------
# get_scattering
# ---------------------------------------------------------------------------


def test_get_scattering_default_is_full_mean(make_hdf):
    """Default get_scattering() returns the per-pixel mean over all frames."""
    frames = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    path = make_hdf(frames)
    ds = HdfDataset(path)
    result = ds.get_scattering()
    assert result.shape == (3, 4)
    assert np.allclose(result, frames.mean(axis=0))


def test_get_scattering_frame_subset(make_hdf):
    """get_scattering(num_frames=2, begin_idx=1) averages frames[1:3]."""
    frames = np.arange(5 * 3 * 4).reshape(5, 3, 4)
    path = make_hdf(frames)
    ds = HdfDataset(path)
    result = ds.get_scattering(num_frames=2, begin_idx=1)
    assert np.allclose(result, frames[1:3].mean(axis=0))


# ---------------------------------------------------------------------------
# get_frames_mean: browsing the frames a few at a time
# ---------------------------------------------------------------------------


def test_frames_of_a_stack_can_be_browsed_one_at_a_time(make_hdf):
    ds = HdfDataset(make_hdf(np.zeros((5, 3, 4))))
    assert ds.num_browsable_frames == 5
    assert ds.default_frame_average == 1


def test_get_frames_mean(make_hdf):
    frames = np.arange(5 * 3 * 4, dtype=np.int32).reshape(5, 3, 4)
    ds = HdfDataset(make_hdf(frames))
    one = ds.get_frames_mean(2, 3)
    assert one.shape == (3, 4)
    assert one.dtype == np.float32
    assert np.array_equal(one, frames[2])
    assert np.array_equal(ds.get_frames_mean(1, 4), frames[1:4].mean(axis=0))
    assert np.array_equal(ds.get_frames_mean(0, 5), ds.get_scattering(num_frames=0))


def test_get_frames_mean_reads_unsigned_counts_as_signed(make_hdf):
    """As get_scattering: the overflow marker of an unsigned detector (65535
    for uint16) shows as -1, not as a huge count."""
    frames = np.ones((3, 2, 2), dtype=np.uint16)
    frames[:, 0, 0] = 65535
    ds = HdfDataset(make_hdf(frames))
    assert np.array_equal(ds.get_frames_mean(0, 3), [[-1, 1], [1, 1]])
    assert np.array_equal(ds.get_frames_mean(0, 3), ds.get_scattering(num_frames=0))


@pytest.mark.parametrize("begin_idx, end_idx", [(-1, 2), (2, 2), (3, 1), (4, 6), (5, 6)])
def test_get_frames_mean_rejects_range_outside_frames(make_hdf, begin_idx, end_idx):
    ds = HdfDataset(make_hdf(np.zeros((5, 3, 4))))
    with pytest.raises(ValueError):
        ds.get_frames_mean(begin_idx, end_idx)


def test_single_image_has_no_frames_to_browse(make_hdf):
    ds = HdfDataset(make_hdf(np.zeros((3, 4))))
    assert ds.num_browsable_frames == 0
    with pytest.raises(ValueError):
        ds.get_frames_mean(0, 1)


# ---------------------------------------------------------------------------
# resolve_frame_range
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "total_frames, start_frame, num_frames, expected",
    [
        (5000, 0, 0, 5000),  # all frames
        (5000, 100, None, 4900),  # ... that remain
        (5000, 0, 300, 300),  # an exact count
        (5000, 4900, 300, 100),  # ... clamped to the end
        (5000, 0, 1, 1),
        (20000, 0, 0.5, 10000),  # a fraction of the frames
        (20000, 0, 0.2, 4000),
        (20000, 15000, 0.5, 5000),  # ... of the whole dataset, then clamped
        (1500, 0, 0.5, 1000),  # ... but at least 1000 of them
        (600, 0, 0.5, 600),  # ... or all there are
        (20000, 0, -1, 4000),  # the default subset is a fifth
        (3000, 0, -1, 1000),
        (3000, 2500, -1, 500),
    ],
)
def test_resolve_frame_range(total_frames, start_frame, num_frames, expected):
    assert resolve_frame_range(total_frames, start_frame, num_frames) == expected


def test_resolve_frame_range_negative_is_a_fifth():
    """``num_frames < 0`` keeps meaning ``max(1000, total // 5)`` frames, and
    is what the fraction 0.2 gives."""
    for total_frames in range(1, 60000, 7):
        expected = min(total_frames, max(1000, total_frames // 5))
        assert resolve_frame_range(total_frames, 0, -1) == expected
        assert resolve_frame_range(total_frames, 0, 0.2) == expected


@pytest.mark.parametrize("start_frame", [-1, 10, 11])
def test_resolve_frame_range_rejects_start_outside_dataset(start_frame):
    with pytest.raises(ValueError):
        resolve_frame_range(10, start_frame, 0)


# ---------------------------------------------------------------------------
# average_frames_parallel
# ---------------------------------------------------------------------------


def _is_direct(path, data_path="/entry/data/data"):
    """Whether the dataset qualifies for the direct (HDF5-bypassing) chunk read."""
    with h5py.File(path, "r") as f:
        dataset = f[data_path]
        return _probe_direct(dataset, 0, dataset.shape[0]) is not None


def test_average_frames_parallel_small_path(make_hdf):
    """Fewer than chunk_size (32) frames of a layout that has to be read
    through h5py take the single-process branch."""
    frames = np.arange(10 * 3 * 4).reshape(10, 3, 4)
    path = make_hdf(frames, compression="gzip")
    assert not _is_direct(path)
    result = average_frames_parallel(path, num_frames=0)
    assert result.dtype == np.float32
    assert np.allclose(result, frames.mean(axis=0))


def test_average_frames_parallel_large_path(make_hdf):
    """More than chunk_size (32) frames of a layout that has to be read
    through h5py exercise the multiprocessing branch."""
    frames = np.arange(40 * 3 * 3).reshape(40, 3, 3)
    path = make_hdf(frames, compression="gzip")
    assert not _is_direct(path)
    result = average_frames_parallel(path, num_frames=0)
    assert result.dtype == np.float32
    assert np.allclose(result, frames.mean(axis=0))


def test_hdf_dataset_2d_returns_image(make_hdf):
    """A 2-D dataset is returned as-is by HdfDataset (cast to float32)."""
    arr = np.arange(4 * 5).reshape(4, 5)
    path = make_hdf(arr)
    ds = HdfDataset(path)
    assert ds.ndim == 2
    result = ds.get_scattering()
    assert result.dtype == np.float32
    assert result.shape == (4, 5)
    assert np.allclose(result, arr.astype(np.float32))


def test_average_frames_parallel_rejects_2d(make_hdf):
    """average_frames_parallel is 3-D only; 2-D input raises."""
    path = make_hdf(np.arange(4 * 5).reshape(4, 5))
    with pytest.raises(ValueError):
        average_frames_parallel(path)


# ---------------------------------------------------------------------------
# average_frames_parallel: direct chunk read
# ---------------------------------------------------------------------------

_FRAME_SHAPE = (16, 20)

# create_dataset kwargs of the layouts the direct read handles
_DIRECT_LAYOUTS = {
    "contiguous": {},
    "chunked": {"chunks": (1, *_FRAME_SHAPE)},
    "bitshuffle-lz4": {
        "chunks": (1, *_FRAME_SHAPE),
        "compression": hdf5plugin.Bitshuffle(cname="lz4"),
    },
    "bitshuffle-zstd": {
        "chunks": (1, *_FRAME_SHAPE),
        "compression": hdf5plugin.Bitshuffle(cname="zstd"),
    },
    "lz4": {"chunks": (1, *_FRAME_SHAPE), "compression": hdf5plugin.LZ4()},
    # several LZ4 blocks per frame
    "lz4-blocks": {"chunks": (1, *_FRAME_SHAPE), "compression": hdf5plugin.LZ4(nbytes=96)},
    "blosc": {"chunks": (1, *_FRAME_SHAPE), "compression": hdf5plugin.Blosc()},
    "blosc-zstd-bitshuffle": {
        "chunks": (1, *_FRAME_SHAPE),
        "compression": hdf5plugin.Blosc(
            cname="zstd", shuffle=hdf5plugin.Blosc.BITSHUFFLE
        ),
    },
}


def _random_frames(dtype, num_frames=40):
    """Frames that are noise in their top half (incompressible: the LZ4 filter
    stores such blocks as is) and sparse counts below, with the dtype's
    extreme values thrown in."""
    rng = np.random.default_rng(0)
    dtype = np.dtype(dtype)
    shape = (num_frames, *_FRAME_SHAPE)
    if dtype.kind == "f":
        return rng.normal(size=shape).astype(dtype)
    info = np.iinfo(dtype)
    frames = rng.integers(0, 3, size=shape).astype(dtype)
    half = _FRAME_SHAPE[0] // 2
    frames[:, :half] = rng.integers(
        info.min, info.max, size=(num_frames, half, _FRAME_SHAPE[1]), endpoint=True
    )
    frames[:, -1, -1] = info.max
    return frames


def _assert_mean(result, frames):
    """``result`` is the mean of ``frames``, unsigned counts reinterpreted as
    signed, to the float32 precision the frames are summed with."""
    if frames.dtype.kind == "u":
        frames = frames.view(np.dtype(f"int{frames.dtype.itemsize * 8}"))
    expected = frames.mean(axis=0, dtype=np.float64)
    atol = 1e-6 * np.abs(frames.astype(np.float64)).max()
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=atol)


@pytest.mark.parametrize("layout", _DIRECT_LAYOUTS)
@pytest.mark.parametrize(
    "dtype", ["uint8", "uint16", "uint32", "int32", "int64", "float32", "float64"]
)
def test_direct_read_mean(make_hdf, layout, dtype):
    """Every layout the direct read accepts averages to the h5py result."""
    frames = _random_frames(dtype)
    path = make_hdf(frames, **_DIRECT_LAYOUTS[layout])
    assert _is_direct(path)
    result = average_frames_parallel(path, num_frames=0, num_processes=3)
    assert result.dtype == np.float32
    assert result.shape == _FRAME_SHAPE
    _assert_mean(result, frames)


def test_direct_read_unsigned_overflow_is_negative(make_hdf):
    """An all-ones unsigned count (a masked pixel) averages to -1, as it does
    through h5py."""
    frames = np.zeros((5, *_FRAME_SHAPE), dtype=np.uint32)
    frames[:, 2, 3] = 2**32 - 1
    path = make_hdf(frames, **_DIRECT_LAYOUTS["bitshuffle-lz4"])
    result = average_frames_parallel(path, num_frames=0)
    assert result[2, 3] == -1
    assert np.count_nonzero(result) == 1


def test_direct_read_frame_subset(make_hdf):
    """Only the requested frames are read."""
    frames = _random_frames("uint16")
    path = make_hdf(frames, **_DIRECT_LAYOUTS["bitshuffle-lz4"])
    result = average_frames_parallel(path, start_frame=7, num_frames=12)
    _assert_mean(result, frames[7:19])
    # fewer frames than workers
    result = average_frames_parallel(path, start_frame=38, num_frames=0)
    _assert_mean(result, frames[38:])


def test_direct_read_long_stack_does_not_overflow(make_hdf):
    """Workers hand over their narrow sums before these can overflow: 600
    int8 frames of 127 would wrap an int16 sum."""
    frames = np.full((600, 4, 5), 127, dtype=np.uint8)
    frames[:, 0, 0] = 128  # -128 once reinterpreted as signed
    path = make_hdf(
        frames, chunks=(1, 4, 5), compression=hdf5plugin.Bitshuffle(cname="lz4")
    )
    assert 600 > 2 * io_utils._SUM_WINDOW
    result = average_frames_parallel(path, num_frames=0, num_processes=2)
    expected = np.full((4, 5), 127, dtype=np.float32)
    expected[0, 0] = -128
    assert np.array_equal(result, expected)


def test_direct_read_keeps_garbage_collection_in_the_calling_thread(make_hdf):
    """A collection that a worker thread trips would destroy the caller's
    garbage from that thread, which crashes a GUI whose Qt objects are among
    it; none may run until the threads are done."""
    frames = _random_frames("uint16", num_frames=200)
    path = make_hdf(frames, **_DIRECT_LAYOUTS["bitshuffle-lz4"])
    caller = threading.get_ident()
    collecting_threads = set()

    def record(phase, info):
        collecting_threads.add(threading.get_ident())

    threshold = gc.get_threshold()
    gc.callbacks.append(record)
    gc.set_threshold(1, 1, 1)  # collect at every opportunity
    try:
        result = average_frames_parallel(path, num_frames=0, num_processes=4)
    finally:
        gc.set_threshold(*threshold)
        gc.callbacks.remove(record)

    _assert_mean(result, frames)
    assert collecting_threads <= {caller}
    assert gc.isenabled()


def test_direct_read_leaves_garbage_collection_disabled_if_it_was(make_hdf):
    frames = _random_frames("uint16", num_frames=4)
    path = make_hdf(frames, **_DIRECT_LAYOUTS["bitshuffle-lz4"])
    gc.disable()
    try:
        average_frames_parallel(path, num_frames=0)
        assert not gc.isenabled()
    finally:
        gc.enable()


def test_direct_read_chunk_stored_without_filter(make_hdf):
    """A chunk the writer stored unfiltered (filter mask set) is read raw."""
    frames = _random_frames("uint16", num_frames=4)
    path = make_hdf(frames, **_DIRECT_LAYOUTS["bitshuffle-lz4"])
    with h5py.File(path, "r+") as f:
        f["/entry/data/data"].id.write_direct_chunk(
            (1, 0, 0), frames[1].tobytes(), filter_mask=1
        )
    assert _is_direct(path)
    result = average_frames_parallel(path, num_frames=0)
    _assert_mean(result, frames)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chunks": (1, *_FRAME_SHAPE), "compression": "gzip"},
        {"chunks": (1, *_FRAME_SHAPE), "shuffle": True, "compression": hdf5plugin.LZ4()},
        {"chunks": (4, *_FRAME_SHAPE), "compression": hdf5plugin.Bitshuffle(cname="lz4")},
        {"dtype": ">u2"},
    ],
    ids=["gzip", "two-filters", "multi-frame-chunks", "big-endian"],
)
def test_other_layouts_are_read_through_h5py(make_hdf, kwargs):
    """Layouts the direct read does not understand still average correctly."""
    frames = _random_frames("uint16")
    path = make_hdf(frames, **kwargs)
    assert not _is_direct(path)
    result = average_frames_parallel(path, num_frames=0)
    _assert_mean(result, frames)


def test_missing_chunk_is_read_through_h5py(tmp_path):
    """A frame that was never written has no chunk to read directly; h5py
    fills it with zeros."""
    frames = _random_frames("uint16", num_frames=5)
    path = str(tmp_path / "gap.h5")
    with h5py.File(path, "w") as f:
        dataset = f.create_dataset(
            "/entry/data/data",
            shape=frames.shape,
            dtype=frames.dtype,
            **_DIRECT_LAYOUTS["bitshuffle-lz4"],
        )
        for n in (0, 1, 3, 4):
            dataset[n] = frames[n]
    frames[2] = 0
    assert not _is_direct(path)
    result = average_frames_parallel(path, num_frames=0)
    _assert_mean(result, frames)
    # a range that skips the gap does qualify
    with h5py.File(path, "r") as f:
        assert _probe_direct(f["/entry/data/data"], 3, 2) is not None
    result = average_frames_parallel(path, start_frame=3, num_frames=2)
    _assert_mean(result, frames[3:])


def test_failed_direct_read_falls_back_to_h5py(make_hdf, monkeypatch, caplog):
    """If the direct read fails, the frames are read through h5py instead."""
    frames = _random_frames("uint16")
    path = make_hdf(frames, **_DIRECT_LAYOUTS["bitshuffle-lz4"])

    def broken(layout, num_workers):
        raise RuntimeError("bitshuffle decompression failed")

    monkeypatch.setattr(io_utils, "_average_direct", broken)
    with caplog.at_level("WARNING"):
        result = average_frames_parallel(path, num_frames=0)
    assert "direct chunk read failed" in caplog.text
    _assert_mean(result, frames)


def test_chunk_table_without_chunk_iter(tmp_path):
    """The per-chunk lookup used with HDF5 < 1.12.3 (no ``chunk_iter``)
    finds the same chunks."""
    frames = _random_frames("uint16", num_frames=6)
    path = str(tmp_path / "data.h5")
    with h5py.File(path, "w") as f:
        dataset = f.create_dataset(
            "/entry/data/data",
            shape=frames.shape,
            dtype=frames.dtype,
            **_DIRECT_LAYOUTS["bitshuffle-lz4"],
        )
        dataset[:5] = frames[:5]  # the last frame is never written

    with h5py.File(path, "r") as f:
        dataset = f["/entry/data/data"]
        lookup = dataset.id.get_chunk_info_by_coord
        old_hdf5 = types.SimpleNamespace(
            id=types.SimpleNamespace(get_chunk_info_by_coord=lookup)
        )
        for expected, table in zip(_chunk_table(dataset, 1, 3), _chunk_table(old_hdf5, 1, 3)):
            assert np.array_equal(expected, table)
        assert _chunk_table(dataset, 3, 3) is None
        assert _chunk_table(old_hdf5, 3, 3) is None


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_missing_data_path_raises_keyerror(make_hdf):
    """Constructing HdfDataset with an absent data_path raises KeyError."""
    frames = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    path = make_hdf(frames)
    with pytest.raises(KeyError):
        HdfDataset(path, data_path="/nope")
