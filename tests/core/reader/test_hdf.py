"""Tests for the HDF5 scattering loader and frame-averaging helper.

Covers :class:`pysimplemask.core.reader.formats.hdf.HdfDataset` and
:func:`pysimplemask.core.reader.io_utils.average_frames_parallel`.
"""

import h5py
import numpy as np
import pytest

from pysimplemask.core.reader.formats.hdf import HdfDataset
from pysimplemask.core.reader.io_utils import average_frames_parallel


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
# average_frames_parallel
# ---------------------------------------------------------------------------


def test_average_frames_parallel_small_path(make_hdf):
    """Fewer than chunk_size (32) frames take the single-process branch."""
    frames = np.arange(10 * 3 * 4).reshape(10, 3, 4)
    path = make_hdf(frames)
    result = average_frames_parallel(path, num_frames=0)
    assert result.dtype == np.float32
    assert np.allclose(result, frames.mean(axis=0))


def test_average_frames_parallel_large_path(make_hdf):
    """More than chunk_size (32) frames exercise the multiprocessing branch."""
    frames = np.arange(40 * 3 * 3).reshape(40, 3, 3)
    path = make_hdf(frames)
    result = average_frames_parallel(path, num_frames=0)
    assert result.dtype == np.float32
    assert np.allclose(result, frames.mean(axis=0))


def test_average_frames_parallel_2d_returns_array(make_hdf):
    """A 2-D dataset is returned as-is (cast to float32)."""
    arr = np.arange(4 * 5).reshape(4, 5)
    path = make_hdf(arr)
    result = average_frames_parallel(path)
    assert result.dtype == np.float32
    assert result.shape == (4, 5)
    assert np.allclose(result, arr.astype(np.float32))


# ---------------------------------------------------------------------------
# error handling
# ---------------------------------------------------------------------------


def test_missing_data_path_raises_keyerror(make_hdf):
    """Constructing HdfDataset with an absent data_path raises KeyError."""
    frames = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    path = make_hdf(frames)
    with pytest.raises(KeyError):
        HdfDataset(path, data_path="/nope")
