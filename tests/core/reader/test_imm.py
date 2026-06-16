"""Tests for the IMM (dense and compression-6 sparse) scattering loader."""

import numpy as np
import pytest

from pysimplemask.core.reader.formats.imm import ImmDataset

# A small two-frame 2x3 stack used by the basic dense/sparse cases.
_BASIC_FRAMES = np.array(
    [[[0, 1, 2], [3, 0, 5]], [[1, 0, 0], [0, 2, 0]]], dtype=np.uint16
)


def test_dense_basic(make_imm):
    path = make_imm(_BASIC_FRAMES, sparse=False)
    ds = ImmDataset(path)
    assert ds.det_size == (2, 3)
    assert ds.is_sparse is False
    assert np.allclose(ds.get_scattering(), _BASIC_FRAMES.mean(axis=0))


def test_sparse_basic(make_imm):
    path = make_imm(_BASIC_FRAMES, sparse=True)
    ds = ImmDataset(path)
    assert ds.is_sparse is True
    assert ds.det_size == (2, 3)
    # Mean is over ALL pixels (including zeros): sparse stores only nonzeros
    # but the accumulation is still divided by the frame count.
    assert np.allclose(ds.get_scattering(), _BASIC_FRAMES.mean(axis=0))


@pytest.mark.parametrize("sparse", [False, True])
def test_frame_subsetting(make_imm, sparse):
    frames = np.arange(4 * 2 * 3).reshape(4, 2, 3).astype(np.uint16)
    path = make_imm(frames, sparse=sparse)
    ds = ImmDataset(path)
    result = ds.get_scattering(num_frames=2, begin_idx=1)
    assert np.allclose(result, frames[1:3].mean(axis=0))


@pytest.mark.parametrize("sparse", [False, True])
def test_random_stack(make_imm, sparse):
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 10, size=(5, 4, 4)).astype(np.uint16)
    path = make_imm(frames, sparse=sparse)
    ds = ImmDataset(path)
    assert ds.det_size == (4, 4)
    assert np.allclose(ds.get_scattering(), frames.mean(axis=0))


@pytest.mark.parametrize("sparse", [False, True])
def test_num_frames_clamped(make_imm, sparse):
    path = make_imm(_BASIC_FRAMES, sparse=sparse)
    ds = ImmDataset(path)
    # Requesting more frames than exist clamps to the 2 available frames and
    # divides by 2 (not by the requested 10).
    result = ds.get_scattering(num_frames=10, begin_idx=0)
    assert np.allclose(result, _BASIC_FRAMES.mean(axis=0))
