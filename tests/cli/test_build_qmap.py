"""Tests for pysimplemask-build-qmap CLI."""

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture
def raw_hdf(tmp_path):
    """Minimal synthetic HDF5 that APS8IDIReader can load."""
    p = tmp_path / "scan.h5"
    rng = np.random.default_rng(0)
    frames = rng.integers(1, 50, size=(5, 32, 30)).astype(np.uint16)
    with h5py.File(p, "w") as h:
        h["/entry/data/data"] = frames
    return str(p)


# ── argument parsing ──────────────────────────────────────────────────────────

def test_no_args_exits_nonzero():
    """build_qmap with no positional argument exits non-zero (argparse error)."""
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.argv=['prog']; "
         "from pysimplemask.cli import build_qmap; build_qmap()"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_default_args_parsed(raw_hdf):
    """Default values are applied when flags are omitted."""
    from pysimplemask.cli import _build_qmap_args

    args = _build_qmap_args([raw_hdf])
    assert args.beamline == "APS_8IDI"
    assert args.begin_idx == 0
    assert args.num_frames == -1
    assert args.mode == "q-phi"
    assert args.dq_num == 10
    assert args.sq_num == 100
    assert args.dp_num == 36
    assert args.sp_num == 360
    assert args.phi_offset == 0.0
    assert args.symmetry_fold == 1
    assert args.style == "linear"
    assert args.max_radius == 384
    assert args.no_find_center is False
    assert args.output_qmap == "qmap.hdf"
    assert args.output_mask == "mask.tif"
    assert args.blemish is None
    assert args.threshold_high is None


def test_custom_args_parsed(raw_hdf):
    """Custom values override defaults."""
    from pysimplemask.cli import _build_qmap_args

    args = _build_qmap_args([
        raw_hdf,
        "--beamline", "APS_8IDI",
        "--begin-idx", "2",
        "--num-frames", "3",
        "--mode", "x-y",
        "--dq-num", "5",
        "--sq-num", "50",
        "--dp-num", "18",
        "--sp-num", "180",
        "--phi-offset", "10.0",
        "--symmetry-fold", "2",
        "--style", "logarithmic",
        "--max-radius", "256",
        "--no-find-center",
        "--output-qmap", "my.hdf",
        "--output-mask", "my.tif",
        "--threshold-high", "1000",
    ])
    assert args.beamline == "APS_8IDI"
    assert args.begin_idx == 2
    assert args.num_frames == 3
    assert args.mode == "x-y"
    assert args.dq_num == 5
    assert args.sq_num == 50
    assert args.dp_num == 18
    assert args.sp_num == 180
    assert args.phi_offset == 10.0
    assert args.symmetry_fold == 2
    assert args.style == "logarithmic"
    assert args.max_radius == 256
    assert args.no_find_center is True
    assert args.output_qmap == "my.hdf"
    assert args.output_mask == "my.tif"
    assert args.threshold_high == 1000.0


# ── end-to-end pipeline ───────────────────────────────────────────────────────

def test_pipeline_produces_outputs(raw_hdf, tmp_path, monkeypatch):
    """Full pipeline: load → center → partition → save."""
    monkeypatch.chdir(tmp_path)
    from pysimplemask.cli import _run_build_qmap, _build_qmap_args

    args = _build_qmap_args([
        raw_hdf,
        "--num-frames", "0",
        "--no-find-center",
        "--mode", "q-phi",
        "--dq-num", "2",
        "--sq-num", "4",
        "--dp-num", "4",
        "--sp-num", "8",
        "--output-qmap", "out.hdf",
        "--output-mask", "out.tif",
    ])
    _run_build_qmap(args)

    assert Path("out.hdf").exists()
    assert Path("out.tif").exists()
    with h5py.File("out.hdf") as h:
        assert "/qmap/dynamic_roi_map" in h
        assert "/qmap/static_roi_map" in h


def test_pipeline_skip_mask_save(raw_hdf, tmp_path, monkeypatch):
    """Passing output-mask='' skips saving the mask TIFF."""
    monkeypatch.chdir(tmp_path)
    from pysimplemask.cli import _run_build_qmap, _build_qmap_args

    args = _build_qmap_args([
        raw_hdf,
        "--num-frames", "0",
        "--no-find-center",
        "--dq-num", "2", "--sq-num", "4",
        "--dp-num", "4", "--sp-num", "8",
        "--output-qmap", "out.hdf",
        "--output-mask", "",
    ])
    _run_build_qmap(args)

    assert Path("out.hdf").exists()
    assert not Path("mask.tif").exists()
    assert not Path("out.tif").exists()


def test_pipeline_threshold_mask(raw_hdf, tmp_path, monkeypatch):
    """--threshold-high masks high-intensity pixels before partitioning."""
    monkeypatch.chdir(tmp_path)
    from pysimplemask.cli import _run_build_qmap, _build_qmap_args
    import tifffile

    args = _build_qmap_args([
        raw_hdf,
        "--num-frames", "0",
        "--no-find-center",
        "--threshold-high", "1",   # masks almost every pixel
        "--dq-num", "2", "--sq-num", "4",
        "--dp-num", "4", "--sp-num", "8",
        "--output-qmap", "out.hdf",
        "--output-mask", "out.tif",
    ])
    _run_build_qmap(args)

    mask = tifffile.imread("out.tif")
    # With threshold=1, most pixels should be masked (mask==0)
    assert mask.mean() < 0.5
