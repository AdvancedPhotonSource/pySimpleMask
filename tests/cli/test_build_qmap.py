# Copyright © UChicago Argonne LLC
# See LICENSE file for details
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
    assert args.dq_num == 36
    assert args.sq_num == 360
    assert args.dp_num == 1
    assert args.sp_num == 1
    assert args.phi_offset == 0.0
    assert args.symmetry_fold == 1
    assert args.style == "linear"
    assert args.max_radius == 384
    assert args.beamstop_diameter == 30
    assert args.no_find_center is False
    assert args.output_qmap == "qmap.hdf"
    assert args.output_mask == "mask.tif"
    assert args.report is None
    assert args.blemish is None
    assert args.threshold_high is None
    assert args.use_groupindex_for_dq is False


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
        "--use-groupindex-for-dq",
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
    assert args.use_groupindex_for_dq is True


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


def _probe_q_range(raw_hdf):
    """Load the dataset headlessly just to read its q-range, for building
    --param-constraint tokens that actually split the detector into groups."""
    from pysimplemask.core import SimpleMaskModel

    probe = SimpleMaskModel()
    assert probe.read_data(raw_hdf, beamline="APS_8IDI", num_frames=0) is True
    q = probe.qmap["q"]
    return float(q[probe.mask].min()), float(q[probe.mask].max())


def test_pipeline_use_groupindex_for_dq(raw_hdf, tmp_path, monkeypatch):
    """--use-groupindex-for-dq drives the dynamic q partition from
    --param-constraint groups instead of a linear rebin."""
    monkeypatch.chdir(tmp_path)
    from pysimplemask.cli import _run_build_qmap, _build_qmap_args

    q_min, q_max = _probe_q_range(raw_hdf)
    q_mid = (q_min + q_max) / 2.0

    args = _build_qmap_args([
        raw_hdf,
        "--num-frames", "0",
        "--no-find-center",
        "--param-constraint", f"q:AND:{q_min}:{q_mid}",
        "--param-constraint", f"q:OR:{q_mid}:{q_max}",
        "--use-groupindex-for-dq",
        "--sq-num", "8",
        "--dp-num", "1", "--sp-num", "1",
        "--output-qmap", "out.hdf",
        "--output-mask", "",
    ])
    _run_build_qmap(args)

    with h5py.File("out.hdf") as h:
        assert h["/qmap/dynamic_num_pts"][0] == 2
        assert h["/qmap/static_num_pts"][0] == 8


def test_pipeline_use_groupindex_with_empty_group_raises(raw_hdf, tmp_path, monkeypatch):
    """One empty constraint group among several must also fail loudly, not just
    the "no constraints at all" case."""
    monkeypatch.chdir(tmp_path)
    from pysimplemask.cli import _run_build_qmap, _build_qmap_args

    q_min, q_max = _probe_q_range(raw_hdf)
    q_mid = (q_min + q_max) / 2.0

    args = _build_qmap_args([
        raw_hdf,
        "--num-frames", "0",
        "--no-find-center",
        "--param-constraint", f"q:AND:{q_min}:{q_mid}",
        "--param-constraint", f"q:OR:{q_max + 100}:{q_max + 200}",  # empty
        "--param-constraint", f"q:OR:{q_mid}:{q_max}",
        "--use-groupindex-for-dq",
        "--sq-num", "9",
        "--output-qmap", "out.hdf",
        "--output-mask", "",
    ])
    with pytest.raises(RuntimeError):
        _run_build_qmap(args)
    assert not Path("out.hdf").exists()


def test_pipeline_use_groupindex_with_overlapping_constraints_raises(
    raw_hdf, tmp_path, monkeypatch
):
    """Overlapping --param-constraint ranges must fail loudly, naming the rows."""
    monkeypatch.chdir(tmp_path)
    from pysimplemask.cli import _run_build_qmap, _build_qmap_args

    q_min, q_max = _probe_q_range(raw_hdf)
    span = q_max - q_min

    args = _build_qmap_args([
        raw_hdf,
        "--num-frames", "0",
        "--no-find-center",
        "--param-constraint", f"q:AND:{q_min}:{q_min + 0.6 * span}",
        "--param-constraint", f"q:OR:{q_min + 0.4 * span}:{q_max}",  # overlaps row 1
        "--use-groupindex-for-dq",
        "--sq-num", "8",
        "--output-qmap", "out.hdf",
        "--output-mask", "",
    ])
    with pytest.raises(RuntimeError, match="overlap"):
        _run_build_qmap(args)
    assert not Path("out.hdf").exists()


def test_pipeline_use_groupindex_without_constraints_raises(raw_hdf, tmp_path, monkeypatch):
    """--use-groupindex-for-dq with no parametrization groups must fail loudly,
    not silently fall back to a linear rebin or write an empty qmap file."""
    monkeypatch.chdir(tmp_path)
    from pysimplemask.cli import _run_build_qmap, _build_qmap_args

    args = _build_qmap_args([
        raw_hdf,
        "--num-frames", "0",
        "--no-find-center",
        "--use-groupindex-for-dq",
        "--output-qmap", "out.hdf",
        "--output-mask", "",
    ])
    with pytest.raises(RuntimeError):
        _run_build_qmap(args)
    assert not Path("out.hdf").exists()
