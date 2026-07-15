# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Tests for the qmap summary report generator."""

import h5py
import numpy as np
import pytest

from pysimplemask.core import SimpleMaskModel
from pysimplemask.core.report import generate_report


@pytest.fixture
def loaded_model(tmp_path):
    """A SimpleMaskModel with data loaded and a partition computed."""
    p = tmp_path / "scan.h5"
    rng = np.random.default_rng(42)
    frames = rng.integers(1, 100, size=(3, 64, 60)).astype(np.uint16)
    with h5py.File(p, "w") as h:
        h["/entry/data/data"] = frames
    m = SimpleMaskModel()
    m.read_data(str(p), beamline="APS_8IDI", num_frames=0)
    m.compute_partition(mode="q-phi", dq_num=2, sq_num=4, dp_num=1, sp_num=1)
    return m


def test_generate_report_creates_pdf(loaded_model, tmp_path):
    out = tmp_path / "report.pdf"
    generate_report(loaded_model, str(out))
    assert out.exists()
    assert out.stat().st_size > 1024  # a real PDF is bigger than 1 KB


def test_generate_report_overwrite(loaded_model, tmp_path):
    out = tmp_path / "report.pdf"
    generate_report(loaded_model, str(out))
    generate_report(loaded_model, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_report_with_no_partition(tmp_path):
    """Report should not crash when no partition has been computed yet."""
    p = tmp_path / "scan.h5"
    rng = np.random.default_rng(0)
    with h5py.File(p, "w") as h:
        h["/entry/data/data"] = rng.integers(1, 50, size=(3, 32, 30)).astype(np.uint16)
    m = SimpleMaskModel()
    m.read_data(str(p), beamline="APS_8IDI", num_frames=0)
    out = tmp_path / "report.pdf"
    generate_report(m, str(out))
    assert out.exists()
