# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Tests for the qmap summary report generator."""

import h5py
import numpy as np
import pytest

from pysimplemask.core import SimpleMaskModel
from pysimplemask.core.report import (
    generate_report,
    generate_report_from_qmap,
    report_from_qmap,
)


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


def test_generate_report_creates_png(loaded_model, tmp_path):
    out = tmp_path / "report.png"
    generate_report(loaded_model, str(out))
    assert out.exists()
    assert out.stat().st_size > 1024
    with open(out, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


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


def test_generate_report_from_qmap_pdf(loaded_model, tmp_path):
    qmap_h5 = tmp_path / "my_qmap.h5"
    loaded_model.save_partition(str(qmap_h5))
    out = tmp_path / "report.pdf"

    result = generate_report_from_qmap(str(qmap_h5), str(out))
    assert str(out) == result
    assert out.exists()
    assert out.stat().st_size > 1024


def test_generate_report_from_qmap_png(loaded_model, tmp_path):
    qmap_h5 = tmp_path / "my_qmap.h5"
    loaded_model.save_partition(str(qmap_h5))
    out = tmp_path / "report.png"

    result = generate_report_from_qmap(str(qmap_h5), str(out))
    assert str(out) == result
    assert out.exists()
    assert out.stat().st_size > 1024
    with open(out, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_generate_report_from_qmap_default_output(loaded_model, tmp_path):
    qmap_h5 = tmp_path / "partition.h5"
    loaded_model.save_partition(str(qmap_h5))

    result = generate_report_from_qmap(str(qmap_h5))
    expected = tmp_path / "partition.pdf"
    assert result == str(expected)
    assert expected.exists()
    assert expected.stat().st_size > 1024


def test_generate_report_from_qmap_explicit_raw_data(loaded_model, tmp_path):
    qmap_h5 = tmp_path / "qmap.hdf"
    loaded_model.save_partition(str(qmap_h5))
    out = tmp_path / "report.png"

    raw_path = str(tmp_path / "scan.h5")
    generate_report_from_qmap(str(qmap_h5), str(out), raw_data=raw_path)
    assert out.exists()
    assert out.stat().st_size > 1024


def test_generate_report_from_qmap_missing_source_file(tmp_path):
    """When source_file referenced in qmap does not exist on disk, report generates gracefully."""
    qmap_h5 = tmp_path / "isolated_qmap.h5"
    with h5py.File(qmap_h5, "w") as f:
        grp = f.create_group("/qmap")
        grp.create_dataset("mask", data=np.ones((32, 32), dtype=np.uint8))
        grp.create_dataset("static_roi_map", data=np.ones((32, 32), dtype=np.uint32))
        grp.create_dataset("dynamic_roi_map", data=np.ones((32, 32), dtype=np.uint32))
        grp.create_dataset("static_num_pts", data=np.array([4, 1]))
        grp.create_dataset("dynamic_num_pts", data=np.array([2, 1]))
        grp.create_dataset("map_names", data=["q", "phi"])
        grp.create_dataset("source_file", data="/nonexistent/path/data.h5")

    out = tmp_path / "isolated_report.png"
    generate_report_from_qmap(str(qmap_h5), str(out))
    assert out.exists()
    assert out.stat().st_size > 1024


def test_generate_report_from_qmap_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        generate_report_from_qmap(str(tmp_path / "does_not_exist.h5"))


def test_generate_report_from_qmap_invalid_hdf(tmp_path):
    txt_file = tmp_path / "not_hdf.txt"
    txt_file.write_text("just some text")
    with pytest.raises(ValueError, match="not a valid HDF5"):
        generate_report_from_qmap(str(txt_file))


def test_report_from_qmap_alias():
    assert report_from_qmap is generate_report_from_qmap


def test_generate_report_landscape_default(loaded_model, tmp_path):
    import matplotlib.pyplot as plt
    out = tmp_path / "report_landscape.png"
    generate_report(loaded_model, str(out))
    img = plt.imread(out)
    h, w = img.shape[:2]
    assert w > h  # landscape width > height


def test_generate_report_portrait_option(loaded_model, tmp_path):
    import matplotlib.pyplot as plt
    out = tmp_path / "report_portrait.png"
    generate_report(loaded_model, str(out), orientation="portrait")
    img = plt.imread(out)
    h, w = img.shape[:2]
    assert h > w  # portrait height > width


def test_generate_report_from_qmap_orientation_options(loaded_model, tmp_path):
    import matplotlib.pyplot as plt
    qmap_h5 = tmp_path / "qmap_orient.h5"
    loaded_model.save_partition(str(qmap_h5))

    out_l = tmp_path / "from_qmap_landscape.png"
    generate_report_from_qmap(str(qmap_h5), str(out_l), orientation="landscape")
    img_l = plt.imread(out_l)
    assert img_l.shape[1] > img_l.shape[0]

    out_p = tmp_path / "from_qmap_portrait.png"
    generate_report_from_qmap(str(qmap_h5), str(out_p), orientation="portrait")
    img_p = plt.imread(out_p)
    assert img_p.shape[0] > img_p.shape[1]


def test_generate_report_with_beam_center_crosshair(loaded_model, tmp_path):
    """Verify that beam center crop with intertwined crosshairs renders without error."""
    # Set center inside frame (H=64, W=60)
    loaded_model.dset.metadata["beam_center_y"] = 32
    loaded_model.dset.metadata["beam_center_x"] = 30
    loaded_model.dset.metadata["detector_distance"] = 1000
    loaded_model.dset.metadata["pixel_size"] = 75e-6
    out = tmp_path / "report_with_center.png"
    generate_report(loaded_model, str(out))
    assert out.exists()
    assert out.stat().st_size > 1024

    # Also test through generate_report_from_qmap
    qmap_h5 = tmp_path / "qmap_with_center.h5"
    loaded_model.save_partition(str(qmap_h5))
    out_qmap = tmp_path / "report_qmap_with_center.png"
    generate_report_from_qmap(str(qmap_h5), str(out_qmap))
    assert out_qmap.exists()
    assert out_qmap.stat().st_size > 1024


def test_generate_report_from_qmap_num_frames(loaded_model, tmp_path):
    """Test generating report from qmap with explicit num_frames and begin_idx."""
    qmap_h5 = tmp_path / "qmap_nf.h5"
    loaded_model.save_partition(str(qmap_h5))
    out = tmp_path / "report_nf.pdf"
    generate_report_from_qmap(
        str(qmap_h5),
        str(out),
        begin_idx=1,
        num_frames=2,
    )
    assert out.exists()
    assert out.stat().st_size > 1024



