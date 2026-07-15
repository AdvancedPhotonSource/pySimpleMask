# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import subprocess
import sys

import numpy as np

from pysimplemask.core import SimpleMaskModel


def _frames():
    # 4 frames of 16x12 with a hot strip so thresholding has an effect
    frames = np.zeros((4, 16, 12), dtype=np.uint16)
    frames[:, 8, :] = 100
    return frames


def test_model_load_threshold_partition_save(tmp_path, make_hdf):
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True
    assert m.shape == (16, 12)

    # threshold mask then a polygon mask, fully headless
    m.mask_evaluate("mask_threshold", low=0, high=50,
                    low_enable=False, high_enable=True)
    m.mask_apply("mask_threshold")
    m.add_polygon([(0, 0), (0, 4), (4, 4), (4, 0)], mode="exclusive")
    m.evaluate_draw()
    m.mask_apply("mask_draw")
    assert m.mask.shape == (16, 12)
    assert not m.mask[1, 1]  # inside the excluded polygon

    out_mask = tmp_path / "mask.tif"
    m.save_mask(str(out_mask))
    assert out_mask.exists()

    m.compute_partition(mode="q-phi", dq_num=2, sq_num=4, dp_num=4, sp_num=8)
    out_qmap = tmp_path / "qmap.hdf"
    m.save_partition(str(out_qmap))
    assert out_qmap.exists()


def test_importing_core_does_not_import_qt():
    code = "import pysimplemask, sys; print('PySide6' in sys.modules)"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "False"
