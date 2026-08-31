# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from pysimplemask.core import SimpleMaskModel
from pysimplemask.core.partition import check_consistency


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


def test_update_parameters_recomputes_stale_partition(tmp_path, make_hdf):
    """Recomputing the qmap must also refresh an already-computed partition.

    Otherwise dset.data_display's dqmap/sqmap_partition channels keep showing a
    partition derived from the discarded geometry, with nothing telling the
    caller it is now stale.
    """
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True

    partition_kwargs = dict(mode="q-phi", dq_num=2, sq_num=4, dp_num=4, sp_num=8)
    m.compute_partition(**partition_kwargs)

    new_center = {"beam_center_x": m.dset.metadata["beam_center_x"] + 4.0}
    with patch.object(m, "compute_partition", wraps=m.compute_partition) as spy:
        m.update_parameters(new_metadata=new_center)

    spy.assert_called_once_with(**partition_kwargs)


def test_update_parameters_does_not_compute_partition_when_none_exists(tmp_path, make_hdf):
    """Before any partition has been computed, recomputing the qmap must not
    trigger a partition computation out of nowhere."""
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True
    assert m.new_partition is None

    with patch.object(m, "compute_partition", wraps=m.compute_partition) as spy:
        m.update_parameters(new_metadata={"beam_center_x": 500.0})

    spy.assert_not_called()


def test_compute_partition_with_groupindex_for_dq(tmp_path, make_hdf):
    """Constraint groups from the parametrization mask can drive dq directly,
    instead of a linear rebin of the q-range."""
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True
    assert m.get_parameter_group_count() == 0

    q = m.qmap["q"]
    q_min, q_max = q[m.mask].min(), q[m.mask].max()
    q_mid = (q_min + q_max) / 2.0
    num_groups = 2
    constraints = [
        ("q", "AND", m.qmap_unit["q"], q_min, q_mid),
        ("q", "OR", m.qmap_unit["q"], q_mid, q_max),
    ]
    m.mask_evaluate("mask_parameter", constraints=constraints)
    m.mask_apply("mask_parameter")
    assert m.get_parameter_group_count() == num_groups

    sq_num = 8
    partition = m.compute_partition(
        mode="q-phi", use_groupindex_for_dq=True, sq_num=sq_num, dp_num=2, sp_num=4
    )

    assert partition is not None
    assert partition["dynamic_num_pts"][0] == num_groups
    assert partition["static_num_pts"][0] == num_groups * (sq_num // num_groups)
    assert check_consistency(
        partition["dynamic_roi_map"], partition["static_roi_map"], m.mask
    )


def test_compute_partition_with_groupindex_for_dq_without_groups_raises(
    tmp_path, make_hdf
):
    """use_groupindex_for_dq without an evaluated parametrization mask must fail
    loudly rather than silently falling back to a linear rebin."""
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True

    with pytest.raises(RuntimeError):
        m.compute_partition(
            mode="q-phi", use_groupindex_for_dq=True, sq_num=8, dp_num=2, sp_num=4
        )
    assert m.new_partition is None


def test_compute_partition_with_groupindex_for_dq_raises_on_empty_group(
    tmp_path, make_hdf
):
    """An empty constraint group (its own range matches 0 pixels) must fail loudly.

    combine_partitions compacts away empty bins in dynamic_roi_map/static_roi_map
    but leaves v_list_dim0 uncompacted, so silently proceeding would save a qmap
    file where v_list_dim0 no longer lines up with the roi_map's group numbers.
    """
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True

    q = m.qmap["q"]
    q_min, q_max = q[m.mask].min(), q[m.mask].max()
    q_mid = (q_min + q_max) / 2.0
    unit = m.qmap_unit["q"]
    constraints = [
        ("q", "AND", unit, q_min, q_mid),
        ("q", "OR", unit, q_max + 100, q_max + 200),  # matches nothing: empty group
        ("q", "OR", unit, q_mid, q_max),
    ]
    m.mask_evaluate("mask_parameter", constraints=constraints)
    m.mask_apply("mask_parameter")
    assert m.get_parameter_group_count() == 3

    with pytest.raises(RuntimeError):
        m.compute_partition(
            mode="q-phi", use_groupindex_for_dq=True, sq_num=9, dp_num=2, sp_num=4
        )
    assert m.new_partition is None


def test_compute_partition_with_groupindex_for_dq_raises_on_overlap(tmp_path, make_hdf):
    """Overlapping constraint ranges violate the non-overlapping-groups assumption
    and must fail loudly, with the offending rows named, rather than silently
    resolving via last-write-wins."""
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True

    q = m.qmap["q"]
    q_min, q_max = q[m.mask].min(), q[m.mask].max()
    q_span = q_max - q_min
    unit = m.qmap_unit["q"]
    # row0: [q_min, q_min + 0.6*span]   row1: [q_min + 0.4*span, q_max]  -> overlap
    constraints = [
        ("q", "AND", unit, q_min, q_min + 0.6 * q_span),
        ("q", "OR", unit, q_min + 0.4 * q_span, q_max),
    ]
    m.mask_evaluate("mask_parameter", constraints=constraints)
    m.mask_apply("mask_parameter")
    assert m.get_parameter_group_count() == 2

    with pytest.raises(RuntimeError, match="overlap"):
        m.compute_partition(
            mode="q-phi", use_groupindex_for_dq=True, sq_num=8, dp_num=2, sp_num=4
        )
    assert m.new_partition is None


def test_importing_core_does_not_import_qt():
    code = "import pysimplemask, sys; print('PySide6' in sys.modules)"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "False"
