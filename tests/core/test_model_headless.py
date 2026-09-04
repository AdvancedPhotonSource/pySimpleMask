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


def test_compute_partition_with_groupindex_for_subpartition(tmp_path, make_hdf):
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
    dp_num, sp_num = 2, 4
    partition = m.compute_partition(
        mode="q-phi", use_groupindex_for_subpartition=True, dq_num=1, sq_num=sq_num,
        dp_num=dp_num, sp_num=sp_num,
    )

    assert partition is not None
    assert partition["dynamic_num_pts"][0] == num_groups
    assert partition["static_num_pts"][0] == num_groups * sq_num
    # phi (axis1) is also sub-partitioned per group, not shared globally
    assert partition["dynamic_num_pts"][1] == num_groups * dp_num
    assert partition["static_num_pts"][1] == num_groups * sp_num
    assert check_consistency(
        partition["dynamic_roi_map"], partition["static_roi_map"], m.mask
    )


def test_compute_partition_with_groupindex_for_subpartition_multi_bin_per_group(
    tmp_path, make_hdf
):
    """dq_num/sq_num are per-group sub-partition sizes: each group gets its own
    independent dq_num dynamic bins (not just one), combined across groups."""
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True

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

    dq_num, sq_num = 2, 8
    partition = m.compute_partition(
        mode="q-phi", use_groupindex_for_subpartition=True, dq_num=dq_num,
        sq_num=sq_num, dp_num=2, sp_num=4,
    )

    assert partition is not None
    assert partition["dynamic_num_pts"][0] == num_groups * dq_num
    assert partition["static_num_pts"][0] == num_groups * sq_num
    assert check_consistency(
        partition["dynamic_roi_map"], partition["static_roi_map"], m.mask
    )


def test_compute_partition_with_groupindex_for_subpartition_on_ellipse_mode(
    tmp_path, make_hdf
):
    """use_groupindex_for_subpartition also drives the dynamic equivalent-q axis
    for the ellipse (eq-ephi) mode: compute_partition swaps in the ellipse-corrected
    rho/phi maps before the groupindex machinery runs, so each constraint group
    gets its own dq_num-bin sub-partition exactly as it does for plain q-phi. The
    phi (dp/sp) axis is also sub-partitioned per group, using that same group's
    own ellipse fit (see test_compute_partition_ellipse_fits_geometry_per_group)."""
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True

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
    dp_num, sp_num = 2, 4
    partition = m.compute_partition(
        mode="eq-ephi", use_groupindex_for_subpartition=True, dq_num=1, sq_num=sq_num,
        dp_num=dp_num, sp_num=sp_num,
    )

    assert partition is not None
    assert partition["dynamic_num_pts"][0] == num_groups
    assert partition["static_num_pts"][0] == num_groups * sq_num
    assert partition["dynamic_num_pts"][1] == num_groups * dp_num
    assert partition["static_num_pts"][1] == num_groups * sp_num
    assert check_consistency(
        partition["dynamic_roi_map"], partition["static_roi_map"], m.mask
    )


def test_compute_partition_ellipse_fits_geometry_per_group(tmp_path, make_hdf):
    """use_groupindex_for_subpartition in eq-ephi mode fits an independent
    ellipse to each pixel-group's own footprint, instead of reusing one
    whole-mask fit for every group — both rho (radial) and phi (angular) come
    from that same per-group fit, so the two axes stay geometrically
    consistent within a group. self.qmap is restored to true q/phi afterward
    regardless."""
    from unittest.mock import patch

    from pysimplemask.core import model as model_module

    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True

    q_before = m.qmap["q"].copy()
    phi_before = m.qmap["phi"].copy()

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

    calls = []
    original = model_module.find_ellipse_parameters

    def spy(image):
        calls.append(np.asarray(image).copy())
        return original(image)

    with patch.object(model_module, "find_ellipse_parameters", side_effect=spy):
        partition = m.compute_partition(
            mode="eq-ephi", use_groupindex_for_subpartition=True, dq_num=1,
            sq_num=8, dp_num=2, sp_num=4,
        )

    assert partition is not None
    # one whole-mask fit (the fallback for rho/phi) + one fit per group
    assert len(calls) == 1 + num_groups
    whole_mask_call, group_calls = calls[0], calls[1:]
    np.testing.assert_array_equal(whole_mask_call, m.mask)
    for group_mask in group_calls:
        assert 0 < group_mask.sum() < m.mask.sum()

    # self.qmap must be restored to the true q/phi maps afterward, not left
    # swapped to the ellipse-fit rho/phi
    np.testing.assert_array_equal(m.qmap["q"], q_before)
    np.testing.assert_array_equal(m.qmap["phi"], phi_before)


def test_compute_partition_with_groupindex_for_subpartition_on_xy_mode(
    tmp_path, make_hdf
):
    """use_groupindex_for_subpartition works for any axis0, not just q/rho —
    here the constraint groups are defined on 'x' and drive the x-y mode's
    dynamic-x sub-partition."""
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True

    x = m.qmap["x"]
    x_min, x_max = x[m.mask].min(), x[m.mask].max()
    x_mid = (x_min + x_max) / 2.0
    num_groups = 2
    constraints = [
        ("x", "AND", m.qmap_unit["x"], x_min, x_mid),
        ("x", "OR", m.qmap_unit["x"], x_mid, x_max),
    ]
    m.mask_evaluate("mask_parameter", constraints=constraints)
    m.mask_apply("mask_parameter")
    assert m.get_parameter_group_count() == num_groups

    sq_num = 8
    partition = m.compute_partition(
        mode="x-y", use_groupindex_for_subpartition=True, dq_num=1, sq_num=sq_num,
        dp_num=2, sp_num=4,
    )

    assert partition is not None
    assert partition["dynamic_num_pts"][0] == num_groups
    assert partition["static_num_pts"][0] == num_groups * sq_num
    assert check_consistency(
        partition["dynamic_roi_map"], partition["static_roi_map"], m.mask
    )


def test_compute_partition_with_groupindex_for_subpartition_without_groups_raises(
    tmp_path, make_hdf
):
    """use_groupindex_for_subpartition without an evaluated parametrization mask
    must fail loudly rather than silently falling back to a linear rebin."""
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True

    with pytest.raises(RuntimeError):
        m.compute_partition(
            mode="q-phi", use_groupindex_for_subpartition=True, sq_num=8, dp_num=2,
            sp_num=4,
        )
    assert m.new_partition is None


def test_compute_partition_with_groupindex_for_subpartition_raises_on_empty_group(
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
            mode="q-phi", use_groupindex_for_subpartition=True, dq_num=1, sq_num=9,
            dp_num=2, sp_num=4,
        )
    assert m.new_partition is None


def test_compute_partition_with_groupindex_for_subpartition_raises_on_overlap(
    tmp_path, make_hdf
):
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
            mode="q-phi", use_groupindex_for_subpartition=True, dq_num=1, sq_num=8,
            dp_num=2, sp_num=4,
        )
    assert m.new_partition is None


def test_compute_partition_with_groupindex_for_subpartition_uses_draw_source(
    tmp_path, make_hdf
):
    """use_groupindex_for_subpartition works from Draw-tab groups too, not just
    Parametrization — same end-to-end shape as the existing parametrization test."""
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True

    m.add_polygon([(0, 0), (0, 4), (16, 4), (16, 0)], mode="inclusive", group_index=1)
    m.add_polygon([(0, 6), (0, 12), (16, 12), (16, 6)], mode="inclusive", group_index=2)
    m.evaluate_draw()
    m.mask_apply("mask_draw")
    assert m.get_active_group_source() == "mask_draw"
    num_groups = 2

    sq_num = 8
    partition = m.compute_partition(
        mode="q-phi", use_groupindex_for_subpartition=True, dq_num=1, sq_num=sq_num,
        dp_num=2, sp_num=4,
    )

    assert partition is not None
    assert partition["dynamic_num_pts"][0] == num_groups
    assert check_consistency(
        partition["dynamic_roi_map"], partition["static_roi_map"], m.mask
    )


def test_compute_partition_with_groupindex_for_subpartition_prefers_later_source(
    tmp_path, make_hdf
):
    """If both Parametrization and Draw have valid groups, use whichever was
    (re-)evaluated more recently — here, 3 draw groups evaluated after 2
    parametrization groups."""
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True

    m.mask_evaluate("mask_parameter", constraints=_two_group_constraints(m))
    m.mask_apply("mask_parameter")

    m.add_polygon([(0, 0), (0, 3), (16, 3), (16, 0)], mode="inclusive", group_index=1)
    m.add_polygon([(0, 4), (0, 7), (16, 7), (16, 4)], mode="inclusive", group_index=2)
    m.add_polygon([(0, 8), (0, 11), (16, 11), (16, 8)], mode="inclusive", group_index=3)
    m.evaluate_draw()
    m.mask_apply("mask_draw")

    partition = m.compute_partition(
        mode="q-phi", use_groupindex_for_subpartition=True, dq_num=1, sq_num=9,
        dp_num=2, sp_num=4,
    )

    assert partition["dynamic_num_pts"][0] == 3  # draw's group count, not parametrization's


def test_evaluate_draw_group_index_map_labels_grouped_inclusive_rois(tmp_path, make_hdf):
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True

    m.add_polygon([(0, 0), (0, 4), (4, 4), (4, 0)], mode="inclusive", group_index=1)
    m.add_polygon([(8, 8), (8, 12), (12, 12), (12, 8)], mode="inclusive", group_index=2)
    m.evaluate_draw()
    m.mask_apply("mask_draw")

    assert m.get_draw_group_count() == 2
    gmap = m.mask_kernel.workers["mask_draw"].group_index_map
    assert gmap[1, 1] == 1
    assert gmap[9, 9] == 2


def test_get_draw_group_count_is_zero_before_any_draw_evaluate(tmp_path, make_hdf):
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True
    assert m.get_draw_group_count() == 0


def _two_group_constraints(m):
    q = m.qmap["q"]
    q_min, q_max = q[m.mask].min(), q[m.mask].max()
    q_mid = (q_min + q_max) / 2.0
    return [
        ("q", "AND", m.qmap_unit["q"], q_min, q_mid),
        ("q", "OR", m.qmap_unit["q"], q_mid, q_max),
    ]


def _draw_two_groups(m):
    m.add_polygon([(0, 0), (0, 4), (4, 4), (4, 0)], mode="inclusive", group_index=1)
    m.add_polygon([(8, 8), (8, 12), (12, 12), (12, 8)], mode="inclusive", group_index=2)
    m.evaluate_draw()


def test_get_active_group_count_zero_when_neither_source_has_groups(tmp_path, make_hdf):
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True
    assert m.get_active_group_count() == 0
    assert m.get_active_group_source() is None


def test_get_active_group_source_is_parametrization_when_only_it_has_groups(
    tmp_path, make_hdf
):
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True
    m.mask_evaluate("mask_parameter", constraints=_two_group_constraints(m))
    assert m.get_active_group_source() == "mask_parameter"
    assert m.get_active_group_count() == 2


def test_get_active_group_source_is_draw_when_only_it_has_groups(tmp_path, make_hdf):
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True
    _draw_two_groups(m)
    assert m.get_active_group_source() == "mask_draw"
    assert m.get_active_group_count() == 2


def test_get_active_group_source_prefers_more_recently_evaluated(tmp_path, make_hdf):
    """When both sources have valid groups, the one (re-)evaluated later wins."""
    path = make_hdf(_frames(), name="scan.h5")
    m = SimpleMaskModel()
    assert m.read_data(path, beamline="APS_8IDI", num_frames=0) is True

    m.mask_evaluate("mask_parameter", constraints=_two_group_constraints(m))
    _draw_two_groups(m)  # evaluated later
    assert m.get_active_group_source() == "mask_draw"

    m.mask_evaluate("mask_parameter", constraints=_two_group_constraints(m))  # now later
    assert m.get_active_group_source() == "mask_parameter"


def test_importing_core_does_not_import_qt():
    code = "import pysimplemask, sys; print('PySide6' in sys.modules)"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "False"
