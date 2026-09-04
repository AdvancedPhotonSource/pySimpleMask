# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""GUI controller regression tests (run headless via the offscreen Qt platform)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from pysimplemask.gui.control.main_window import SimpleMaskGUI


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _load_gui(tmp_path, frames):
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as h:
        h["/entry/data/data"] = np.asarray(frames)
    gui = SimpleMaskGUI()
    assert gui.sm.read_data(str(path), beamline="APS_8IDI", num_frames=0)
    # Mirror what load() does: detect rawdata and configure the frame controls.
    has_rawdata, num_frames = gui._detect_rawdata()
    gui._set_rawdata_enabled(has_rawdata)
    if has_rawdata:
        gui.horizontalSlider_frame.setRange(0, num_frames - 1)
        gui.horizontalSlider_frame.setValue(0)
        gui.spinBox_current_frame.setRange(0, num_frames - 1)
        gui.spinBox_current_frame.setValue(0)
        gui._pending_frame_idx = 0
    return gui


def test_update_parameters_accepts_ndarray_center(qapp, tmp_path):
    # find_center returns an np.ndarray; update_parameters must accept it.
    # Regression: `if new_center_vh:` raised ValueError on a 2-element array.
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.update_parameters(new_center_vh=np.array([8.0, 6.0]))
    cy, cx = gui.sm.get_center(mode="vh")
    assert np.allclose([cy, cx], [8.0, 6.0], atol=1e-6)


def test_find_center_button_no_crash(qapp, tmp_path):
    # Full reported path: find_center() -> sm.find_center() (ndarray) ->
    # update_parameters(new_center_vh=ndarray). Must not raise.
    rng = np.random.default_rng(0)
    frames = rng.integers(1, 50, size=(3, 32, 30)).astype(np.uint16)
    gui = _load_gui(tmp_path, frames)
    gui.find_center()
    cy, cx = gui.sm.get_center(mode="vh")
    assert np.isfinite(cy) and np.isfinite(cx)


def test_update_xmap_limits_excludes_masked_pixels(qapp, tmp_path):
    """The min/max shown and the spinbox range must ignore masked-out pixels."""
    gui = _load_gui(tmp_path, np.ones((2, 8, 8), dtype=np.uint16))
    xmap_name = next(iter(gui.sm.qmap))
    gui.comboBox_param_xmap_name.addItem(xmap_name)
    gui.comboBox_param_xmap_name.setCurrentText(xmap_name)

    xmap = gui.sm.qmap[xmap_name]
    max_idx = np.unravel_index(np.argmax(xmap), xmap.shape)
    gui.sm.mask[max_idx] = False
    unmasked_max = xmap[gui.sm.mask].max()
    assert unmasked_max < xmap.max()  # sanity: the mask actually excludes the peak

    gui.update_xmap_limits()

    assert gui.doubleSpinBox_param_vend.value() == pytest.approx(unmasked_max)
    assert gui.doubleSpinBox_param_vend.maximum() == pytest.approx(unmasked_max)
    assert gui.doubleSpinBox_param_vbeg.minimum() == pytest.approx(
        xmap[gui.sm.mask].min()
    )


def test_log_scale_checkbox_switches_display_mode(qapp, tmp_path):
    """Toggling plot_log changes data_display channels 0 and 1."""
    rng = np.random.default_rng(42)
    frames = rng.integers(1, 200, size=(3, 16, 14)).astype(np.uint16)
    gui = _load_gui(tmp_path, frames)

    # Default: log checked — channel 0 should be log-scaled (all values finite, no very large raw counts)
    gui.plot_log.setChecked(True)
    gui.plot()
    ch0_log = gui.sm.dset.data_display[0].copy()

    # Uncheck: linear — channel 0 should be raw counts (larger values)
    gui.plot_log.setChecked(False)
    gui.plot()
    ch0_lin = gui.sm.dset.data_display[0].copy()

    # Log values are smaller than linear for data > 1 (log10(x) < x for x > 1)
    assert ch0_log.max() < ch0_lin.max(), "log-scaled max should be less than linear max"
    # Channel 0 under log should equal log10 of channel 0 under linear (approximately)
    np.testing.assert_allclose(ch0_log, np.log10(ch0_lin), rtol=1e-5)


def test_mask_apply_current_tab_dispatches_correctly(qapp, tmp_path):
    """mask_apply_current_tab calls mask_apply with the target for the active tab."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    calls = []

    with patch.object(gui, "mask_apply", side_effect=lambda t: calls.append(t)):
        # Tab 0: Files (blemish section removed from UI) → mask_file only
        gui.MaskWidget.setCurrentIndex(0)
        gui.mask_apply_current_tab()
        assert calls == ["mask_file"], f"Tab 0: got {calls}"
        calls.clear()

        # Tab 2: Binary → mask_threshold
        gui.MaskWidget.setCurrentIndex(2)
        gui.mask_apply_current_tab()
        assert calls == ["mask_threshold"], f"Tab 2: got {calls}"
        calls.clear()

        # Tab 5: Parametrization → mask_parameter
        gui.MaskWidget.setCurrentIndex(5)
        gui.mask_apply_current_tab()
        assert calls == ["mask_parameter"], f"Tab 5: got {calls}"


def test_tab_mask_targets_constant(qapp, tmp_path):
    """_TAB_MASK_TARGETS has one entry per MaskWidget tab (6 total)."""
    from pysimplemask.gui.control.main_window import _TAB_MASK_TARGETS
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    assert len(_TAB_MASK_TARGETS) == gui.MaskWidget.count()


def test_rawdata_item_disabled_by_default(qapp, tmp_path):
    """rawdata plot_index item is grayed out before any data is loaded."""
    gui = SimpleMaskGUI()
    model = gui.plot_index.model()
    item = model.item(0)   # index 0 = rawdata
    assert item is not None
    assert not item.isEnabled(), "rawdata should be disabled before data load"


def test_frame_controls_hidden_by_default(qapp, tmp_path):
    """Frame controls are invisible on startup."""
    gui = SimpleMaskGUI()
    assert not gui.label_frame.isVisible()
    assert not gui.horizontalSlider_frame.isVisible()
    assert not gui.spinBox_current_frame.isVisible()


def test_non_rawdata_channel_uses_correct_data_display_slice(qapp, tmp_path):
    """plot_index=1 (scattering) shows data_display[0], not data_display[1]."""
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16) * 5)
    # Navigate away then back to 1 to ensure the signal fires (combobox skips no-op setCurrentIndex)
    gui.plot_index.setCurrentIndex(2)
    gui.plot_index.setCurrentIndex(1)
    # mp1.image should be data_display[0] (the scattering channel)
    assert gui.mp1.image is not None
    expected = gui.sm.dset.data_display[0]
    np.testing.assert_array_equal(gui.mp1.image, expected)


def test_detect_rawdata_returns_false_for_non_hdf(qapp, tmp_path):
    """_detect_rawdata returns False for non-HDF files."""
    gui = SimpleMaskGUI()
    # No data loaded → not ready
    ok, n = gui._detect_rawdata()
    assert not ok
    assert n == 0


def test_detect_rawdata_returns_false_for_single_frame_hdf(qapp, tmp_path):
    """_detect_rawdata returns False when /entry/data/data has only 1 frame."""
    gui = _load_gui(tmp_path, np.ones((1, 20, 24), dtype=np.uint16))
    ok, n = gui._detect_rawdata()
    assert not ok
    assert n == 0


def test_detect_rawdata_returns_true_for_multi_frame_hdf(qapp, tmp_path):
    """_detect_rawdata returns True when /entry/data/data has > 1 frames."""
    gui = _load_gui(tmp_path, np.ones((5, 20, 24), dtype=np.uint16))
    ok, n = gui._detect_rawdata()
    assert ok
    assert n == 5


def test_rawdata_enabled_after_loading_multi_frame_hdf(qapp, tmp_path):
    """rawdata combobox item is enabled after loading a multi-frame HDF."""
    gui = _load_gui(tmp_path, np.ones((5, 20, 24), dtype=np.uint16))
    model = gui.plot_index.model()
    assert model.item(0).isEnabled(), "rawdata should be enabled for multi-frame HDF"


def test_rawdata_disabled_after_loading_single_frame_hdf(qapp, tmp_path):
    """rawdata combobox item is disabled after loading a single-frame HDF."""
    gui = _load_gui(tmp_path, np.ones((1, 20, 24), dtype=np.uint16))
    model = gui.plot_index.model()
    assert not model.item(0).isEnabled()


def test_set_apply_state_toggles_bold_with_enabled(qapp, tmp_path):
    """_set_apply_state keeps isEnabled() and font().bold() in lockstep."""
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui._set_apply_state(True)
    assert gui.btn_mask_apply.isEnabled()
    assert gui.btn_mask_apply.font().bold()
    gui._set_apply_state(False)
    assert not gui.btn_mask_apply.isEnabled()
    assert not gui.btn_mask_apply.font().bold()


def test_set_update_parameters_state_toggles_bold_with_enabled(qapp, tmp_path):
    """_set_update_parameters_state keeps isEnabled() and font().bold() in lockstep."""
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui._set_update_parameters_state(True)
    assert gui.btn_update_parameters.isEnabled()
    assert gui.btn_update_parameters.font().bold()
    gui._set_update_parameters_state(False)
    assert not gui.btn_update_parameters.isEnabled()
    assert not gui.btn_update_parameters.font().bold()


def test_apply_button_bold_after_evaluate_and_apply(qapp, tmp_path):
    """mask_evaluate_current_tab / mask_apply_current_tab drive bold via _set_apply_state."""
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.MaskWidget.setCurrentIndex(2)  # Binary tab -> mask_threshold
    gui.mask_evaluate_current_tab()
    assert gui.btn_mask_apply.isEnabled()
    assert gui.btn_mask_apply.font().bold()
    gui.mask_apply_current_tab()
    assert not gui.btn_mask_apply.isEnabled()
    assert not gui.btn_mask_apply.font().bold()


def test_update_parameters_button_bold_after_edit_and_sync(qapp, tmp_path):
    """display_metadata / update_parameter_to_dset drive bold via _set_update_parameters_state."""
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.display_metadata()
    assert not gui.btn_update_parameters.isEnabled()
    assert not gui.btn_update_parameters.font().bold()
    param = gui.metadata_parameter.children()[0]
    gui.update_parameter_to_dset(None, [(param, "value", param.value())])
    assert gui.btn_update_parameters.isEnabled()
    assert gui.btn_update_parameters.font().bold()


def _click_message_box(monkeypatch, button_text):
    """Patch QMessageBox.exec so it synchronously clicks the button with this text.

    QMessageBox.clickedButton() is set as a side effect of a button's clicked signal
    firing inside the real (blocking) exec() event loop. Calling .click() on the
    target button reproduces that without needing a real event loop.
    """
    def _fake_exec(self):
        for button in self.buttons():
            if button.text() == button_text:
                button.click()
                return 0
        raise AssertionError(f"no QMessageBox button with text {button_text!r}")

    monkeypatch.setattr(QMessageBox, "exec", _fake_exec)


def _forbid_message_box(monkeypatch):
    """Patch QMessageBox.exec to fail the test if a dialog is shown at all."""
    def _fail_exec(self):
        raise AssertionError("QMessageBox.exec should not have been called")

    monkeypatch.setattr(QMessageBox, "exec", _fail_exec)


def test_maybe_prompt_metadata_update_skips_dialog_when_synced(qapp, tmp_path, monkeypatch):
    """No dialog at all when metadata is already in sync."""
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui._set_update_parameters_state(False)
    _forbid_message_box(monkeypatch)
    gui._maybe_prompt_metadata_update()  # must not raise


def test_maybe_prompt_metadata_update_choosing_update_calls_update_parameters(qapp, tmp_path, monkeypatch):
    """Choosing 'Update Metadata' calls update_parameters()."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui._set_update_parameters_state(True)
    _click_message_box(monkeypatch, "Update Metadata")
    with patch.object(gui, "update_parameters") as mock_update:
        gui._maybe_prompt_metadata_update()
    mock_update.assert_called_once()


def test_maybe_prompt_metadata_update_choosing_continue_skips_update(qapp, tmp_path, monkeypatch):
    """Choosing 'Continue Without Updating' does not call update_parameters()."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui._set_update_parameters_state(True)
    _click_message_box(monkeypatch, "Continue Without Updating")
    with patch.object(gui, "update_parameters") as mock_update:
        gui._maybe_prompt_metadata_update()
    mock_update.assert_not_called()


def test_mask_evaluate_current_tab_checks_metadata_staleness(qapp, tmp_path):
    """mask_evaluate_current_tab calls the staleness check."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.MaskWidget.setCurrentIndex(2)
    with patch.object(gui, "_maybe_prompt_metadata_update") as mock_prompt:
        gui.mask_evaluate_current_tab()
    mock_prompt.assert_called_once()


def test_mask_apply_current_tab_checks_metadata_staleness(qapp, tmp_path):
    """mask_apply_current_tab calls the staleness check."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.MaskWidget.setCurrentIndex(2)
    with patch.object(gui, "_maybe_prompt_metadata_update") as mock_prompt:
        gui.mask_apply_current_tab()
    mock_prompt.assert_called_once()


def test_maybe_prompt_unapplied_mask_returns_true_when_nothing_pending(qapp, tmp_path, monkeypatch):
    """No dialog, and returns True, when there's no evaluated-but-unapplied mask."""
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui._set_apply_state(False)
    _forbid_message_box(monkeypatch)
    assert gui._maybe_prompt_unapplied_mask() is True


def test_maybe_prompt_unapplied_mask_apply_and_continue(qapp, tmp_path, monkeypatch):
    """'Apply & Continue' applies the pending mask directly (skipping the
    metadata prompt) and returns True."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui._set_apply_state(True)
    _click_message_box(monkeypatch, "Apply && Continue")
    with patch.object(gui, "_apply_current_tab_targets") as mock_apply:
        result = gui._maybe_prompt_unapplied_mask()
    assert result is True
    mock_apply.assert_called_once()


def test_maybe_prompt_unapplied_mask_apply_and_continue_skips_metadata_prompt(qapp, tmp_path, monkeypatch):
    """'Apply & Continue' does not re-trigger the metadata-staleness prompt,
    even if metadata is still stale — regression test for the nested-dialog
    finding from manual verification."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.MaskWidget.setCurrentIndex(2)  # Binary tab -> mask_threshold
    gui._set_apply_state(True)
    gui._set_update_parameters_state(True)  # metadata still stale
    _click_message_box(monkeypatch, "Apply && Continue")
    with patch.object(gui, "_maybe_prompt_metadata_update") as mock_metadata_prompt:
        result = gui._maybe_prompt_unapplied_mask()
    assert result is True
    mock_metadata_prompt.assert_not_called()
    assert not gui.btn_mask_apply.isEnabled()  # mask was actually applied


def test_maybe_prompt_unapplied_mask_continue_anyway(qapp, tmp_path, monkeypatch):
    """'Continue Anyway' returns True without applying the pending mask."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui._set_apply_state(True)
    _click_message_box(monkeypatch, "Continue Anyway")
    with patch.object(gui, "mask_apply_current_tab") as mock_apply:
        result = gui._maybe_prompt_unapplied_mask()
    assert result is True
    mock_apply.assert_not_called()


def test_maybe_prompt_unapplied_mask_cancel(qapp, tmp_path, monkeypatch):
    """'Cancel' returns False without applying the pending mask."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui._set_apply_state(True)
    _click_message_box(monkeypatch, "Cancel")
    with patch.object(gui, "mask_apply_current_tab") as mock_apply:
        result = gui._maybe_prompt_unapplied_mask()
    assert result is False
    mock_apply.assert_not_called()


def test_compute_partition_aborts_when_unapplied_mask_cancelled(qapp, tmp_path):
    """compute_partition does not compute anything if the prompt returns False."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    with patch.object(gui, "_maybe_prompt_unapplied_mask", return_value=False):
        with patch.object(gui.sm, "compute_partition") as mock_compute:
            gui.compute_partition()
    mock_compute.assert_not_called()


def test_compute_partition_proceeds_when_unapplied_mask_confirmed(qapp, tmp_path):
    """compute_partition proceeds normally if the prompt returns True."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    with patch.object(gui, "_maybe_prompt_unapplied_mask", return_value=True):
        with patch.object(gui.sm, "compute_partition") as mock_compute:
            gui.compute_partition()
    mock_compute.assert_called_once()


def test_compute_partition_refreshes_already_selected_dynamic_map(qapp, tmp_path):
    """Recomputing refreshes the image when the dynamic map is already selected."""
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.plot_index.setCurrentIndex(4)
    image_before = gui.mp1.image

    gui.compute_partition()

    assert gui.mp1.image is not image_before
    np.testing.assert_array_equal(gui.mp1.image, gui.sm.dset.data_display[3])


def test_update_parameters_refreshes_stale_dynamic_partition_display(qapp, tmp_path):
    """Clicking "Update Parameters" (recompute the qmap) while the dynamic
    partition is on screen must refresh it too, not just the qmap geometry
    channels — otherwise the displayed dynamic/static partition silently goes
    stale relative to the new geometry."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.compute_partition()
    gui.plot_index.setCurrentIndex(4)  # dynamic_q_partition

    with patch.object(
        gui.sm, "compute_partition", wraps=gui.sm.compute_partition
    ) as spy:
        gui.update_parameters(new_center_vh=(8.0, 6.0))

    spy.assert_called_once()
    np.testing.assert_array_equal(gui.mp1.image, gui.sm.dset.data_display[3])


def test_save_mask_does_not_report_success_when_partition_cancelled(qapp, tmp_path):
    """save_mask must not call save_partition (or show success) if compute_partition
    was aborted (e.g. by cancelling the unapplied-mask prompt) and left new_partition unset."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.comboBox_output_type.setCurrentText("Nexus-XPCS")
    assert gui.sm.new_partition is None

    save_path = str(tmp_path / "out.hdf")
    with patch.object(QFileDialog, "getSaveFileName", return_value=(save_path, "")):
        with patch.object(gui, "compute_partition") as mock_compute:
            with patch.object(gui.sm, "save_partition") as mock_save:
                gui.save_mask()
    mock_compute.assert_called_once()
    mock_save.assert_not_called()


def _apply_parametrization_groups(gui, constraints):
    gui.sm.mask_evaluate("mask_parameter", constraints=constraints)
    gui.mask_apply("mask_parameter")


def _two_group_constraints(gui):
    q = gui.sm.qmap["q"]
    q_min, q_max = q[gui.sm.mask].min(), q[gui.sm.mask].max()
    q_mid = (q_min + q_max) / 2.0
    return [
        ("q", "AND", gui.sm.qmap_unit["q"], q_min, q_mid),
        ("q", "OR", gui.sm.qmap_unit["q"], q_mid, q_max),
    ]


def test_groupindex_checkbox_reverts_when_no_groups_available(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    assert gui.sm.get_parameter_group_count() == 0

    gui.checkBox_use_groupindex_for_subpartition.setChecked(True)

    assert gui.checkBox_use_groupindex_for_subpartition.isChecked() is False
    assert gui.sb_dqnum.isEnabled()
    assert gui.statusbar.currentMessage() != ""


def test_groupindex_checkbox_sets_both_axes_and_stays_enabled(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    _apply_parametrization_groups(gui, _two_group_constraints(gui))

    gui.checkBox_use_groupindex_for_subpartition.setChecked(True)

    assert gui.checkBox_use_groupindex_for_subpartition.isChecked() is True
    assert gui.sb_dqnum.value() == 1
    assert gui.sb_sqnum.value() == 9
    assert gui.sb_dpnum.value() == 1
    assert gui.sb_spnum.value() == 9
    assert gui.sb_dqnum.isEnabled()
    assert gui.sb_sqnum.isEnabled()
    assert gui.sb_dpnum.isEnabled()
    assert gui.sb_spnum.isEnabled()


def test_groupindex_checkbox_sets_both_axes_on_ellipse_tab(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.tabWidget.setCurrentIndex(2)  # ellipse
    _apply_parametrization_groups(gui, _two_group_constraints(gui))

    gui.checkBox_use_groupindex_for_subpartition.setChecked(True)

    assert gui.checkBox_use_groupindex_for_subpartition.isChecked() is True
    assert gui.sb_dxnum_2.value() == 1
    assert gui.sb_sxnum_2.value() == 9
    assert gui.sb_dynum_2.value() == 1
    assert gui.sb_synum_2.value() == 9
    assert gui.sb_dxnum_2.isEnabled()
    assert gui.sb_sxnum_2.isEnabled()
    assert gui.sb_dynum_2.isEnabled()
    assert gui.sb_synum_2.isEnabled()


def test_groupindex_checkbox_sets_both_axes_on_xy_mesh_tab(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.tabWidget.setCurrentIndex(1)  # xy-mesh
    _apply_parametrization_groups(gui, _two_group_constraints(gui))

    gui.checkBox_use_groupindex_for_subpartition.setChecked(True)

    assert gui.checkBox_use_groupindex_for_subpartition.isChecked() is True
    assert gui.sb_dxnum.value() == 1
    assert gui.sb_sxnum.value() == 9
    assert gui.sb_dynum.value() == 1
    assert gui.sb_synum.value() == 9
    assert gui.sb_dxnum.isEnabled()
    assert gui.sb_sxnum.isEnabled()
    assert gui.sb_dynum.isEnabled()
    assert gui.sb_synum.isEnabled()


def test_groupindex_checkbox_sets_both_axes_on_general_tab(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.tabWidget.setCurrentIndex(3)  # general
    _apply_parametrization_groups(gui, _two_group_constraints(gui))

    gui.checkBox_use_groupindex_for_subpartition.setChecked(True)

    assert gui.checkBox_use_groupindex_for_subpartition.isChecked() is True
    assert gui.sb_partition_dn0.value() == 1
    assert gui.sb_partition_sn0.value() == 9
    assert gui.sb_partition_dn1.value() == 1
    assert gui.sb_partition_sn1.value() == 9
    assert gui.sb_partition_dn0.isEnabled()
    assert gui.sb_partition_sn0.isEnabled()
    assert gui.sb_partition_dn1.isEnabled()
    assert gui.sb_partition_sn1.isEnabled()


def test_groupindex_checkbox_does_not_clobber_manual_dqnum_after_reapply(qapp, tmp_path):
    """dq_num/sq_num are now genuine user-editable per-group parameters, not a
    read-only mirror of num_groups — a manual override must survive a later
    mask re-apply instead of being silently reset."""
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    q = gui.sm.qmap["q"]
    q_min, q_max = q[gui.sm.mask].min(), q[gui.sm.mask].max()
    unit = gui.sm.qmap_unit["q"]

    _apply_parametrization_groups(gui, _two_group_constraints(gui))
    gui.checkBox_use_groupindex_for_subpartition.setChecked(True)
    assert gui.sb_dqnum.value() == 1
    assert gui.sb_sqnum.value() == 9

    gui.sb_dqnum.setValue(5)
    gui.sb_sqnum.setValue(20)

    q1 = q_min + (q_max - q_min) / 3.0
    q2 = q_min + 2 * (q_max - q_min) / 3.0
    _apply_parametrization_groups(
        gui,
        [
            ("q", "AND", unit, q_min, q1),
            ("q", "OR", unit, q1, q2),
            ("q", "OR", unit, q2, q_max),
        ],
    )

    assert gui.sb_dqnum.value() == 5     # untouched by the reapply
    assert gui.sb_sqnum.value() == 20    # untouched by the reapply
    assert gui.checkBox_use_groupindex_for_subpartition.isEnabled()  # 3 groups now present
    assert "(3 groups)" in gui.checkBox_use_groupindex_for_subpartition.text()


def test_groupindex_checkbox_disabled_with_no_groups_on_qphi_tab(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.tabWidget.setCurrentIndex(0)  # q-phi

    assert not gui.checkBox_use_groupindex_for_subpartition.isEnabled()
    assert "(0 groups)" in gui.checkBox_use_groupindex_for_subpartition.text()


def test_groupindex_checkbox_label_shows_singular_group_count(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    q = gui.sm.qmap["q"]
    q_min, q_max = q[gui.sm.mask].min(), q[gui.sm.mask].max()
    unit = gui.sm.qmap_unit["q"]
    _apply_parametrization_groups(gui, [("q", "AND", unit, q_min, q_max)])

    assert gui.sm.get_active_group_count() == 1
    assert "(1 group)" in gui.checkBox_use_groupindex_for_subpartition.text()
    assert "(1 groups)" not in gui.checkBox_use_groupindex_for_subpartition.text()


def test_groupindex_checkbox_enabled_on_qphi_tab_with_groups(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.tabWidget.setCurrentIndex(0)  # q-phi

    _apply_parametrization_groups(gui, _two_group_constraints(gui))

    assert gui.checkBox_use_groupindex_for_subpartition.isEnabled()
    assert "(2 groups)" in gui.checkBox_use_groupindex_for_subpartition.text()


def test_groupindex_checkbox_disabled_with_no_groups_on_xy_mesh_tab(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.tabWidget.setCurrentIndex(1)  # xy-mesh

    assert not gui.checkBox_use_groupindex_for_subpartition.isEnabled()


def test_groupindex_checkbox_enabled_on_xy_mesh_tab_with_groups(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    _apply_parametrization_groups(gui, _two_group_constraints(gui))

    gui.tabWidget.setCurrentIndex(1)  # xy-mesh

    assert gui.checkBox_use_groupindex_for_subpartition.isEnabled()


def test_groupindex_checkbox_disabled_with_no_groups_on_general_tab(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.tabWidget.setCurrentIndex(3)  # general

    assert not gui.checkBox_use_groupindex_for_subpartition.isEnabled()


def test_groupindex_checkbox_enabled_on_general_tab_with_groups(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    _apply_parametrization_groups(gui, _two_group_constraints(gui))

    gui.tabWidget.setCurrentIndex(3)  # general

    assert gui.checkBox_use_groupindex_for_subpartition.isEnabled()


def test_groupindex_checkbox_enabled_on_ellipse_tab_with_groups(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    _apply_parametrization_groups(gui, _two_group_constraints(gui))

    gui.tabWidget.setCurrentIndex(2)  # ellipse

    assert gui.checkBox_use_groupindex_for_subpartition.isEnabled()


def test_groupindex_checkbox_enabled_state_updates_on_mask_apply(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.tabWidget.setCurrentIndex(0)  # q-phi
    assert not gui.checkBox_use_groupindex_for_subpartition.isEnabled()

    _apply_parametrization_groups(gui, _two_group_constraints(gui))

    assert gui.checkBox_use_groupindex_for_subpartition.isEnabled()


def test_compute_partition_with_empty_group_does_not_report_false_success(qapp, tmp_path):
    """If a constraint group ends up empty, compute_partition_general now raises
    instead of silently returning None — verify the GUI doesn't then show a
    misleading "New partition is generated." message or update new_partition."""
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    q = gui.sm.qmap["q"]
    q_min, q_max = q[gui.sm.mask].min(), q[gui.sm.mask].max()
    q_mid = (q_min + q_max) / 2.0
    unit = gui.sm.qmap_unit["q"]
    _apply_parametrization_groups(
        gui,
        [
            ("q", "AND", unit, q_min, q_mid),
            ("q", "OR", unit, q_max + 100, q_max + 200),  # empty group
            ("q", "OR", unit, q_mid, q_max),
        ],
    )
    gui.checkBox_use_groupindex_for_subpartition.setChecked(True)
    assert gui.sb_dqnum.value() == 1

    gui.statusbar.clearMessage()
    gui.compute_partition()

    assert gui.sm.new_partition is None
    assert gui.statusbar.currentMessage() != "New partition is generated."


def _select_draw(gui, sl_type="Circle", sl_mode="exclusive"):
    gui.MaskWidget.setCurrentIndex(1)  # Draw tab
    gui.cb_selector_type.setCurrentText(sl_type)
    gui.cb_selector_mode.setCurrentText(sl_mode)


@pytest.fixture
def draw_gui(qapp, tmp_path):
    """A loaded GUI for Draw-tab (pyqtgraph ROI) tests.

    Explicitly tears down the ROI-bearing view and forces a GC pass right
    after each test: leaving pg.ROI/QGraphicsObject items from several
    SimpleMaskGUI instances for Python's cyclic GC to collect lazily (e.g.
    during a *later* test's ROI construction) segfaults under the offscreen
    QPA platform.
    """
    import gc

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    yield gui
    gui.mp1.clear()
    gui.close()
    del gui
    gc.collect()


def test_draw_group_table_columns_stretch_to_fill_width(qapp, tmp_path):
    """tableView_draw's columns should stretch to fill the widget, like the
    Parametrization tab's tableView does (group-index stays content-sized)."""
    from PySide6.QtWidgets import QHeaderView

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    header = gui.tableView_draw.horizontalHeader()
    assert header.sectionResizeMode(0) == QHeaderView.ResizeToContents
    assert header.sectionResizeMode(1) == QHeaderView.Stretch
    assert header.sectionResizeMode(2) == QHeaderView.Stretch


def test_inclusive_draw_adds_row_to_draw_group_table(draw_gui):
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")

    gui.add_drawing()

    assert gui.model_draw.rowCount() == 1
    assert gui.model_draw.data(gui.model_draw.index(0, 0), Qt.DisplayRole) == "1"
    assert gui.model_draw.data(gui.model_draw.index(0, 1), Qt.DisplayRole) == "Circle"


def test_moving_roi_updates_table_center(draw_gui):
    """Dragging/resizing a live drawn ROI must refresh its table row's
    displayed center, not leave it frozen at the position it was drawn at."""
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    roi_key = gui.model_draw.roi_keys()[0]
    center_col = gui.model_draw.headers.index("center (row, col)")
    before = gui.model_draw.data(gui.model_draw.index(0, center_col), Qt.DisplayRole)

    roi = gui.mp1.roi[roi_key]
    roi.setPos(roi.pos() + pg.Point(5, 5))

    after = gui.model_draw.data(gui.model_draw.index(0, center_col), Qt.DisplayRole)
    assert after != before


def test_exclusive_draw_adds_row_with_zero_group_index(draw_gui):
    """Exclusive draws are tracked too (for completeness), but always show
    group-index 0 — they're never part of a use_groupindex_for_subpartition group."""
    gui = draw_gui
    _select_draw(gui, "Circle", "exclusive")

    gui.add_drawing()

    assert gui.model_draw.rowCount() == 1
    assert gui.model_draw.data(gui.model_draw.index(0, 0), Qt.DisplayRole) == "0"


def test_mixed_inclusive_exclusive_draws_number_inclusive_only(draw_gui):
    """group-index numbering counts only inclusive rows; interleaved
    exclusive rows show 0 and don't consume a number."""
    gui = draw_gui
    _select_draw(gui, "Circle", "exclusive")
    gui.add_drawing()
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    _select_draw(gui, "Rectangle", "exclusive")
    gui.add_drawing()
    _select_draw(gui, "Rectangle", "inclusive")
    gui.add_drawing()

    displayed = [
        gui.model_draw.data(gui.model_draw.index(r, 0), Qt.DisplayRole)
        for r in range(4)
    ]
    assert displayed == ["0", "1", "0", "2"]


def test_mask_evaluate_draw_excludes_exclusive_rows_from_group_count(draw_gui):
    """The exclusive row in the table must not affect num_groups even though
    it's now tracked alongside the inclusive ones."""
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    _select_draw(gui, "Rectangle", "exclusive")
    gui.add_drawing()

    gui.mask_evaluate("mask_draw")

    assert gui.model_draw.rowCount() == 2
    assert gui.sm.get_draw_group_count() == 1


def test_multiple_inclusive_draws_increment_group_index(draw_gui):
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")

    gui.add_drawing()
    gui.add_drawing()

    assert gui.model_draw.rowCount() == 2
    assert gui.model_draw.data(gui.model_draw.index(1, 0), Qt.DisplayRole) == "2"


def test_removing_roi_from_canvas_removes_draw_group_table_row(draw_gui):
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    roi_key = gui.model_draw.roi_keys()[0]

    gui.mp1.remove_item(roi_key)

    assert gui.model_draw.rowCount() == 0


def test_replot_does_not_remove_drawn_rois(draw_gui):
    """plot() calls mp1.clear() on every redraw (channel switch, log toggle,
    after every Evaluate/Apply); drawn ROI shapes and their table rows must
    survive that, not just the ROI's own explicit remove handle."""
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    roi_key = gui.model_draw.roi_keys()[0]

    gui.mp1.clear()

    assert roi_key in gui.mp1.roi
    assert gui.model_draw.rowCount() == 1


def test_clear_still_removes_non_roi_overlay_items(draw_gui):
    """clear() must still wipe non-drawn overlay items (e.g. the center
    marker plot() re-adds on every redraw) — only roi_* drawn shapes survive."""
    gui = draw_gui
    marker = pg.ScatterPlotItem()
    gui.mp1.add_item(marker, label="center")

    gui.mp1.clear()

    assert "center" not in gui.mp1.roi


def test_mask_evaluate_draw_wires_group_index_map_into_worker(draw_gui):
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    gui.add_drawing()
    _select_draw(gui, "Rectangle", "exclusive")
    gui.add_drawing()

    gui.mask_evaluate("mask_draw")

    assert gui.sm.get_draw_group_count() == 2


def test_mask_evaluate_draw_keeps_roi_visible_on_canvas(draw_gui):
    """Evaluate previews the mask but must not remove the ROI from the canvas
    — matching every other tab's evaluate=preview/apply=commit split. Only
    Apply actually consumes/removes the drawn ROI."""
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    roi_key = gui.model_draw.roi_keys()[0]

    gui.mask_evaluate("mask_draw")

    assert roi_key in gui.mp1.roi
    assert gui.model_draw.rowCount() == 1


def test_mask_apply_draw_keeps_roi_on_canvas_and_table_row(draw_gui):
    """Apply commits the mask but must not touch the canvas or the table —
    the ROI stays live/draggable until the user explicitly removes it."""
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    roi_key = gui.model_draw.roi_keys()[0]
    gui.mask_evaluate("mask_draw")

    gui.mask_apply("mask_draw")

    assert roi_key in gui.mp1.roi
    assert gui.model_draw.rowCount() == 1


def test_new_draw_after_apply_appends_instead_of_resetting(draw_gui):
    """Drawing again after a previous batch was applied keeps growing the
    table (group-index keeps climbing) rather than starting over at 1."""
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    gui.mask_evaluate("mask_draw")
    gui.mask_apply("mask_draw")
    assert gui.model_draw.rowCount() == 1

    gui.add_drawing()

    assert gui.model_draw.rowCount() == 2
    assert gui.model_draw.data(gui.model_draw.index(1, 0), Qt.DisplayRole) == "2"


def test_repeated_evaluate_without_apply_reflects_all_still_live_rois(draw_gui):
    """Since Evaluate no longer removes ROIs, a second Evaluate before any
    Apply still sees every ROI drawn so far as live (nothing was consumed)."""
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    gui.mask_evaluate("mask_draw")
    assert gui.sm.get_draw_group_count() == 1

    gui.add_drawing()
    gui.mask_evaluate("mask_draw")

    assert gui.sm.get_draw_group_count() == 2


def test_remove_draw_row_removes_roi_from_canvas_and_table(draw_gui):
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    roi_key = gui.model_draw.roi_keys()[0]

    gui._remove_draw_row(0)

    assert gui.model_draw.rowCount() == 0
    assert roi_key not in gui.mp1.roi


def test_remove_draw_row_invalid_row_is_noop(draw_gui):
    gui = draw_gui
    gui._remove_draw_row(0)  # empty table
    assert gui.model_draw.rowCount() == 0


class _FakeContextMenu:
    """Stand-in for QMenu in tests: QMenu.exec() opens a real modal popup
    under the offscreen QPA platform that can't be intercepted the way
    QMessageBox.exec can be monkeypatched (confirmed hangs otherwise), so we
    swap out the whole class instead of trying to patch exec()."""

    selected_text = None   # which addAction() text exec() should "pick"
    instances = []

    def __init__(self, parent=None):
        self._actions = {}
        _FakeContextMenu.instances.append(self)

    def addAction(self, text):
        action = object()
        self._actions[text] = action
        return action

    def exec(self, pos):
        return self._actions.get(_FakeContextMenu.selected_text)


def test_draw_table_context_menu_remove_removes_row_and_roi(draw_gui, monkeypatch):
    """Right-click > Remove on a draw-table row removes both the row and the
    still-live ROI from the canvas (mirrors the ROI's own remove handle)."""
    import pysimplemask.gui.control.main_window as mw

    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    roi_key = gui.model_draw.roi_keys()[0]

    _FakeContextMenu.instances = []
    _FakeContextMenu.selected_text = "Remove"
    monkeypatch.setattr(mw, "QMenu", _FakeContextMenu)
    pos = gui.tableView_draw.visualRect(gui.model_draw.index(0, 0)).center()

    gui.tableView_draw.customContextMenuRequested.emit(pos)

    assert len(_FakeContextMenu.instances) == 1
    assert gui.model_draw.rowCount() == 0
    assert roi_key not in gui.mp1.roi


def test_draw_table_context_menu_does_nothing_on_empty_area(draw_gui, monkeypatch):
    import pysimplemask.gui.control.main_window as mw

    gui = draw_gui
    _FakeContextMenu.instances = []
    monkeypatch.setattr(mw, "QMenu", _FakeContextMenu)

    gui.tableView_draw.customContextMenuRequested.emit(gui.tableView_draw.rect().center())

    assert _FakeContextMenu.instances == []


def test_mask_apply_draw_group_index_is_contiguous_for_live_rois_only(draw_gui):
    """End-to-end regression: evaluate -> apply -> explicitly remove -> draw a
    new ROI -> evaluate again must produce a fresh, correct group count (1),
    not leak state from the removed ROI's earlier group_index."""
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    gui.mask_evaluate("mask_draw")
    gui.mask_apply("mask_draw")
    gui._remove_draw_row(0)  # explicitly discard it; table row 1 is gone too now

    gui.add_drawing()  # new ROI becomes table row 1 again (only *live* one)
    gui.mask_evaluate("mask_draw")

    assert gui.model_draw.rowCount() == 1
    assert gui.sm.get_draw_group_count() == 1


def test_add_new_inclusive_roi_after_evaluate_apply_includes_it_too(draw_gui):
    """Regression: draw+evaluate+apply one inclusive ROI, then (without
    touching that ROI) draw a second, non-overlapping inclusive ROI and
    evaluate+apply again. The second region must actually widen gui.sm.mask
    — a previous bug ANDed the fresh mask_draw computation against the
    already-narrowed mask left over from the first apply, silently excluding
    every new region added this way."""
    gui = draw_gui
    _select_draw(gui, "Rectangle", "inclusive")

    gui.add_drawing()
    roi1 = gui.mp1.roi[gui.model_draw.roi_keys()[0]]
    roi1.setPos([0, 0])
    roi1.setSize([8, 8])
    gui.mask_evaluate("mask_draw")
    gui.mask_apply("mask_draw")
    mask_after_first = gui.sm.mask.copy()

    gui.add_drawing()
    roi2 = gui.mp1.roi[gui.model_draw.roi_keys()[1]]
    roi2.setPos([16, 0])
    roi2.setSize([8, 8])
    gui.mask_evaluate("mask_draw")
    gui.mask_apply("mask_draw")

    assert gui.sm.mask.sum() > mask_after_first.sum()


def test_compute_partition_uses_groupindex_for_subpartition_end_to_end(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    _apply_parametrization_groups(gui, _two_group_constraints(gui))
    gui.checkBox_use_groupindex_for_subpartition.setChecked(True)
    gui.sb_sqnum.setValue(8)

    gui.compute_partition()

    assert gui.sm.new_partition is not None
    assert gui.sm.new_partition["dynamic_num_pts"][0] == 2


def test_compute_partition_uses_groupindex_for_subpartition_on_ellipse_tab_end_to_end(qapp, tmp_path):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.tabWidget.setCurrentIndex(2)  # ellipse
    _apply_parametrization_groups(gui, _two_group_constraints(gui))
    gui.checkBox_use_groupindex_for_subpartition.setChecked(True)

    gui.compute_partition()

    assert gui.sm.new_partition is not None
    assert gui.sm.new_partition["dynamic_num_pts"][0] == 2


def test_compute_partition_uses_groupindex_for_subpartition_on_xy_mesh_tab_end_to_end(
    qapp, tmp_path
):
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.tabWidget.setCurrentIndex(1)  # xy-mesh
    _apply_parametrization_groups(gui, _two_group_constraints(gui))
    gui.checkBox_use_groupindex_for_subpartition.setChecked(True)

    gui.compute_partition()

    assert gui.sm.new_partition is not None
    assert gui.sm.new_partition["dynamic_num_pts"][0] == 2


def _draw_two_inclusive_groups(gui):
    gui.MaskWidget.setCurrentIndex(1)  # Draw tab
    gui.cb_selector_type.setCurrentText("Circle")
    gui.cb_selector_mode.setCurrentText("inclusive")
    gui.add_drawing()
    gui.add_drawing()
    gui.mask_evaluate("mask_draw")
    gui.mask_apply("mask_draw")


def test_groupindex_checkbox_uses_draw_group_count(draw_gui):
    """The checkbox works from Draw-tab groups too, not just Parametrization."""
    gui = draw_gui
    _draw_two_inclusive_groups(gui)

    gui.checkBox_use_groupindex_for_subpartition.setChecked(True)

    assert gui.checkBox_use_groupindex_for_subpartition.isChecked() is True
    assert gui.sb_dqnum.value() == 1
    assert gui.sb_sqnum.value() == 9
    assert gui.sb_dqnum.isEnabled()


def test_groupindex_checkbox_does_not_clobber_manual_dqnum_after_draw_apply(draw_gui):
    """A manual dq_num override must survive a later draw-group re-apply, even
    when the newly-active source switches from parametrization to draw groups."""
    gui = draw_gui
    _apply_parametrization_groups(gui, _two_group_constraints(gui))
    gui.checkBox_use_groupindex_for_subpartition.setChecked(True)
    assert gui.sb_dqnum.value() == 1

    gui.sb_dqnum.setValue(7)

    gui.MaskWidget.setCurrentIndex(1)  # Draw tab
    gui.cb_selector_type.setCurrentText("Circle")
    gui.cb_selector_mode.setCurrentText("inclusive")
    gui.add_drawing()
    gui.add_drawing()
    gui.add_drawing()
    gui.mask_evaluate("mask_draw")
    gui.mask_apply("mask_draw")  # evaluated+applied later -> becomes active source

    assert gui.sm.get_active_group_source() == "mask_draw"
    assert gui.sb_dqnum.value() == 7  # untouched by the reapply


def test_compute_partition_uses_draw_groupindex_for_subpartition_end_to_end(draw_gui):
    """End-to-end through the real GUI call chain. A single inclusive group
    keeps the ROI geometry uncontentious on the tiny test canvas — the exact
    multi-group math is already covered at the model level (test_model_headless).
    """
    gui = draw_gui
    gui.MaskWidget.setCurrentIndex(1)  # Draw tab
    gui.cb_selector_type.setCurrentText("Circle")
    gui.cb_selector_mode.setCurrentText("inclusive")
    gui.add_drawing()
    gui.mask_evaluate("mask_draw")
    gui.mask_apply("mask_draw")
    gui.checkBox_use_groupindex_for_subpartition.setChecked(True)
    gui.sb_sqnum.setValue(4)

    gui.compute_partition()

    assert gui.sm.new_partition is not None
    assert gui.sm.new_partition["dynamic_num_pts"][0] == 1
