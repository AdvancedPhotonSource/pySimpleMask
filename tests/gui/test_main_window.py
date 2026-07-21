# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""GUI controller regression tests (run headless via the offscreen Qt platform)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

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
