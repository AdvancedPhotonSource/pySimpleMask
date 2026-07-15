"""GUI controller regression tests (run headless via the offscreen Qt platform)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

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
