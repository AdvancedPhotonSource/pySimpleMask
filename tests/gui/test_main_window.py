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
