# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""GUI controller regression tests (run headless via the offscreen Qt platform)."""

import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pyqtgraph as pg
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox

from pysimplemask.core.mask import MaskAssemble
from pysimplemask.core.reader.base_reader import DISPLAY_FIELD, FileReader
from pysimplemask.gui.control.main_window import (
    _FRAME_RANGE_CUSTOM,
    _FRAME_RANGE_NUM_FRAMES,
    _RAW_PLOT_INDICES,
    _RAWDATA_IDX,
    _RAWDATA_MASK_IDX,
    SimpleMaskGUI,
    _MaskTab,
    _plot_index,
)


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
    # a non-angular, integer-valued map: its spin boxes are clamped to the
    # data, and its values survive the spin boxes' decimal rounding exactly
    xmap_name = "x"
    gui.comboBox_param_xmap_name.addItem(xmap_name)
    gui.comboBox_param_xmap_name.setCurrentText(xmap_name)

    xmap = gui.sm.qmap[xmap_name]
    gui.sm.mask[:, -1] = False  # drop the whole last column (x == max)
    unmasked_max = xmap[gui.sm.mask].max()
    assert unmasked_max < xmap.max()  # sanity: the mask actually excludes the peak

    gui.update_xmap_limits()

    assert gui.doubleSpinBox_param_vend.value() == pytest.approx(unmasked_max)
    assert gui.doubleSpinBox_param_vend.maximum() == pytest.approx(unmasked_max)
    assert gui.doubleSpinBox_param_vbeg.minimum() == pytest.approx(
        xmap[gui.sm.mask].min()
    )


def test_update_xmap_limits_allows_phi_range_across_seam(qapp, tmp_path):
    """phi (deg) spin boxes must not be clamped to the map's min/max, so a
    pie that crosses ±180° (e.g. [170, 190]) can be entered; the default
    values and the info label still reflect the map's actual range."""
    gui = _load_gui(tmp_path, np.ones((2, 8, 8), dtype=np.uint16))
    # the default 8-ID-I center lies far outside this tiny frame; put the beam
    # in the middle so phi covers the full circle, including the ±180° seam
    gui.update_parameters(new_center_vh=np.array([3.5, 3.5]))
    gui.comboBox_param_xmap_name.addItem("phi")
    gui.comboBox_param_xmap_name.setCurrentText("phi")
    gui.update_xmap_limits()

    phi = gui.sm.qmap["phi"]
    assert gui.sm.qmap_unit["phi"] == "deg"
    assert gui.doubleSpinBox_param_vbeg.minimum() == pytest.approx(-360.0)
    assert gui.doubleSpinBox_param_vend.maximum() == pytest.approx(360.0)
    assert gui.doubleSpinBox_param_vbeg.value() == pytest.approx(phi.min())
    assert gui.doubleSpinBox_param_vend.value() == pytest.approx(phi.max())
    assert f"{phi.max():.7f}" in gui.label_param_info.text()

    # 190 used to be silently clamped to phi.max(); now the row keeps it, and
    # the core wraps it into a single 20° pie around ±180°.
    gui.doubleSpinBox_param_vbeg.setValue(170.0)
    gui.doubleSpinBox_param_vend.setValue(190.0)
    gui.add_param_constraint()
    assert gui.model._data[-1] == ["phi", gui.comboBox_param_logic.currentText(),
                                   "deg", 170.0, 190.0]

    gui.sm.mask_evaluate("mask_parameter", constraints=gui.model._data)
    got = gui.sm.mask_kernel.workers["mask_parameter"].get_mask().astype(bool)
    expected = (phi >= 170.0) | (phi <= -170.0)
    assert expected.any() and not expected.all()  # sanity: a real pie
    assert np.array_equal(got, expected)


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
        # Files (blemish section removed from UI) → mask_file only
        gui.MaskWidget.setCurrentIndex(_MaskTab.FILES)
        gui.mask_apply_current_tab()
        assert calls == ["mask_file"], f"Files: got {calls}"
        calls.clear()

        gui.MaskWidget.setCurrentIndex(_MaskTab.BINARY)
        gui.mask_apply_current_tab()
        assert calls == ["mask_threshold"], f"Binary: got {calls}"
        calls.clear()

        gui.MaskWidget.setCurrentIndex(_MaskTab.PARAMETER)
        gui.mask_apply_current_tab()
        assert calls == ["mask_parameter"], f"Parametrization: got {calls}"


def test_tab_mask_targets_constant(qapp, tmp_path):
    """_TAB_MASK_TARGETS has one entry per MaskWidget tab (6 total)."""
    from pysimplemask.gui.control.main_window import _TAB_MASK_TARGETS
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    assert len(_TAB_MASK_TARGETS) == gui.MaskWidget.count()


def test_named_indices_match_ui_order(qapp):
    """_MaskTab and _plot_index name the widgets' items as laid out in
    mask.ui, so reordering the .ui without updating them fails here."""
    gui = SimpleMaskGUI()
    tab_labels = {
        _MaskTab.FILES: "Blemish/Files",
        _MaskTab.DRAW: "Draw",
        _MaskTab.BINARY: "Binary",
        _MaskTab.MANUAL: "Manual",
        _MaskTab.OUTLIER: "Outlier",
        _MaskTab.PARAMETER: "Parametrization",
    }
    assert gui.MaskWidget.count() == len(_MaskTab)
    for tab, label in tab_labels.items():
        assert gui.MaskWidget.tabText(tab) == label

    # DISPLAY_FIELD name → plot_index label (the .ui spells partitions out).
    channel_labels = {
        "scattering": "scattering",
        "scattering * mask": "scattering * mask",
        "mask": "mask",
        "dqmap_partition": "dynamic_q_partition",
        "sqmap_partition": "static_q_partition",
        "preview": "preview",
    }
    assert set(channel_labels) == set(DISPLAY_FIELD)
    # the raw frames come first, as read and with the mask applied
    assert gui.plot_index.itemText(_RAWDATA_IDX) == "rawdata"
    assert gui.plot_index.itemText(_RAWDATA_MASK_IDX) == "rawdata * mask"
    assert gui.plot_index.count() == len(_RAW_PLOT_INDICES) + len(DISPLAY_FIELD)
    for field, label in channel_labels.items():
        assert gui.plot_index.itemText(_plot_index(field)) == label


def test_frame_range_presets_match_ui(qapp):
    """Every comboBox_frame_range item of mask.ui is a preset of
    _FRAME_RANGE_NUM_FRAMES or the custom range, and startup selects all
    the frames."""
    gui = SimpleMaskGUI()
    box = gui.comboBox_frame_range
    items = [box.itemText(n) for n in range(box.count())]
    assert items == [*_FRAME_RANGE_NUM_FRAMES, _FRAME_RANGE_CUSTOM]
    assert box.currentText() == "All frames"


def test_begin_and_frames_inputs_enabled_only_for_custom_range(qapp):
    gui = SimpleMaskGUI()
    inputs = (
        gui.label_begin_idx,
        gui.spinBox_begin_idx,
        gui.label_num_frames,
        gui.spinBox_num_frames,
    )
    for preset in _FRAME_RANGE_NUM_FRAMES:
        gui.comboBox_frame_range.setCurrentText(preset)
        assert not any(widget.isEnabled() for widget in inputs), preset
    gui.comboBox_frame_range.setCurrentText(_FRAME_RANGE_CUSTOM)
    assert all(widget.isEnabled() for widget in inputs)
    gui.comboBox_frame_range.setCurrentText("All frames")
    assert not any(widget.isEnabled() for widget in inputs)


def test_load_reads_the_selected_frame_range(qapp, tmp_path):
    """load() averages the frames picked in the Frame Range box."""
    # frame n is a flat image of value n: the mean tells which frames were read
    num_frames = 2400
    frames = np.ones((num_frames, 20, 24), dtype=np.uint16)
    frames *= np.arange(num_frames, dtype=np.uint16)[:, None, None]
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as h:
        h["/entry/data/data"] = frames
    gui = SimpleMaskGUI()
    gui.fname.setText(str(path))

    def loaded_mean(frame_range):
        gui.comboBox_frame_range.setCurrentText(frame_range)
        gui.load()
        scat = gui.sm.dset.scat
        assert scat.min() == scat.max()
        return float(scat[0, 0])

    assert loaded_mean("All frames") == np.arange(2400).mean()
    assert loaded_mean("1/2 of frames") == np.arange(1200).mean()
    # a fifth would be 480 frames, but a subset holds at least 1000
    assert loaded_mean("1/5 of frames") == np.arange(1000).mean()

    gui.comboBox_frame_range.setCurrentText(_FRAME_RANGE_CUSTOM)
    gui.spinBox_begin_idx.setValue(2)
    gui.spinBox_num_frames.setValue(3)
    assert loaded_mean(_FRAME_RANGE_CUSTOM) == np.arange(2, 5).mean()
    # loading leaves a custom range as it was typed in
    assert (gui.spinBox_begin_idx.value(), gui.spinBox_num_frames.value()) == (2, 3)

    gui.spinBox_begin_idx.setValue(2390)
    gui.spinBox_num_frames.setValue(0)
    assert gui.spinBox_num_frames.text() == "to end"
    assert loaded_mean(_FRAME_RANGE_CUSTOM) == np.arange(2390, 2400).mean()

    # back on a preset, the load starts at the first frame again
    assert loaded_mean("All frames") == np.arange(2400).mean()


def test_presets_show_the_frame_range_they_load(qapp, tmp_path):
    """The Begin / Frames inputs, disabled for a preset, show what it loads
    from the selected file once that file's frame count is known."""
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as h:
        h["/entry/data/data"] = np.zeros((2400, 4, 5), dtype=np.uint8)
    gui = SimpleMaskGUI()

    def shown(frame_range):
        gui.comboBox_frame_range.setCurrentText(frame_range)
        return gui.spinBox_begin_idx.text(), gui.spinBox_num_frames.text()

    # no file yet: all its frames is "to end", a share of them is not known
    assert shown("All frames") == ("0", "to end")
    assert shown("1/5 of frames") == ("0", "")
    assert shown("1/2 of frames") == ("0", "")

    # selecting a file updates the preset that is showing ...
    gui.fname.setText(str(path))
    assert gui.spinBox_num_frames.text() == "1200"
    assert not gui.spinBox_num_frames.isEnabled()
    # ... and so does switching between them
    assert shown("All frames") == ("0", "2400")
    assert shown("1/5 of frames") == ("0", "1000")   # a subset holds at least 1000
    assert shown("1/2 of frames") == ("0", "1200")

    # Custom starts from the range last shown, and keeps what is entered
    assert shown(_FRAME_RANGE_CUSTOM) == ("0", "1200")
    gui.spinBox_begin_idx.setValue(7)
    gui.spinBox_num_frames.setValue(30)
    gui.fname.setText(str(tmp_path / "missing.h5"))
    assert shown(_FRAME_RANGE_CUSTOM) == ("7", "30")

    # a file whose frame count is not known blanks the share again, and
    # Custom brings the number back
    assert shown("1/2 of frames") == ("0", "")
    assert shown(_FRAME_RANGE_CUSTOM) == ("0", "30")


def test_load_recounts_the_frames_of_a_file_that_grew(qapp, tmp_path):
    """The frame count is taken when the file is selected; a file still being
    written has more frames by the time it is loaded."""
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as h:
        h["/entry/data/data"] = np.ones((10, 20, 24), dtype=np.uint16)
    gui = SimpleMaskGUI()
    gui.fname.setText(str(path))
    assert gui.spinBox_num_frames.text() == "10"

    with h5py.File(path, "w") as h:
        h["/entry/data/data"] = np.ones((25, 20, 24), dtype=np.uint16)
    assert gui.spinBox_num_frames.text() == "10"
    gui.load()
    assert gui.spinBox_num_frames.text() == "25"


def test_rawdata_item_disabled_by_default(qapp, tmp_path):
    """The rawdata plot_index items are grayed out before any data is loaded."""
    gui = SimpleMaskGUI()
    model = gui.plot_index.model()
    for idx in _RAW_PLOT_INDICES:
        item = model.item(idx)
        assert item is not None
        assert not item.isEnabled(), "rawdata should be disabled before data load"


def test_frame_controls_hidden_by_default(qapp, tmp_path):
    """Frame controls are disabled on startup."""
    gui = SimpleMaskGUI()
    assert not gui.label_frame.isEnabled()
    assert not gui.horizontalSlider_frame.isEnabled()
    assert not gui.spinBox_current_frame.isEnabled()


def test_xy_label_keeps_fixed_screen_offset_from_cursor_at_any_zoom(qapp, tmp_path):
    """The floating (x, y) hover label sits a fixed number of *screen* pixels
    from the cursor whether the view is zoomed out or zoomed far in.
    Regression: the offset used to be in image-pixel units, so at high zoom
    the label drifted a large fraction of the viewport away from the cursor."""
    from pyqtgraph.Qt import QtCore

    from pysimplemask.gui.control.main_window import _XY_LABEL_OFFSET_PX

    gui = _load_gui(tmp_path, np.ones((3, 200, 300), dtype=np.uint16) * 5)
    gui.plot(reset_view=True)
    gui.resize(1000, 800)
    gui.show()
    qapp.processEvents()
    gui.checkBox_showxy.setChecked(True)
    vb = gui.mp1.getView()

    # Whole image, then a ~50x zoom onto a handful of pixels.
    for xr, yr in [((0, 300), (0, 200)), ((100, 106), (50, 54))]:
        vb.setRange(xRange=xr, yRange=yr, padding=0)
        qapp.processEvents()
        cursor_view = QtCore.QPointF(sum(xr) / 2 + 0.3, sum(yr) / 2 + 0.7)
        cursor_scene = vb.mapViewToScene(cursor_view)
        gui.show_location(cursor_scene)

        label = gui._xy_text_item
        assert label is not None, f"label missing at range {xr}, {yr}"
        delta = vb.mapViewToScene(label.pos()) - cursor_scene
        assert abs(delta.x() - _XY_LABEL_OFFSET_PX) < 1e-6, (xr, yr, delta)
        assert abs(delta.y() - _XY_LABEL_OFFSET_PX) < 1e-6, (xr, yr, delta)
        col, row = int(cursor_view.x()), int(cursor_view.y())
        assert label.textItem.toPlainText() == f"({col}, {row})"


def test_raw_file_dialog_hides_metadata_sidecars(qapp, tmp_path):
    """The "Select raw file" proxy hides *_metadata.hdf files (they are the
    metadata sidecar, auto-discovered by the reader, never the data to load)
    but keeps every other file and every directory, even one whose name ends
    with the suffix, so the user can still navigate."""
    import time

    from PySide6.QtCore import QDir
    from PySide6.QtWidgets import QFileSystemModel

    from pysimplemask.gui.control.main_window import _HideMetadataFilesProxyModel

    (tmp_path / "scan_0001.hdf").write_bytes(b"")
    (tmp_path / "scan_metadata.hdf").write_bytes(b"")
    (tmp_path / "other.h5").write_bytes(b"")
    (tmp_path / "sub_metadata.hdf").mkdir()

    src = QFileSystemModel()
    src.setFilter(QDir.Filter.AllEntries | QDir.Filter.NoDotAndDotDot)
    proxy = _HideMetadataFilesProxyModel()
    proxy.setSourceModel(src)
    root = src.setRootPath(str(tmp_path))
    if src.canFetchMore(root):
        src.fetchMore(root)
    # QFileSystemModel populates from a worker thread; pump events until all 4 entries land.
    deadline = time.monotonic() + 15
    while src.rowCount(root) < 4 and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.02)
    assert src.rowCount(root) == 4, "QFileSystemModel did not populate"

    proot = proxy.mapFromSource(root)
    visible = sorted(proxy.data(proxy.index(r, 0, proot)) for r in range(proxy.rowCount(proot)))
    assert visible == ["other.h5", "scan_0001.hdf", "sub_metadata.hdf"]


def test_raw_file_dialog_is_configured_with_metadata_filter(qapp):
    """select_raw() uses Qt's own dialog (native ones ignore proxy models)
    with the metadata-hiding proxy, picking a single existing file."""
    from PySide6.QtWidgets import QFileDialog

    from pysimplemask.gui.control.main_window import _HideMetadataFilesProxyModel

    gui = SimpleMaskGUI()
    dlg = gui._raw_file_dialog()
    assert dlg.testOption(QFileDialog.Option.DontUseNativeDialog)
    assert dlg.fileMode() == QFileDialog.FileMode.ExistingFile
    assert isinstance(dlg.proxyModel(), _HideMetadataFilesProxyModel)
    assert dlg.nameFilters() and "*.hdf" in dlg.nameFilters()[0]
    dlg.deleteLater()


def _hot_pixel_frames(shape=(3, 24, 32), base=10.0, hot=((4, 4), (16, 24))):
    """Flat frames with hot pixels at the given (row, col) spots."""
    frames = np.full(shape, base, dtype=np.float32)
    for r, c in hot:
        frames[:, r, c] = 10000.0
    return frames


def test_outlier_roi_button_follows_target(qapp, tmp_path):
    """The Outlier tab's Draw ROI button is only live for AdjacentPixels."""
    gui = _load_gui(tmp_path, _hot_pixel_frames())
    gui.comboBox_outlier_target.setCurrentText("CircularRings")
    assert not gui.btn_outlier_roi.isEnabled()
    gui.comboBox_outlier_target.setCurrentText("AdjacentPixels")
    assert gui.btn_outlier_roi.isEnabled()
    assert gui.btn_outlier_roi.text() == "Draw ROI"


def test_outlier_roi_restricts_adjacent_removal_and_is_consumed_by_apply(qapp, tmp_path):
    """AdjacentPixels + a drawn rectangle: only outliers inside the rectangle
    are flagged; the rectangle survives the evaluate redraw and is removed by
    Apply (with the button text following)."""
    from pysimplemask.gui.view.widgets import OUTLIER_ROI_KEY

    gui = _load_gui(tmp_path, _hot_pixel_frames())
    gui.plot()
    gui.comboBox_outlier_target.setCurrentText("AdjacentPixels")
    gui.comboBox_outlier_method.setCurrentText("percentile")
    gui.outlier_num_roi.setValue(8)
    gui.outlier_cutoff.setValue(3.0)

    gui.toggle_outlier_roi()
    roi = gui.mp1.roi[OUTLIER_ROI_KEY]
    assert gui.btn_outlier_roi.text() == "Remove ROI"
    # Cover the lower-right quadrant: x (col) 16..32, y (row) 12..24.
    roi.setPos((16, 12))
    roi.setSize((16, 12))
    region = gui._outlier_roi_region()
    assert region.shape == (24, 32)
    assert region[16, 24] and not region[4, 4]

    gui.mask_evaluate("mask_outlier")
    zero_loc = gui.sm.mask_kernel.workers["mask_outlier"].zero_loc
    locs = set(zip(zero_loc[0].tolist(), zero_loc[1].tolist()))
    assert (16, 24) in locs
    assert (4, 4) not in locs
    # evaluate switches to the preview channel (a redraw): the rectangle stays.
    assert OUTLIER_ROI_KEY in gui.mp1.roi

    gui.mask_apply("mask_outlier")
    assert OUTLIER_ROI_KEY not in gui.mp1.roi
    assert gui.btn_outlier_roi.text() == "Draw ROI"
    # Only the hot pixel inside the rectangle was masked.
    assert not gui.sm.mask[16, 24]
    assert gui.sm.mask[4, 4]


def test_outlier_without_roi_still_scans_whole_image(qapp, tmp_path):
    gui = _load_gui(tmp_path, _hot_pixel_frames())
    gui.comboBox_outlier_target.setCurrentText("AdjacentPixels")
    gui.comboBox_outlier_method.setCurrentText("percentile")
    gui.outlier_num_roi.setValue(8)
    gui.outlier_cutoff.setValue(3.0)
    assert gui._outlier_roi_region() is None
    gui.mask_evaluate("mask_outlier")
    zero_loc = gui.sm.mask_kernel.workers["mask_outlier"].zero_loc
    locs = set(zip(zero_loc[0].tolist(), zero_loc[1].tolist()))
    assert {(4, 4), (16, 24)} <= locs


def test_reevaluate_redraws_preview_when_already_selected(qapp, tmp_path):
    """Evaluating again while "preview" is already shown must redraw it.
    Regression: setCurrentIndex(preview) on the current index emits no
    currentIndexChanged, so the canvas kept the stale preview until the user
    switched the display target away and back."""
    gui = _load_gui(tmp_path, _hot_pixel_frames())
    gui.plot()
    gui.comboBox_outlier_target.setCurrentText("AdjacentPixels")
    gui.outlier_num_roi.setValue(8)
    gui.outlier_cutoff.setValue(3.0)
    gui.mask_evaluate("mask_outlier")
    assert gui.plot_index.currentIndex() == _plot_index("preview")

    gui.outlier_cutoff.setValue(5.0)
    with patch.object(gui.mp1, "setImage", wraps=gui.mp1.setImage) as spy:
        gui.mask_evaluate("mask_outlier")
    spy.assert_called_once()
    shown = spy.call_args.args[0]
    assert np.array_equal(shown, gui.sm.dset.data_display[DISPLAY_FIELD.index("preview")])


def test_outlier_roi_removed_when_target_switches_to_rings(qapp, tmp_path):
    from pysimplemask.gui.view.widgets import OUTLIER_ROI_KEY

    gui = _load_gui(tmp_path, _hot_pixel_frames())
    gui.plot()
    gui.comboBox_outlier_target.setCurrentText("AdjacentPixels")
    gui.toggle_outlier_roi()
    assert OUTLIER_ROI_KEY in gui.mp1.roi
    # Toggling again removes it; drawing once more brings it back.
    gui.toggle_outlier_roi()
    assert OUTLIER_ROI_KEY not in gui.mp1.roi
    gui.toggle_outlier_roi()
    assert OUTLIER_ROI_KEY in gui.mp1.roi
    gui.comboBox_outlier_target.setCurrentText("CircularRings")
    assert OUTLIER_ROI_KEY not in gui.mp1.roi
    assert not gui.btn_outlier_roi.isEnabled()


def test_outlier_roi_is_not_a_draw_roi(qapp, tmp_path):
    """The Outlier rectangle must not leak into the Draw tab's mask."""
    from pysimplemask.gui.model.roi_extract import extract_roi_geometry
    from pysimplemask.gui.view.widgets import OUTLIER_ROI_KEY

    gui = _load_gui(tmp_path, _hot_pixel_frames())
    gui.plot()
    gui.comboBox_outlier_target.setCurrentText("AdjacentPixels")
    gui.toggle_outlier_roi()
    assert OUTLIER_ROI_KEY in gui.mp1.roi
    assert extract_roi_geometry(gui.mp1.roi, gui.mp1.imageItem) == []


def test_manual_include_similar_intensity_unchecked_by_default(qapp):
    """Manual tab: 'Include points with similar intensity' starts unchecked, and
    its Radius / Variation controls start disabled and follow the checkbox."""
    gui = SimpleMaskGUI()
    assert not gui.mask_list_include.isChecked()
    assert not gui.mask_list_radius.isEnabled()
    assert not gui.mask_list_variation.isEnabled()
    gui.mask_list_include.setChecked(True)
    assert gui.mask_list_radius.isEnabled()
    assert gui.mask_list_variation.isEnabled()


def test_non_rawdata_channel_uses_correct_data_display_slice(qapp, tmp_path):
    """plot_index=1 (scattering) shows data_display[0], not data_display[1]."""
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16) * 5)
    # Navigate away then back to scattering to ensure the signal fires
    # (combobox skips no-op setCurrentIndex)
    gui.plot_index.setCurrentIndex(_plot_index("scattering * mask"))
    gui.plot_index.setCurrentIndex(_plot_index("scattering"))
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
    """The rawdata combobox items are enabled after loading a multi-frame HDF."""
    gui = _load_gui(tmp_path, np.ones((5, 20, 24), dtype=np.uint16))
    model = gui.plot_index.model()
    for idx in _RAW_PLOT_INDICES:
        assert model.item(idx).isEnabled(), "rawdata should be enabled for multi-frame HDF"


def test_rawdata_disabled_after_loading_single_frame_hdf(qapp, tmp_path):
    """The rawdata combobox items are disabled after loading a single-frame HDF."""
    gui = _load_gui(tmp_path, np.ones((1, 20, 24), dtype=np.uint16))
    model = gui.plot_index.model()
    for idx in _RAW_PLOT_INDICES:
        assert not model.item(idx).isEnabled()


# A Rigaku frame is a list of photon events. In these, frame n puts
# (n % 7) + 1 counts into pixel n % 5 of the module's first row.
_RIGAKU_NUM_FRAMES = 250
_RIGAKU_MODULE_SHAPE = (512, 1024)


def _write_rigaku_module(path, num_frames=_RIGAKU_NUM_FRAMES):
    """Write a Rigaku 500k module file of 64-bit event words
    ``[frame:24][pixel index:21]...[count:12]``."""
    frame = np.arange(num_frames, dtype=np.uint64)
    words = (
        (frame << np.uint64(40))
        | ((frame % np.uint64(5)) << np.uint64(16))
        | (frame % np.uint64(7) + np.uint64(1))
    )
    words.tofile(path)


def _rigaku_module_sum(begin_idx, end_idx):
    """Sum of the frames ``[begin_idx, end_idx)`` of that module."""
    image = np.zeros(_RIGAKU_MODULE_SHAPE)
    for n in range(begin_idx, end_idx):
        image[0, n % 5] += n % 7 + 1
    return image


def _load_file(gui, path):
    gui.fname.setText(str(path))
    gui.load()
    assert gui.sm.is_ready()


def _show_rawdata_frame(gui, idx):
    """Step the frame browser to ``idx`` and read it now, rather than when
    the debounce timer fires."""
    gui.spinBox_current_frame.setValue(idx)
    gui._frame_timer.stop()
    gui._read_and_show_frame()


def test_rawdata_browses_rigaku_500k_frames(qapp, tmp_path):
    """A Rigaku 500k file has rawdata, browsed 100 frames at a time by default."""
    path = tmp_path / "scan.bin"
    _write_rigaku_module(path)
    gui = SimpleMaskGUI()
    gui.plot_log.setChecked(False)
    _load_file(gui, path)

    assert gui.plot_index.model().item(_RAWDATA_IDX).isEnabled()
    assert gui.spinBox_frame_average.value() == 100
    # 250 frames are two whole windows of 100
    assert gui.horizontalSlider_frame.maximum() == 1
    assert gui.spinBox_current_frame.maximum() == 1

    gui.plot_index.setCurrentIndex(_RAWDATA_IDX)
    assert gui.spinBox_frame_average.isEnabled()
    assert gui.mp1.image.shape == _RIGAKU_MODULE_SHAPE
    assert np.allclose(gui.mp1.image, _rigaku_module_sum(0, 100) / 100)
    _show_rawdata_frame(gui, 1)
    assert gui.horizontalSlider_frame.value() == 1
    assert np.allclose(gui.mp1.image, _rigaku_module_sum(100, 200) / 100)

    # a narrower window gives more steps, and stays on step 1
    gui.spinBox_frame_average.setValue(10)
    assert gui.horizontalSlider_frame.maximum() == 24
    assert np.allclose(gui.mp1.image, _rigaku_module_sum(10, 20) / 10)
    _show_rawdata_frame(gui, 24)
    assert np.allclose(gui.mp1.image, _rigaku_module_sum(240, 250) / 10)

    # single frames, and a window wider than the file
    gui.spinBox_frame_average.setValue(1)
    _show_rawdata_frame(gui, 249)
    assert np.allclose(gui.mp1.image, _rigaku_module_sum(249, 250))
    gui.spinBox_frame_average.setValue(1000)
    assert gui.horizontalSlider_frame.maximum() == 0
    assert np.allclose(gui.mp1.image, _rigaku_module_sum(0, 250) / 250)


def test_rawdata_browses_rigaku_extend_frames(qapp, tmp_path):
    """A RigakuExtend (.bix) file has rawdata too, with the same default window."""
    # as _write_rigaku_module, in the RigakuExtend layout of the event words:
    # [frame:32][pixel index:20][count:12]
    frame = np.arange(_RIGAKU_NUM_FRAMES, dtype=np.uint64)
    words = (
        (frame << np.uint64(32))
        | ((frame % np.uint64(5)) << np.uint64(12))
        | (frame % np.uint64(7) + np.uint64(1))
    )
    path = tmp_path / "scan.bix"
    words.tofile(path)
    gui = SimpleMaskGUI()
    gui.plot_log.setChecked(False)
    _load_file(gui, path)

    assert gui.plot_index.model().item(_RAWDATA_IDX).isEnabled()
    assert gui.spinBox_frame_average.value() == 100
    assert gui.horizontalSlider_frame.maximum() == 1

    gui.plot_index.setCurrentIndex(_RAWDATA_IDX)
    _show_rawdata_frame(gui, 1)
    # the same events as the 500k module, on the 448 columns of this one
    assert gui.mp1.image.shape == (512, 448)
    assert np.allclose(gui.mp1.image, _rigaku_module_sum(100, 200)[:, :448] / 100)


def test_rawdata_shows_rigaku_frames_in_log_scale(qapp, tmp_path):
    path = tmp_path / "scan.bin"
    _write_rigaku_module(path)
    gui = SimpleMaskGUI()
    gui.plot_log.setChecked(True)
    _load_file(gui, path)
    gui.plot_index.setCurrentIndex(_RAWDATA_IDX)

    mean = _rigaku_module_sum(0, 100) / 100
    # the pixels without a count show as the faintest one that has any
    expected = np.log10(np.maximum(mean, mean[mean > 0].min()))
    assert np.allclose(gui.mp1.image, expected)


def test_rawdata_browses_rigaku_3m_frames(qapp, tmp_path):
    """A Rigaku 3M dataset (six module files) has rawdata as well, stitched."""
    # module file N holds 10 * N frames fewer than the first
    module_frames = {n: _RIGAKU_NUM_FRAMES - 10 * n for n in range(6)}
    for suffix, num_frames in module_frames.items():
        _write_rigaku_module(tmp_path / f"scan.bin.00{suffix}", num_frames)
    gui = SimpleMaskGUI()
    gui.plot_log.setChecked(False)
    _load_file(gui, tmp_path / "scan.bin.000")

    assert gui.plot_index.model().item(_RAWDATA_IDX).isEnabled()
    assert gui.spinBox_frame_average.value() == 100
    # the dataset is as long as its longest module file
    assert gui.horizontalSlider_frame.maximum() == 1

    gui.plot_index.setCurrentIndex(_RAWDATA_IDX)
    gui.spinBox_frame_average.setValue(50)
    _show_rawdata_frame(gui, 4)  # frames 200-249
    image = gui.mp1.image
    assert image.shape == (1676, 2100)
    # Module files (5, 0, 4, 1, 3, 2) fill the 3x2 layout row by row; block
    # (r, c) starts at row r * (512 + 70), column c * (1024 + 52).
    filled = np.zeros(image.shape, dtype=bool)
    for block, suffix in enumerate((5, 0, 4, 1, 3, 2)):
        top, left = (block // 2) * 582, (block % 2) * 1076
        where = (slice(top, top + 512), slice(left, left + 1024))
        # a module file that ends within the window adds nothing after its end
        expected = _rigaku_module_sum(200, max(200, module_frames[suffix])) / 50
        assert np.allclose(image[where], expected), suffix
        filled[where] = True
    assert image[0, 1076:1081].any()  # module file 0, all 50 frames
    assert not image[0, 0:5].any()  # module file 5 ends at frame 200
    assert not image[~filled].any()  # the gaps between the modules


def _load_frames_with_blemish(tmp_path, monkeypatch, frames, blemish):
    """Load an HDF stack of ``frames`` into a new GUI, on a detector whose
    default blemish is ``blemish``."""
    monkeypatch.setattr(
        MaskAssemble, "get_default_blemish", lambda self: blemish.copy()
    )
    path = tmp_path / "scan.h5"
    with h5py.File(path, "w") as h:
        h["/entry/data/data"] = frames
    gui = SimpleMaskGUI()
    _load_file(gui, path)
    assert np.array_equal(gui.sm.mask_kernel.blemish, blemish)
    return gui


def _frames_with_a_hot_pixel():
    """Six frames of 1-49 counts, with a hot pixel at (5, 5)."""
    rng = np.random.default_rng(1)
    frames = rng.integers(1, 50, size=(6, 20, 24)).astype(np.int32)
    frames[:, 5, 5] = 10**6
    return frames


def test_rawdata_times_mask_shows_frames_with_the_current_mask(
    qapp, tmp_path, monkeypatch
):
    """ "rawdata * mask" browses the frames like "rawdata", with the current
    mask applied: the blemish until a mask is applied on top of it."""
    frames = _frames_with_a_hot_pixel()
    blemish = np.ones(frames.shape[1:], dtype=bool)
    blemish[0, :3] = False
    gui = _load_frames_with_blemish(tmp_path, monkeypatch, frames, blemish)
    gui.plot_log.setChecked(False)

    # no mask applied yet: the blemish
    gui.plot_index.setCurrentIndex(_RAWDATA_MASK_IDX)
    for widget in (
        gui.label_frame,
        gui.horizontalSlider_frame,
        gui.spinBox_current_frame,
        gui.spinBox_frame_average,
        gui.label_average,
    ):
        assert widget.isEnabled()
    _show_rawdata_frame(gui, 2)
    assert np.array_equal(gui.mp1.image, frames[2] * blemish)
    assert not gui.mp1.image[0, :3].any()
    assert gui.mp1.image[5, 5] == 10**6
    # ... also for an average of frames
    gui.spinBox_frame_average.setValue(2)
    assert np.array_equal(gui.mp1.image, frames[4:6].mean(axis=0) * blemish)
    gui.spinBox_frame_average.setValue(1)

    # "rawdata" shows the same frame as it was read
    gui.plot_index.setCurrentIndex(_RAWDATA_IDX)
    assert np.array_equal(gui.mp1.image, frames[2])

    # a mask applied on top of the blemish shows as well
    gui.sm.mask_evaluate(
        "mask_threshold", low=0, high=1000, low_enable=True, high_enable=True
    )
    gui.mask_apply("mask_threshold")
    assert gui.plot_index.currentIndex() == _plot_index("scattering * mask")
    mask = blemish.copy()
    mask[5, 5] = False
    assert np.array_equal(gui.sm.mask, mask)
    gui.plot_index.setCurrentIndex(_RAWDATA_MASK_IDX)
    assert np.array_equal(gui.mp1.image, frames[2] * mask)
    assert gui.mp1.image[5, 5] == 0

    # undoing it leaves the blemish again
    gui.mask_action("undo")
    gui.plot_index.setCurrentIndex(_RAWDATA_MASK_IDX)
    assert np.array_equal(gui.mp1.image, frames[2] * blemish)


def test_rawdata_times_mask_in_log_scale(qapp, tmp_path, monkeypatch):
    """Masked pixels show as the faintest of the pixels that are left."""
    frames = _frames_with_a_hot_pixel()
    blemish = np.ones(frames.shape[1:], dtype=bool)
    blemish[5, 5] = False  # the hot pixel
    blemish[0, :3] = False
    gui = _load_frames_with_blemish(tmp_path, monkeypatch, frames, blemish)
    gui.plot_log.setChecked(True)

    gui.plot_index.setCurrentIndex(_RAWDATA_MASK_IDX)
    _show_rawdata_frame(gui, 3)
    kept = frames[3][blemish]
    expected = np.where(blemish, np.log10(frames[3]), np.log10(kept.min()))
    assert np.allclose(gui.mp1.image, expected)
    assert np.isclose(gui.mp1.image.max(), np.log10(kept.max()))  # not the hot pixel's


def test_rawdata_times_mask_browses_rigaku_frames(qapp, tmp_path, monkeypatch):
    blemish = np.ones(_RIGAKU_MODULE_SHAPE, dtype=bool)
    blemish[0, 1:3] = False  # two of the five pixels the frames hit
    monkeypatch.setattr(
        MaskAssemble, "get_default_blemish", lambda self: blemish.copy()
    )
    path = tmp_path / "scan.bin"
    _write_rigaku_module(path)
    gui = SimpleMaskGUI()
    gui.plot_log.setChecked(False)
    _load_file(gui, path)

    assert gui.plot_index.model().item(_RAWDATA_MASK_IDX).isEnabled()
    gui.plot_index.setCurrentIndex(_RAWDATA_MASK_IDX)
    _show_rawdata_frame(gui, 1)
    assert gui.spinBox_frame_average.value() == 100
    raw = _rigaku_module_sum(100, 200) / 100
    assert np.allclose(gui.mp1.image, raw * blemish)
    assert not gui.mp1.image[0, 1:3].any()
    assert gui.mp1.image[0, 0] > 0 and raw[0, 1] > 0


def test_frame_average_default_follows_the_loaded_format(qapp, tmp_path):
    """Each load sets the average window to the default of its format: 100
    frames for Rigaku, single frames for HDF."""
    rigaku_path = tmp_path / "scan.bin"
    _write_rigaku_module(rigaku_path)
    hdf_path = tmp_path / "scan.h5"
    frames = np.ones((6, 20, 24), dtype=np.uint16)
    frames *= np.arange(6, dtype=np.uint16)[:, None, None]
    with h5py.File(hdf_path, "w") as h:
        h["/entry/data/data"] = frames
    gui = SimpleMaskGUI()
    gui.plot_log.setChecked(False)

    _load_file(gui, hdf_path)
    assert gui.spinBox_frame_average.value() == 1
    assert gui.horizontalSlider_frame.maximum() == 5

    # load shows the averaged scattering: changing the window to the default
    # must not make it read raw frames as well
    with patch.object(FileReader, "get_frames_mean") as get_frames_mean:
        _load_file(gui, rigaku_path)
    get_frames_mean.assert_not_called()
    assert gui.spinBox_frame_average.value() == 100
    assert gui.horizontalSlider_frame.maximum() == 1
    assert gui.plot_index.currentIndex() == _plot_index("scattering * mask")

    _load_file(gui, hdf_path)
    assert gui.spinBox_frame_average.value() == 1
    assert gui.horizontalSlider_frame.maximum() == 5
    gui.plot_index.setCurrentIndex(_RAWDATA_IDX)
    _show_rawdata_frame(gui, 4)
    assert np.array_equal(gui.mp1.image, frames[4])

    # a window the user changed is back at the default after the next load
    gui.spinBox_frame_average.setValue(3)
    assert np.array_equal(gui.mp1.image, frames[3:6].mean(axis=0))
    _load_file(gui, hdf_path)
    assert gui.spinBox_frame_average.value() == 1


def test_rawdata_disabled_after_loading_single_frame_rigaku(qapp, tmp_path):
    """A Rigaku file of a single frame has nothing to browse."""
    path = tmp_path / "scan.bin"
    _write_rigaku_module(path, num_frames=1)
    gui = SimpleMaskGUI()
    _load_file(gui, path)
    assert gui._detect_rawdata() == (False, 0)
    for idx in _RAW_PLOT_INDICES:
        assert not gui.plot_index.model().item(idx).isEnabled()


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
    gui.MaskWidget.setCurrentIndex(_MaskTab.BINARY)  # -> mask_threshold
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
    gui.MaskWidget.setCurrentIndex(_MaskTab.BINARY)
    with patch.object(gui, "_maybe_prompt_metadata_update") as mock_prompt:
        gui.mask_evaluate_current_tab()
    mock_prompt.assert_called_once()


def test_mask_apply_current_tab_checks_metadata_staleness(qapp, tmp_path):
    """mask_apply_current_tab calls the staleness check."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.MaskWidget.setCurrentIndex(_MaskTab.BINARY)
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
    gui.MaskWidget.setCurrentIndex(_MaskTab.BINARY)  # -> mask_threshold
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
    gui.plot_index.setCurrentIndex(_plot_index("dqmap_partition"))
    image_before = gui.mp1.image

    gui.compute_partition()

    assert gui.mp1.image is not image_before
    np.testing.assert_array_equal(
        gui.mp1.image, gui.sm.dset.data_display[DISPLAY_FIELD.index("dqmap_partition")]
    )


def test_update_parameters_refreshes_stale_dynamic_partition_display(qapp, tmp_path):
    """Clicking "Update Parameters" (recompute the qmap) while the dynamic
    partition is on screen must refresh it too, not just the qmap geometry
    channels — otherwise the displayed dynamic/static partition silently goes
    stale relative to the new geometry."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    gui.compute_partition()
    gui.plot_index.setCurrentIndex(_plot_index("dqmap_partition"))

    with patch.object(
        gui.sm, "compute_partition", wraps=gui.sm.compute_partition
    ) as spy:
        gui.update_parameters(new_center_vh=(8.0, 6.0))

    spy.assert_called_once()
    np.testing.assert_array_equal(
        gui.mp1.image, gui.sm.dset.data_display[DISPLAY_FIELD.index("dqmap_partition")]
    )


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
    gui.MaskWidget.setCurrentIndex(_MaskTab.DRAW)
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
    gui.mp1.remove_rois()
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


def test_loading_a_new_file_removes_drawn_rois(draw_gui, tmp_path):
    """Drawn ROIs belong to the dataset they were drawn on: they must not
    survive load(), or the next Draw evaluate rasterizes them against the
    new dataset's geometry."""
    gui = draw_gui
    _select_draw(gui, "Circle", "inclusive")
    gui.add_drawing()
    assert gui.model_draw.rowCount() == 1

    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as h:
        h["/entry/data/data"] = np.ones((3, 20, 24), dtype=np.uint16)
    gui.fname.setText(str(path))
    gui.load()

    assert not [key for key in gui.mp1.roi if key.startswith("roi_")]
    assert gui.model_draw.rowCount() == 0
    assert gui.sm.draw_rois == []


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
    gui.MaskWidget.setCurrentIndex(_MaskTab.DRAW)
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

    gui.MaskWidget.setCurrentIndex(_MaskTab.DRAW)
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
    gui.MaskWidget.setCurrentIndex(_MaskTab.DRAW)
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


def test_file_dialogs_always_open_in_launch_dir(qapp, tmp_path):
    """Every browse dialog starts in the launch directory, even after the user
    picked a file from somewhere else in an earlier browse."""
    launch_dir = tmp_path / "launch"
    other_dir = tmp_path / "elsewhere"
    launch_dir.mkdir()
    other_dir.mkdir()
    # A valid (empty) constraints file so load_param_constraints succeeds.
    chosen = other_dir / "constraints.json"
    chosen.write_text("[]")

    gui = SimpleMaskGUI(path=str(launch_dir))
    assert gui.work_dir == str(launch_dir)

    seen_dirs = []

    def fake_open(parent, caption="", dir="", filter="", *args, **kwargs):
        seen_dirs.append(dir)
        return (str(chosen), "")

    def fake_save(parent, caption="", dir="", filter="", *args, **kwargs):
        seen_dirs.append(dir)
        return ("", "")

    # select_raw() builds a QFileDialog instance (so it can hide *_metadata.hdf
    # via a proxy model) instead of the static getOpenFileName: stand in for
    # its modal exec() and record where that dialog starts.
    def fake_exec(dlg):
        seen_dirs.append(dlg.directory().absolutePath())
        return 1   # QDialog.DialogCode.Accepted

    with patch.object(QFileDialog, "getOpenFileName", side_effect=fake_open), \
         patch.object(QFileDialog, "getSaveFileName", side_effect=fake_save), \
         patch.object(QFileDialog, "exec", fake_exec), \
         patch.object(QFileDialog, "selectedFiles", lambda dlg: [str(chosen)]):
        gui.select_raw()
        assert gui.fname.text() == str(chosen)
        # Browsing to another folder must not move the remembered directory.
        assert gui.work_dir == str(launch_dir)
        gui.select_raw()
        gui.save_param_constraints()
        gui.load_param_constraints()

    assert seen_dirs, "no dialog was invoked"
    assert all(d == str(launch_dir) for d in seen_dirs), seen_dirs


def test_work_dir_defaults_to_cwd_when_no_path(qapp, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    gui = SimpleMaskGUI()
    assert gui.work_dir == str(tmp_path)
    assert gui._dialog_dir() == str(tmp_path)
