import json
import logging
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter
from PySide6.QtCore import QByteArray
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHeaderView,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QRadioButton,
    QSpinBox,
    QTabWidget,
)

from pyqtgraph.Qt import QtCore

from pysimplemask import __version__
from pysimplemask.core.io import load_pixel_list, text_to_array
from pysimplemask.core.model import SimpleMaskModel
from pysimplemask.core.partition import least_multiple
from pysimplemask.gui.model.roi_extract import extract_roi_geometry
from pysimplemask.gui.model.table_model import XmapConstraintsTableModel
from pysimplemask.gui.view.ui_mask import Ui_SimpleMask as Ui

HOME_DIR = Path.home()
CONFIG_FILE = HOME_DIR / ".pysimplemask" / "config.json"
CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s][%(module)s]: %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def exception_hook(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = exception_hook


PVMAP = {
    "beam_center_x": "db_cenx",
    "beam_center_y": "db_ceny",
    "energy": "db_energy",
    "pixel_size": "db_pix_dim",
    "detector_distance": "db_det_dist",
}


class SimpleMaskGUI(QMainWindow, Ui):
    def __init__(self, path=None):
        super(SimpleMaskGUI, self).__init__()

        self.setupUi(self)
        self.setWindowTitle(f"pySimpleMask {__version__}")
        
        # Set application icon
        try:
            from PySide6.QtGui import QIcon
            icon_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "resources", "logo.svg"
            )
            if os.path.exists(icon_path):
                self.setWindowIcon(QIcon(icon_path))
        except Exception:
            pass
            
        self.btn_load.clicked.connect(self.load)
        self.btn_plot.clicked.connect(self.plot)
        self.btn_compute_qpartition.clicked.connect(self.compute_partition)
        self.btn_compute_qpartition2.clicked.connect(self.compute_partition)
        self.btn_compute_qpartition3.clicked.connect(self.compute_partition)
        self.btn_compute_qpartition4.clicked.connect(self.compute_partition)

        self.btn_select_raw.clicked.connect(self.select_raw)
        # self.btn_select_txt.clicked.connect(self.select_txt)
        self.btn_update_parameters.clicked.connect(self.update_parameters)
        self.btn_swapxy.clicked.connect(lambda: self.update_parameters(swapxy=True))
        self.pushButton_goto_max.clicked.connect(self.goto_max)

        self.btn_find_center.clicked.connect(self.find_center)

        # need a function for save button -- simple_mask_ui
        self.pushButton.clicked.connect(self.save_mask)

        self.plot_index.currentIndexChanged.connect(self.mp1.setCurrentIndex)

        # reset, redo, undo botton for mask
        self.btn_mask_reset.clicked.connect(lambda: self.mask_action("reset"))
        self.btn_mask_redo.clicked.connect(lambda: self.mask_action("redo"))
        self.btn_mask_undo.clicked.connect(lambda: self.mask_action("undo"))

        # headless core model + view-side mouse hover
        self.sm = SimpleMaskModel()
        self.mp1.scene.sigMouseMoved.connect(self.show_location)
        self.metadata_parameter = None
        self.mp1.sigTimeChanged.connect(self.update_index)
        self.state = "lock"
        self._xy_text_item = None   # pg.TextItem shown next to cursor when checkBox_showxy is on

        # mask_list
        self.btn_mask_list_load.clicked.connect(self.mask_list_load)
        self.btn_mask_list_clear.clicked.connect(self.mask_list_clear)
        self.btn_mask_list_add.clicked.connect(self.mask_list_add)

        self.btn_mask_list_evaluate.clicked.connect(
            lambda: self.mask_evaluate("mask_list")
        )
        self.btn_mask_list_apply.clicked.connect(lambda: self.mask_apply("mask_list"))

        # blemish
        self.btn_select_blemish.clicked.connect(self.select_blemish)
        self.btn_apply_blemish.clicked.connect(
            lambda: self.mask_evaluate("mask_blemish")
        )
        self.btn_mask_blemish_apply.clicked.connect(
            lambda: self.mask_apply("mask_blemish")
        )

        # mask_file
        self.btn_select_maskfile.clicked.connect(self.select_maskfile)
        self.btn_apply_maskfile.clicked.connect(lambda: self.mask_evaluate("mask_file"))
        self.btn_mask_file_apply.clicked.connect(lambda: self.mask_apply("mask_file"))

        # draw method / array
        self.btn_mask_draw_add.clicked.connect(self.add_drawing)
        self.btn_mask_draw_evaluate.clicked.connect(
            lambda: self.mask_evaluate("mask_draw")
        )
        self.btn_mask_draw_apply.clicked.connect(lambda: self.mask_apply("mask_draw"))

        # binary threshold
        self.btn_mask_threshold_evaluate.clicked.connect(
            lambda: self.mask_evaluate("mask_threshold")
        )
        self.btn_mask_threshold_apply.clicked.connect(
            lambda: self.mask_apply("mask_threshold")
        )
        self.checkBox_threshold_low_preset.currentIndexChanged.connect(
            lambda: self.update_threshold_preset("low")
        )
        self.checkBox_threshold_high_preset.currentIndexChanged.connect(
            lambda: self.update_threshold_preset("high")
        )

        # btn_mask_outlier_evaluate
        self.btn_mask_outlier_evaluate.clicked.connect(
            lambda: self.mask_evaluate("mask_outlier")
        )
        self.btn_mask_outlier_apply.clicked.connect(
            lambda: self.mask_apply("mask_outlier")
        )

        self.mask_outlier_hdl.setBackground((255, 255, 255))
        self.comboBox_outlier_target.currentIndexChanged.connect(
            self._on_outlier_target_changed
        )
        self.comboBox_outlier_method.currentIndexChanged.connect(
            self._update_outlier_info
        )
        self._on_outlier_target_changed()  # sync label/default for the initial selection
        self.comboBox_outlier_target.currentIndexChanged.connect(
            self._update_outlier_info
        )
        self.mp1.scene.sigMouseClicked.connect(self.mouse_clicked)

        # xmap constraint
        self.model = XmapConstraintsTableModel()
        self.tableView.setModel(self.model)
        self.btn_mask_param_add.clicked.connect(self.add_param_constraint)
        self.btn_mask_param_delete.clicked.connect(self.delete_param_constraint)

        self.btn_mask_param_evaluate.clicked.connect(
            lambda: self.mask_evaluate("mask_parameter")
        )
        self.btn_mask_param_apply.clicked.connect(
            lambda: self.mask_apply("mask_parameter")
        )

        self.work_dir = None
        if path is not None:
            path = os.path.abspath(path)
            if os.path.isfile(path):
                self.fname.setText(str(path))
                self.work_dir = os.path.dirname(path)
            elif os.path.isdir(path):
                self.work_dir = path
        else:
            self.work_dir = os.path.expanduser("~")

        self.MaskWidget.setCurrentIndex(0)
        header = self.tableView.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        self.tabWidget.setCurrentIndex(0)

        self.comboBox_partition_mapname0.currentIndexChanged.connect(
            self.update_partition_mapname
        )
        self.comboBox_partition_mapname1.currentIndexChanged.connect(
            self.update_partition_mapname
        )
        self.save_load_settings(mode="load")
        self._setup_tooltips()
        self.show()

    def _setup_tooltips(self):
        # ── Input file & data loading ─────────────────────────────────────────
        self.btn_select_raw.setToolTip(
            "Browse for a scattering data file (.h5, .hdf, .tiff, .imm, .bin)"
        )
        self.fname.setToolTip("Path to the scattering data file")
        self.comboBox_beamline.setToolTip(
            "Select the beamline configuration that matches your data source"
        )
        self.spinBox_3.setToolTip("First frame index to include (0-based)")
        self.spinBox_4.setToolTip(
            "Frames to load and average.\n"
            "  0 = all remaining frames from begin index\n"
            " >0 = exactly that many frames (clamped to file end)\n"
            " <0 = representative subset: max(1000, total ÷ 5)"
        )
        self.btn_load.setToolTip(
            "Load and average the selected frame range from the data file"
        )

        # ── Metadata & parameters ─────────────────────────────────────────────
        self.metadata_tree.setToolTip(
            "Edit detector geometry (distance, pixel size, energy, beam center).\n"
            "Changes take effect after clicking Update Parameters."
        )
        self.btn_update_parameters.setToolTip(
            "Recompute the q/φ geometry from the current metadata values"
        )

        # ── Beam center ───────────────────────────────────────────────────────
        self.btn_find_center.setToolTip(
            "Auto-detect beam center using centro-symmetry cross-correlation"
        )
        self.pushButton_goto_max.setToolTip(
            "Jump to the frame with the highest total intensity"
        )
        self.btn_swapxy.setToolTip("Swap the X and Y coordinates of the beam center")

        # ── Mask history ──────────────────────────────────────────────────────
        self.btn_mask_undo.setToolTip("Undo the last mask apply operation")
        self.btn_mask_redo.setToolTip("Redo the last undone mask operation")
        self.btn_mask_reset.setToolTip(
            "Reset the mask to its initial state (blemish map only)"
        )

        # ── Blemish tab ───────────────────────────────────────────────────────
        self.btn_select_blemish.setToolTip(
            "Browse for a blemish / dead-pixel map file (HDF5 or TIFF)"
        )
        self.blemish_fname.setToolTip("Path to the blemish file")
        self.blemish_path.setToolTip(
            "HDF5 dataset path inside the blemish file (e.g. /qmap/mask)"
        )
        self.btn_apply_blemish.setToolTip(
            "Preview the blemish mask on the image without committing"
        )
        self.btn_mask_blemish_apply.setToolTip(
            "AND the blemish mask into the current working mask"
        )

        # ── Mask file tab ─────────────────────────────────────────────────────
        self.btn_select_maskfile.setToolTip(
            "Browse for an external mask file (HDF5 or TIFF)"
        )
        self.maskfile_fname.setToolTip("Path to the external mask file")
        self.maskfile_path.setToolTip(
            "HDF5 dataset path inside the mask file (e.g. /xpcs/mask)"
        )
        self.btn_apply_maskfile.setToolTip(
            "Preview the imported mask on the image without committing"
        )
        self.btn_mask_file_apply.setToolTip(
            "AND the imported mask into the current working mask"
        )

        # ── Draw tab ──────────────────────────────────────────────────────────
        self.cb_selector_type.setToolTip(
            "Shape to draw: Circle, Polygon, Ellipse, or Rectangle"
        )
        self.cb_selector_mode.setToolTip(
            "Exclusive: mask pixels inside the shape\n"
            "Inclusive: unmask pixels inside the shape"
        )
        self.cb_selector_color.setToolTip("Overlay color for the drawn ROI shape")
        self.plot_width.setToolTip("Line width (px) of the drawn shape outline")
        self.spinBox_num_edges.setToolTip("Number of vertices for polygon shapes")
        self.btn_mask_draw_add.setToolTip(
            "Add a new draggable ROI shape to the image"
        )
        self.btn_mask_draw_evaluate.setToolTip(
            "Preview the mask produced by all drawn shapes"
        )
        self.btn_mask_draw_apply.setToolTip(
            "Apply drawn shapes into the working mask"
        )

        # ── Threshold tab ─────────────────────────────────────────────────────
        self.checkBox_threshold_low_enable.setToolTip(
            "Enable lower threshold — pixels below the value will be masked"
        )
        self.binary_threshold_low.setToolTip(
            "Lower intensity threshold; pixels below this value are masked"
        )
        self.checkBox_threshold_high_enable.setToolTip(
            "Enable upper threshold — pixels above the value will be masked"
        )
        self.binary_threshold_high.setToolTip(
            "Upper intensity threshold; pixels above this value are masked"
        )
        self.checkBox_threshold_low_preset.setToolTip(
            "Quick-set the lower threshold to a preset value"
        )
        self.checkBox_threshold_high_preset.setToolTip(
            "Quick-set the upper threshold to a preset value"
        )
        self.pushButton_5.setToolTip(
            "Erode: shrink masked regions by one pixel layer"
        )
        self.pushButton_9.setToolTip(
            "Dilate: expand masked regions by one pixel layer"
        )
        self.pushButton_16.setToolTip(
            "Morphological open: erode then dilate — removes small masked islands"
        )
        self.pushButton_15.setToolTip(
            "Morphological close: dilate then erode — fills small masked holes"
        )
        self.btn_mask_threshold_evaluate.setToolTip(
            "Preview threshold + morphology result"
        )
        self.btn_mask_threshold_apply.setToolTip(
            "Apply threshold and morphology into the working mask"
        )

        # ── Manual list tab ───────────────────────────────────────────────────
        self.mask_list_1based.setToolTip(
            "Treat coordinates as 1-indexed (MATLAB/SPEC convention) instead of 0-indexed"
        )
        self.mask_list_rowcol.setToolTip(
            "Treat coordinates as (row, col) instead of (x, y)"
        )
        self.btn_mask_list_load.setToolTip(
            "Load pixel coordinates from a CSV, JSON, or text file"
        )
        self.mask_list_input.setToolTip(
            "Type pixel coordinates manually, e.g.  (x1,y1),(x2,y2)"
        )
        self.btn_mask_list_add.setToolTip(
            "Add the typed coordinates to the pixel list"
        )
        self.mask_list_include.setToolTip(
            "When double-clicking the image, also select neighboring pixels\n"
            "with similar intensity (within Radius and Variation limits)"
        )
        self.mask_list_radius.setToolTip(
            "Search radius (px) around a double-clicked pixel for intensity matching"
        )
        self.mask_list_variation.setToolTip(
            "Maximum intensity difference (%) to include a neighboring pixel"
        )
        self.mask_list_xylist.setToolTip(
            "Pixel coordinates queued to be masked — double-click the image to add points"
        )
        self.btn_mask_list_evaluate.setToolTip("Preview the pixel-list mask")
        self.btn_mask_list_clear.setToolTip("Remove all entries from the pixel list")
        self.btn_mask_list_apply.setToolTip(
            "Apply the pixel list into the working mask"
        )

        # ── Outlier tab ───────────────────────────────────────────────────────
        self.comboBox_outlier_target.setToolTip(
            "CircularRings: detect outliers within q-annuli\n"
            "AdjacentPixels: detect outliers within fixed spatial tiles"
        )
        self.outlier_num_roi.setToolTip(
            "Number of q-rings (CircularRings) or tile size in pixels (AdjacentPixels)"
        )
        self.comboBox_outlier_method.setToolTip(
            "Detection statistic:\n"
            "  median_absolute_deviation — robust to non-Gaussian tails\n"
            "  percentile — simpler rank-based cutoff"
        )
        self.outlier_cutoff.setToolTip(
            "Threshold in σ (MAD mode) or percentile above which pixels are flagged"
        )
        self.btn_mask_outlier_evaluate.setToolTip("Preview detected outliers")
        self.btn_mask_outlier_apply.setToolTip(
            "Apply outlier mask into the working mask"
        )

        # ── Parametrization tab ───────────────────────────────────────────────
        self.comboBox_param_xmap_name.setToolTip(
            "Select a geometry map (q, phi, x, y, chi…) to constrain"
        )
        self.doubleSpinBox_param_vbeg.setToolTip(
            "Minimum value of the selected map to keep unmasked"
        )
        self.doubleSpinBox_param_vend.setToolTip(
            "Maximum value of the selected map to keep unmasked"
        )
        self.comboBox_param_logic.setToolTip(
            "Logical operator combining multiple constraint rows: AND / OR / NOT"
        )
        self.btn_mask_param_add.setToolTip("Add a new constraint row to the table")
        self.btn_mask_param_delete.setToolTip("Remove the selected constraint row")
        self.btn_mask_param_evaluate.setToolTip(
            "Preview the mask produced by the constraint table"
        )
        self.btn_mask_param_apply.setToolTip(
            "Apply the parameter constraints into the working mask"
        )
        self.tableView.setToolTip(
            "Geometry constraints — rows are combined with the selected logic operator"
        )

        # ── Display controls ──────────────────────────────────────────────────
        self.plot_index.setToolTip(
            "What to display: scattering image, mask, dynamic/static partition, or preview"
        )
        self.plot_center.setToolTip("Show or hide the beam center crosshair")
        self.checkBox_showxy.setToolTip(
            "Show (x, y) pixel coordinates next to the cursor while hovering over the image"
        )
        self.plot_log.setToolTip("Use logarithmic color scale")
        self.plot_cmap.setToolTip("Colormap for intensity display")
        self.btn_plot.setToolTip("Refresh the image display")
        self.infobar.setToolTip("Pixel coordinates and intensity value under the cursor")

        # ── Q-phi partition ───────────────────────────────────────────────────
        self.sb_sqnum.setToolTip("Number of q-rings in the static (fine) partition")
        self.sb_dqnum.setToolTip("Number of q-rings in the dynamic (coarse) partition")
        self.sb_spnum.setToolTip("Number of φ sectors in the static partition")
        self.sb_dpnum.setToolTip("Number of φ sectors in the dynamic partition")
        self.doubleSpinBox_phi_offset.setToolTip(
            "Rotate the φ=0 reference direction by this angle (degrees).\n"
            "Useful to avoid pixels along horizontal or vertical lines\n"
            "being split across two different azimuthal sectors."
        )
        self.partition_style.setToolTip(
            "Radial bin spacing: Linear (uniform Δq) or Logarithmic (uniform Δq/q)"
        )
        self.spinBox_symmetry_fold.setToolTip(
            "Assume N-fold symmetry around the beam center (1 = none).\n"
            "At N > 1, azimuthal sectors separated by 360°/N are treated as equivalent."
        )
        self.btn_compute_qpartition.setToolTip(
            "Compute the q-φ partition map with the current settings"
        )

        # ── X-Y mesh partition ────────────────────────────────────────────────
        self.sb_sxnum.setToolTip("Number of x bins in the static partition")
        self.sb_dxnum.setToolTip("Number of x bins in the dynamic partition")
        self.sb_synum.setToolTip("Number of y bins in the static partition")
        self.sb_dynum.setToolTip("Number of y bins in the dynamic partition")
        self.btn_compute_qpartition2.setToolTip("Compute the x-y mesh partition")

        # ── Ellipse (eq-ephi) partition ───────────────────────────────────────
        self.sb_sxnum_2.setToolTip(
            "Static equivalent-q rings (ellipse-corrected radial bins)"
        )
        self.sb_dxnum_2.setToolTip("Dynamic equivalent-q rings")
        self.sb_synum_2.setToolTip(
            "Static equivalent-φ sectors (ellipse-corrected azimuthal bins)"
        )
        self.sb_dynum_2.setToolTip("Dynamic equivalent-φ sectors")
        self.btn_compute_qpartition4.setToolTip(
            "Compute the ellipse-corrected eq-ephi partition"
        )

        # ── General two-axis partition ────────────────────────────────────────
        self.comboBox_partition_mapname0.setToolTip(
            "Primary map axis for the custom partition (q, phi, x, y…)"
        )
        self.comboBox_partition_mapname1.setToolTip(
            "Secondary map axis for the custom partition"
        )
        self.sb_partition_sn0.setToolTip("Static bins for the primary axis")
        self.sb_partition_dn0.setToolTip("Dynamic bins for the primary axis")
        self.sb_partition_sn1.setToolTip("Static bins for the secondary axis")
        self.sb_partition_dn1.setToolTip("Dynamic bins for the secondary axis")
        self.comboBox_partition_style0.setToolTip(
            "Bin spacing for the primary axis: Linear or Logarithmic"
        )
        self.comboBox_partition_style1.setToolTip(
            "Bin spacing for the secondary axis: Linear or Logarithmic"
        )
        self.btn_compute_qpartition3.setToolTip(
            "Compute the custom two-axis partition"
        )

        # ── Output ────────────────────────────────────────────────────────────
        self.comboBox_output_type.setToolTip(
            "Export format:\n"
            "  Nexus-XPCS — HDF5 with mask and partition maps\n"
            "  Mask-Only  — TIFF mask file only"
        )
        self.pushButton.setToolTip(
            "Save the current mask and partition to disk"
        )

    def goto_max(self):
        self.sm.goto_max()
        self.display_metadata()
        self.plot()

    def update_partition_mapname(self):
        idx0 = self.comboBox_partition_mapname0.currentIndex()
        idx1 = self.comboBox_partition_mapname1.currentIndex()
        if idx0 == idx1:
            total_size = self.comboBox_partition_mapname1.count()
            new_idx = (idx0 + 1) % total_size
            self.comboBox_partition_mapname1.setCurrentIndex(new_idx)

    def mouse_clicked(self, event):
        if not event.double():
            return
        current_idx = self.MaskWidget.currentIndex()
        if current_idx not in [3, 5]:
            return

        if not self.mp1.scene.itemsBoundingRect().contains(event.pos()):
            return

        mouse_point = self.mp1.getView().mapSceneToView(event.pos())
        col = int(mouse_point.x())
        row = int(mouse_point.y())
        if current_idx == 3:
            # manual mode; select the dead pixel with mouse click
            if not self.mask_list_include.isChecked():
                self.mask_list_add_pts([np.array([col, row])])
            else:
                kwargs = {
                    "radius": self.mask_list_radius.value(),
                    "variation": self.mask_list_variation.value(),
                    "cen": (row, col),
                }
                pos = self.sm.get_pts_with_similar_intensity(**kwargs)
                self.mask_list_add_pts(pos)

    def mask_action(self, action):
        if not self.is_ready():
            return
        self.sm.mask_action(action)
        self.plot_index.setCurrentIndex(0)
        self.plot_index.setCurrentIndex(1)

    def find_center(self):
        if not self.is_ready():
            return
        try:
            self.btn_find_center.setText("Finding Center ...")
            self.btn_find_center.setDisabled(True)
            self.centralwidget.repaint()
            center_vh = self.sm.find_center(beamstop_diameter=0)
        except Exception:
            traceback.print_exc()
            self.statusbar.showMessage("Failed to find center. Abort", 2000)
        else:
            cen_old = self.sm.get_center()
            self.update_parameters(new_center_vh=center_vh)
            cen_new = self.sm.get_center()
            logger.info(f"found center: {cen_old} --> {cen_new}")
        finally:
            self.btn_find_center.setText("Find Center")
            self.btn_find_center.setEnabled(True)

    _OUTLIER_INFO = {
        ("CircularRings", "median_absolute_deviation"): (
            "Divides the image into concentric q-rings. "
            "Within each ring, flags pixels whose intensity deviates from the "
            "median by more than cutoff × MAD."
        ),
        ("CircularRings", "percentile"): (
            "Divides the image into concentric q-rings. "
            "Within each ring, clips extremes to the 5–95th percentile range, "
            "then flags pixels more than cutoff × std from the clipped mean."
        ),
        ("AdjacentPixels", "median_absolute_deviation"): (
            "Divides the image into fixed-size square boxes (brightest first). "
            "Within each box, flags pixels whose intensity deviates from the "
            "median by more than cutoff × MAD."
        ),
        ("AdjacentPixels", "percentile"): (
            "Divides the image into fixed-size square boxes (brightest first). "
            "Within each box, clips extremes to the 5–95th percentile range, "
            "then flags pixels more than cutoff × std from the clipped mean."
        ),
    }

    def _on_outlier_target_changed(self):
        target = self.comboBox_outlier_target.currentText()
        if target == "CircularRings":
            self.label_outlier_target_info.setText("num. circular ROI:")
            self.outlier_num_roi.setValue(180)
        elif target == "AdjacentPixels":
            self.label_outlier_target_info.setText("adjacent box size (pixel):")
            self.outlier_num_roi.setValue(32)

    def _update_outlier_info(self):
        target = self.comboBox_outlier_target.currentText()
        method = self.comboBox_outlier_method.currentText()
        text = self._OUTLIER_INFO.get((target, method), "")
        self.statusbar.showMessage(text)

    def mask_evaluate(self, target=None):
        if target is None or not self.is_ready():
            return

        if target == "mask_blemish":
            kwargs = {
                "fname": self.blemish_fname.text(),
                "key": self.blemish_path.text(),
            }
        elif target == "mask_file":
            kwargs = {
                "fname": self.maskfile_fname.text(),
                "key": self.maskfile_path.text(),
            }
        elif target == "mask_list":
            num_row = self.mask_list_xylist.count()
            val = [str(self.mask_list_xylist.item(i).text()) for i in range(num_row)]
            val = " ".join(val)
            xy = text_to_array(val)
            xy = xy[0 : xy.size // 2 * 2].reshape(-1, 2).T
            xy = np.roll(xy, shift=1, axis=0)
            kwargs = {"zero_loc": xy}
        elif target == "mask_draw":
            rois = extract_roi_geometry(self.mp1.roi, self.mp1.imageItem)
            self.sm.set_draw_rois(rois)
            self.mp1.remove_rois(filter_str="roi_")
            keep = self.sm.evaluate_draw_mask()
            kwargs = {"arr": np.logical_not(keep)}
        elif target == "mask_threshold":
            kwargs = {
                "low": self.binary_threshold_low.value(),
                "high": self.binary_threshold_high.value(),
                "low_enable": self.checkBox_threshold_low_enable.isChecked(),
                "high_enable": self.checkBox_threshold_high_enable.isChecked(),
            }

        elif target == "mask_outlier":
            num = self.outlier_num_roi.value()
            cutoff = self.outlier_cutoff.value()
            method = self.comboBox_outlier_method.currentText()
            method = {"percentile": "percentile", "median_absolute_deviation": "mad"}[
                method
            ]
            outlier_target = self.comboBox_outlier_target.currentText()

            if outlier_target == "CircularRings":
                saxs1d, zero_loc = self.sm.compute_saxs1d(
                    num=num, cutoff=cutoff, method=method
                )
                x_label = "q (Å⁻¹)"
            else:  # AdjacentPixels
                saxs1d, zero_loc = self.sm.compute_adjacent_saxs1d(
                    box_size=num, cutoff=cutoff, method=method
                )
                x_label = "box index (sorted by mean)"

            self.mask_outlier_hdl.clear()
            p = self.mask_outlier_hdl
            p.addLegend()
            p.plot(
                saxs1d[0],
                saxs1d[1],
                name="average_ref",
                pen=pg.mkPen(color="g", width=2),
            )
            # p.plot(saxs1d[0], saxs1d[4], name='average_raw',
            #        pen=pg.mkPen(color='m', width=2))
            p.plot(
                saxs1d[0], saxs1d[2], name="cutoff", pen=pg.mkPen(color="b", width=2)
            )
            p.plot(
                saxs1d[0],
                saxs1d[3],
                name="maximum value",
                pen=pg.mkPen(color="r", width=2),
            )
            p.setLabel("bottom", x_label)
            p.setLabel("left", "Intensity (a.u.)")
            p.setLogMode(y=True)
            kwargs = {"zero_loc": zero_loc}

        elif target == "mask_parameter":
            kwargs = {"constraints": self.model._data}

        msg = self.sm.mask_evaluate(target, **kwargs)
        self.statusbar.showMessage(msg, 10000)
        self.plot_index.setCurrentIndex(0)
        self.plot_index.setCurrentIndex(5)
        return

    def update_threshold_preset(self, target="low"):
        preset_value_mapping = {
            "None": None,
            "0": 0,
            "1": 1,
            "-1": -1,
            "uint8": 255,
            "int8": 127,
            "uint16": 65535,
            "int16": 32767,
            "uint32": 4294967295,
            "int32": 2147483647,
            "uint24": 16777215,
            "int24": 8388607,
        }
        if target == "low":
            key = self.checkBox_threshold_low_preset.currentText()
            if key != "None":
                self.binary_threshold_low.setValue(preset_value_mapping[key])
        elif target == "high":
            key = self.checkBox_threshold_high_preset.currentText()
            if key != "None":
                self.binary_threshold_high.setValue(preset_value_mapping[key])

    def mask_apply(self, target):
        if not self.is_ready():
            return

        self.sm.mask_apply(target)
        # perform evaluate again so the saxs1d shows the new results;
        if target == "mask_outlier":
            self.mask_evaluate(target=target)
        elif target == "mask_list":
            self.mask_list_clear()
        elif target == "mask_parameter":
            self.model.clear()

        self.plot_index.setCurrentIndex(0)
        self.plot_index.setCurrentIndex(1)

    def update_index(self):
        idx = self.mp1.currentIndex
        self.plot_index.setCurrentIndex(idx)
        # make the mask and preview binary
        if idx in [2, 5]:
            self.mp1.setLevels(0, 1)

    def is_ready(self):
        if not self.sm.is_ready():
            self.statusbar.showMessage("No scattering image is loaded.", 500)
            return False
        return True

    def update_parameters(self, swapxy=False, new_center_vh=None):
        if not self.is_ready():
            return

        if swapxy:
            self.sm.dset.swapxy()
        if new_center_vh is not None:
            self.sm.dset.set_center_vh(new_center_vh)
        self.sm.update_parameters()
        self.display_metadata()
        self.plot()

    def select_raw(self):
        default_dir = self.work_dir if self.work_dir else os.getcwd()

        fname = QFileDialog.getOpenFileName(
            self,
            caption="Select raw file hdf",
            dir=default_dir,  # <-- Fixed!
            filter="Supported Formats(*.hdf *.h5 *.hdf5 *.imm *.bin *.tif *.tiff *.fits *.raw *.bin.* *.tpx *.tpx.*)",
        )[0]
        if fname:
            self.fname.setText(fname)
            self.work_dir = os.path.dirname(fname)
        return

    def select_blemish(self):
        fname = QFileDialog.getOpenFileName(
            self,
            "Select blemish file",
            filter="Supported Formats(*.tiff *.tif *.h5 *.hdf *.hdf5)",
        )[0]
        if fname not in [None, ""]:
            self.blemish_fname.setText(fname)

        if fname.endswith(".tif") or fname.endswith(".tiff"):
            self.blemish_path.setDisabled(True)
        else:
            self.blemish_path.setEnabled(True)

        return

    def select_maskfile(self):
        fname = QFileDialog.getOpenFileName(
            self,
            "Select mask file",
            filter="Supported Formats(*.tiff *.tif *.h5 *.hdf *.hdf5)",
        )[0]
        # fname = "../tests/data/triangle_mask/mask_lambda_test.h5"
        if fname not in [None, ""]:
            self.maskfile_fname.setText(fname)
        if fname.endswith(".tif") or fname.endswith(".tiff"):
            self.maskfile_path.setDisabled(True)
        else:
            self.maskfile_path.setEnabled(True)
        return

    def load(self):
        # self.fname.setText('/mnt/c/Users/mqichu/Documents/local_dev/pysimplemask/tests/data/E0135_La0p65_L2_013C_att04_Rq0_00001/E0135_La0p65_L2_013C_att04_Rq0_00001_0001-100000.hdf')
        if not os.path.isfile(self.fname.text()):
            self.select_raw()
        if not os.path.isfile(self.fname.text()):
            self.statusbar.showMessage("select a valid file")
            return

        self.btn_load.setText("loading...")
        self.statusbar.showMessage("loading data...", 120000)
        self.centralwidget.repaint()

        fname = self.fname.text()
        kwargs = {
            "begin_idx": self.spinBox_3.value(),
            "num_frames": self.spinBox_4.value(),
            "beamline": self.comboBox_beamline.currentText(),
        }
        if not self.sm.read_data(fname, **kwargs):
            return
        # else:
        #     stype = self.sm.dset.stype

        for key in (
            "comboBox_param_xmap_name",
            "comboBox_partition_mapname0",
            "comboBox_partition_mapname1",
        ):
            widget = getattr(self, key)
            widget.clear()
            widget.addItems(list(self.sm.qmap.keys()))

        self.comboBox_param_xmap_name.currentIndexChanged.connect(
            self.update_xmap_limits
        )
        self.update_xmap_limits()

        while self.plot_index.count() > 6:
            self.plot_index.removeItem(6)
        for key in self.sm.qmap.keys():
            self.plot_index.addItem(key)

        self.display_metadata()
        self.plot()
        self.statusbar.showMessage("data is loaded", 500)
        self.btn_load.setText("load data")
        self.btn_load.repaint()

    def display_metadata(self):
        param_struct = self.sm.dset.get_parametertree_structure()
        self.metadata_parameter = Parameter.create(**param_struct)
        self.metadata_parameter.sigTreeStateChanged.connect(
            self.update_parameter_to_dset
        )
        self.metadata_tree.setParameters(self.metadata_parameter, showTop=False)

    def update_parameter_to_dset(self, param, changes):
        self.sm.dset.update_metadata_from_changes(changes)

    def _remove_xy_label(self):
        """Remove the floating (x, y) text item from the view."""
        if self._xy_text_item is not None:
            try:
                self.mp1.getView().removeItem(self._xy_text_item)
            except Exception:
                pass
            self._xy_text_item = None

    def show_location(self, pos):
        if self.sm.shape is None:
            return
        if not self.mp1.scene.itemsBoundingRect().contains(pos):
            self._remove_xy_label()
            return
        mouse_point = self.mp1.getView().mapSceneToView(pos)
        idx = self.mp1.currentIndex
        col = int(mouse_point.x())
        row = int(mouse_point.y())
        msg = self.sm.dset.get_coordinates(col, row, idx)
        if msg:
            self.infobar.clear()
            self.infobar.setText(msg)

        # Floating (x, y) label next to the cursor
        h, w = self.sm.shape
        show_xy = self.checkBox_showxy.isChecked()
        if show_xy and 0 <= col < w and 0 <= row < h:
            if self._xy_text_item is None:
                self._xy_text_item = pg.TextItem(
                    color=(255, 255, 255),
                    anchor=(0, 0),
                    fill=pg.mkBrush(0, 0, 0, 160),
                )
                self.mp1.getView().addItem(self._xy_text_item)
            self._xy_text_item.setText(f"({col}, {row})")
            # Offset slightly so the label doesn't sit on top of the cursor pixel
            offset = max(w, h) * 0.01
            self._xy_text_item.setPos(col + offset, row + offset)
        else:
            self._remove_xy_label()

    def plot(self):
        if not self.is_ready():
            return
        cmap = self.plot_cmap.currentText()
        plot_center = self.plot_center.isChecked()
        self.mp1.clear()
        self.mp1.setImage(self.sm.dset.data_display)
        self.mp1.adjust_viewbox()
        self.mp1.set_colormap(cmap)
        if plot_center:
            center = self.sm.get_center(mode="vh")
            h, w = self.sm.shape
            row, col = center[0], center[1]
            # Only draw if the center is within ±1 image dimension of the canvas.
            if -w < col < 2 * w and -h < row < 2 * h:
                t = pg.ScatterPlotItem()
                t.addPoints(x=[col], y=[row], symbol="+", size=15)
                self.mp1.add_item(t, label="center")
        self.mp1.setCurrentIndex(1)
        self.plot_index.setCurrentIndex(1)

    def add_drawing(self):
        if not self.is_ready():
            return
        if self.MaskWidget.currentIndex() != 1:
            return
        color = ("r", "g", "y", "b", "c", "m", "k", "w")[
            self.cb_selector_color.currentIndex()
        ]
        sl_type = self.cb_selector_type.currentText()
        sl_mode = self.cb_selector_mode.currentText()
        width = self.plot_width.value()
        num_edges = self.spinBox_num_edges.value()

        cen = self.sm.get_center(mode="xy")
        shape = self.sm.shape
        if cen[0] < 0 or cen[1] < 0 or cen[0] > shape[1] or cen[1] > shape[0]:
            logger.warning("beam center is out of range, use image center instead")
            cen = (shape[1] // 2, shape[0] // 2)

        if sl_mode == "inclusive":
            pen = pg.mkPen(color=color, width=width, style=QtCore.Qt.DotLine)
        else:
            pen = pg.mkPen(color=color, width=width)
        handle_pen = pg.mkPen(color=color, width=width)
        kwargs = {"pen": pen, "removable": True, "hoverPen": pen,
                  "handlePen": handle_pen, "movable": True}

        if sl_type == "Ellipse":
            size = (120, 160)
            new_roi = pg.EllipseROI(
                (cen[0] - size[0] // 2, cen[1] - size[1] // 2), size, **kwargs)
            new_roi.addScaleHandle([0.5, 0], [0.5, 0.5])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0.5])
            new_roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        elif sl_type == "Circle":
            radius = 60
            new_roi = pg.CircleROI(
                pos=[cen[0] - radius, cen[1] - radius], radius=radius, **kwargs)
            new_roi.addScaleHandle([0.5, 0], [0.5, 0.5])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        elif sl_type == "Polygon":
            offset = np.random.randint(0, 360)
            theta = np.linspace(0, np.pi * 2, num_edges + 1) + offset
            x = 60 * np.cos(theta) + cen[0]
            y = 60 * np.sin(theta) + cen[1]
            new_roi = pg.PolyLineROI(np.vstack([x, y]).T, closed=True, **kwargs)
        elif sl_type == "Rectangle":
            new_roi = pg.RectROI(cen, [200, 150], **kwargs)
            for h in ([0, 0], [0, 0.5], [0, 1], [0.5, 0],
                      [0.5, 1], [1, 0], [1, 0.5], [1, 1]):
                new_roi.addScaleHandle(h, [1 - h[0], 1 - h[1]])
        else:
            logger.error("unsupported ROI type %s for button add", sl_type)
            return
        new_roi.sl_mode = sl_mode
        roi_key = self.mp1.add_item(new_roi)
        new_roi.sigRemoveRequested.connect(lambda: self.mp1.remove_item(roi_key))

    def update_xmap_limits(self):
        xmap_name = self.comboBox_param_xmap_name.currentText()
        if not xmap_name:
            return
        vmin, vmax = self.sm.qmap[xmap_name].min(), self.sm.qmap[xmap_name].max()
        unit = self.sm.qmap_unit[xmap_name]
        self.label_param_minval.setText(f"Min: {vmin:.4f} {unit}")
        self.label_param_maxval.setText(f"Max: {vmax:.4f} {unit}")
        self.doubleSpinBox_param_vbeg.setValue(vmin)
        self.doubleSpinBox_param_vend.setValue(vmax)

    def add_param_constraint(self):
        xmap_name = self.comboBox_param_xmap_name.currentText()
        if not xmap_name:
            return
        vbeg = self.doubleSpinBox_param_vbeg.value()
        vend = self.doubleSpinBox_param_vend.value()
        if vbeg > vend:
            vbeg, vend = vend, vbeg
            self.doubleSpinBox_param_vbeg.setValue(vbeg)
            self.doubleSpinBox_param_vend.setValue(vend)
        logic = self.comboBox_param_logic.currentText()
        unit = self.sm.qmap_unit[xmap_name]
        self.model.addRow([xmap_name, logic, unit, vbeg, vend])

    def delete_param_constraint(self):
        idx = self.tableView.currentIndex().row()
        self.model.removeRow(idx)

    def compute_partition(self):
        if not self.is_ready():
            return

        if self.tabWidget.currentIndex() == 0:
            kwargs = {
                "mode": "q-phi",
                "sq_num": self.sb_sqnum.value(),
                "dq_num": self.sb_dqnum.value(),
                "sp_num": self.sb_spnum.value(),
                "dp_num": self.sb_dpnum.value(),
                "phi_offset": self.doubleSpinBox_phi_offset.value(),
                "style": self.partition_style.currentText(),
                "symmetry_fold": self.spinBox_symmetry_fold.value(),
            }
        elif self.tabWidget.currentIndex() == 1:
            kwargs = {
                "mode": "x-y",
                "sq_num": self.sb_sxnum.value(),
                "sp_num": self.sb_synum.value(),
                "dq_num": self.sb_dxnum.value(),
                "dp_num": self.sb_dynum.value(),
            }
        elif self.tabWidget.currentIndex() == 2:
            map0 = self.comboBox_partition_mapname0.currentText()
            map1 = self.comboBox_partition_mapname1.currentText()
            kwargs = {
                "mode": f"{map0}-{map1}",
                "sq_num": self.sb_partition_sn0.value(),
                "sp_num": self.sb_partition_sn1.value(),
                "dq_num": self.sb_partition_dn0.value(),
                "dp_num": self.sb_partition_dn1.value(),
            }
        elif self.tabWidget.currentIndex() == 3:
            kwargs = {
                "mode": "eq-ephi",
                "sq_num": self.sb_sxnum_2.value(),
                "sp_num": self.sb_synum_2.value(),
                "dq_num": self.sb_dxnum_2.value(),
                "dp_num": self.sb_dynum_2.value(),
            }

        sq_num = least_multiple(kwargs["dq_num"], kwargs["sq_num"])
        sp_num = least_multiple(kwargs["dp_num"], kwargs["sp_num"])
        if sq_num != kwargs["sq_num"]:
            if self.tabWidget.currentIndex() == 0:
                self.sb_sqnum.setValue(sq_num)
            else:
                self.sb_sxnum.setValue(sq_num)
            kwargs["sq_num"] = sq_num
        if sp_num != kwargs["sp_num"]:
            if self.tabWidget.currentIndex() == 0:
                self.sb_spnum.setValue(sp_num)
            else:
                self.sb_synum.setValue(sp_num)
            kwargs["sp_num"] = sp_num

        self.btn_compute_qpartition.setDisabled(True)
        self.statusbar.showMessage("Computing partition ... ", 10000)
        self.centralwidget.repaint()

        try:
            self.sm.compute_partition(**kwargs)
            self.statusbar.showMessage("New partition is generated.", 1000)
            self.plot_index.setCurrentIndex(0)
            self.plot_index.setCurrentIndex(3)
        except Exception:
            traceback.print_exc()
        finally:
            self.btn_compute_qpartition.setEnabled(True)
            # self.btn_compute_qpartition.setStyleSheet("background-color: green")

    def save_mask(self):
        if not self.is_ready():
            return
        save_type = self.comboBox_output_type.currentText()
        if save_type == "Nexus-XPCS":
            save_fname = QFileDialog.getSaveFileName(
                self, caption="Save mask/qmap as", filter="HDF (*.hdf)"
            )[0]
            if save_fname == "":
                logger.warning("received empty filename; exit without saving")
                return
            if not save_fname.endswith(".hdf"):
                save_fname += ".hdf"
            if self.sm.new_partition is None:
                self.compute_partition()
            target_function = self.sm.save_partition
        elif save_type == "Mask-Only":
            save_fname = QFileDialog.getSaveFileName(
                self, caption="Save mask as tif", filter="TIF (*.tif)"
            )[0]
            if not save_fname.endswith(".tif") and not save_fname.endswith(".tiff"):
                save_fname += ".tif"
            target_function = self.sm.save_mask

        try:
            target_function(save_fname)
            success_dialog = QMessageBox(self)
            success_dialog.setIcon(QMessageBox.Information)
            if save_type == "Nexus-XPCS":
                success_dialog.setText(
                    f"Successfully saved mask/qmap to file: {save_fname}"
                )
            else:
                success_dialog.setText(f"Successfully saved mask to file: {save_fname}")
            success_dialog.setDetailedText(save_fname)
            success_dialog.setWindowTitle("Success")
            success_dialog.exec()
        except Exception:
            error_message = traceback.format_exc()
            traceback.print_exc()
            error_dialog = QMessageBox(self)
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText("Failed to save mask/qmap to file")
            error_dialog.setDetailedText(error_message)
            error_dialog.setWindowTitle("Error")
            error_dialog.exec()

    def mask_list_load(self):
        if not self.is_ready():
            return

        fname = QFileDialog.getOpenFileName(
            self,
            "Select mask file",
            filter="Text/Json (*.txt *.csv *.json);;All files(*.*)",
        )[0]
        if fname in ["", None]:
            return

        try:
            xy = load_pixel_list(fname)
        except Exception:
            self.statusbar.showMessage(
                "only support json, csv and space separated file", 500
            )
            return

        if self.mask_list_rowcol.isChecked():
            xy = np.roll(xy, shift=1, axis=1)
        if self.mask_list_1based.isChecked():
            xy = xy - 1

        xy = xy.astype(np.int64)
        self.mask_list_add_pts(xy)

    def mask_list_add(self):
        pts = self.mask_list_input.text()
        self.mask_list_input.clear()
        if len(pts) < 1:
            self.statusbar.showMessage("Input list is almost empty.", 500)
            return

        xy = text_to_array(pts)
        xy = xy[0 : xy.size // 2 * 2].reshape(-1, 2)
        self.mask_list_add_pts(xy)

    def mask_list_add_pts(self, pts):
        for xy in pts:
            xy_str = str(xy)
            if xy_str not in self.sm.bad_pixel_set:
                self.mask_list_xylist.addItem(xy_str)
                self.sm.bad_pixel_set.add(xy_str)
        self.groupBox_11.setTitle("xy list: %d" % len(self.sm.bad_pixel_set))

    def mask_list_clear(self):
        self.sm.bad_pixel_set.clear()
        self.mask_list_xylist.clear()
        self.groupBox_11.setTitle("xy list")

    def save_load_settings(self, mode="save"):
        keys = [
            "comboBox_beamline",
        ]

        if mode == "save":
            config = {}
            # Save widget states
            for key in keys:
                widget = getattr(self, key, None)
                if isinstance(widget, QLineEdit):
                    config[key] = widget.text()
                elif isinstance(widget, QRadioButton):
                    config[key] = widget.isChecked()
                elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                    config[key] = widget.value()
                elif isinstance(widget, QComboBox):
                    config[key] = widget.currentIndex()
                elif isinstance(widget, QCheckBox):
                    config[key] = widget.isChecked()
                elif isinstance(widget, QTabWidget):
                    config[key] = widget.currentIndex()

            if hasattr(self, "splitter"):
                config["splitter_state"] = (
                    self.splitter.saveState().toBase64().data().decode()
                )
            if hasattr(self, "splitter_2"):
                config["splitter_2_state"] = (
                    self.splitter_2.saveState().toBase64().data().decode()
                )

            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
            logger.info(f"Saved configuration to [{CONFIG_FILE}]")

        elif mode == "load":
            if CONFIG_FILE.is_file():
                logger.info(f"Loading configuration from [{CONFIG_FILE}]")
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)

                for key, value in config.items():
                    widget = getattr(self, key, None)
                    if key == "main_geometry":
                        self.restoreGeometry(QByteArray.fromBase64(value.encode()))
                    elif key == "splitter_state" and hasattr(self, "splitter"):
                        self.splitter.restoreState(
                            QByteArray.fromBase64(value.encode())
                        )
                    elif key == "splitter_2_state" and hasattr(self, "splitter_2"):
                        self.splitter_2.restoreState(
                            QByteArray.fromBase64(value.encode())
                        )
                    elif isinstance(widget, QLineEdit):
                        widget.setText(value)
                    elif isinstance(widget, QRadioButton):
                        widget.setChecked(value)
                    elif isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                        widget.setValue(value)
                    elif isinstance(widget, QComboBox):
                        widget.setCurrentIndex(value)
                    elif isinstance(widget, QCheckBox):
                        widget.setChecked(value)
                    elif isinstance(widget, QTabWidget):
                        widget.setCurrentIndex(value)
            else:
                logger.error(f"Configuration file [{CONFIG_FILE}] not found.")

    def closeEvent(self, event):
        self.save_load_settings(mode="save")
        event.accept()


