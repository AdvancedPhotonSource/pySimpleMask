import logging
import os
import time

import h5py
import numpy as np
import pyqtgraph as pg
import tifffile
from pyqtgraph.Qt import QtCore, QtGui

from pysimplemask import __version__

from .area_mask import MaskAssemble
from .file_handler import get_handler
from .find_center import find_center
from .outlier_removal import outlier_removal_with_saxs
from .pyqtgraph_mod import LineROI
from .utils import (
    check_consistency,
    combine_partitions,
    generate_partition,
    hash_numpy_dict,
    optimize_integer_array,
)
from .ellipse_util import compute_ellipse_gradient, find_ellipse_parameters

pg.setConfigOptions(imageAxisOrder="row-major")


logger = logging.getLogger(__name__)


class SimpleMask(object):
    def __init__(self, pg_hdl, infobar):
        self.dset = None
        self.shape = None
        self.qmap = None
        self.mask = None
        self.mask_kernel = None
        self.new_partition = None

        self.hdl = pg_hdl
        self.infobar = infobar
        self.hdl.scene.sigMouseMoved.connect(self.show_location)
        self.bad_pixel_set = set()

    def is_ready(self):
        return self.dset is not None

    def find_center(self):
        if self.dset is None:
            return

        center_guess = self.get_center(mode="vh")
        center = find_center(
            self.dset.scat,
            mask=self.mask,
            center_guess=center_guess,
            scale="log",
        )
        return center

    def mask_evaluate(self, target, **kwargs):
        msg = self.mask_kernel.evaluate(target, **kwargs)
        # preview the mask
        mask = self.mask_kernel.get_one_mask(target)
        self.dset.set_preview(mask)
        return msg

    def mask_action(self, action="undo"):
        self.mask_kernel.redo_undo(action=action)
        self.mask_apply()

    def mask_apply(self, target=None):
        if target == "default_blemish":
            self.mask = self.mask_kernel.blemish
        else:
            self.mask = self.mask_kernel.apply(target)
        self.dset.update_mask(self.mask)

    def get_pts_with_similar_intensity(self, cen=None, radius=50, variation=50):
        return self.dset.get_pts_with_similar_intensity(cen, radius, variation)

    def save_mask(self, save_name):
        mask = self.mask.astype(np.uint8)
        tifffile.imwrite(save_name, mask, compression="LZW")

    def save_partition(self, save_fname, root="/qmap"):
        # if no partition is computed yet
        if self.new_partition is None:
            return

        for key, val in self.new_partition.items():
            self.new_partition[key] = optimize_integer_array(val)

        hash_val = hash_numpy_dict(self.new_partition)
        logger.info("Hash value of the partition: {}".format(hash_val))

        def optimize_save(group_handle, key, val):
            if isinstance(val, np.ndarray) and val.size > 1024:
                compression = "lzf"
            else:
                compression = None
            dset = group_handle.create_dataset(key, data=val, compression=compression)
            return dset

        with h5py.File(save_fname, "w") as hf:
            if root in hf:
                del hf[root]
            group_handle = hf.create_group(root)
            for key, val in self.new_partition.items():
                dset = optimize_save(group_handle, key, val)
                if "_v_list_dim" in key:
                    dim = int(key[-1])
                    dset.attrs["unit"] = self.new_partition["map_units"][dim]
                    dset.attrs["name"] = self.new_partition["map_names"][dim]
                    dset.attrs["size"] = val.size

            group_handle.attrs["hash"] = hash_val
            group_handle.attrs["version"] = __version__

    # generate 2d saxs
    def read_data(self, fname=None, beamline="APS_8IDI", **kwargs):
        self.dset = get_handler(beamline, fname)
        if self.dset is None:
            logger.error(f"failed to create a dataset handler for {fname}")
            return False

        t0 = time.perf_counter()
        self.dset.prepare_data(**kwargs)
        t1 = time.perf_counter()
        logger.info(f"data loaded in {t1 - t0: .1f} seconds")

        self.shape = self.dset.shape
        self.mask = np.ones(self.shape, dtype=bool)

        self.qmap, self.qmap_unit, _ = self.dset.compute_qmap()
        self.mask_kernel = MaskAssemble(self.shape, self.dset.scat)
        self.mask_apply(target="default_blemish")
        self.mask_kernel.update_qmap(self.qmap)
        return True

    def show_location(self, pos):
        if not self.hdl.scene.itemsBoundingRect().contains(pos) or self.shape is None:
            return
        mouse_point = self.hdl.getView().mapSceneToView(pos)
        idx = self.hdl.currentIndex
        col = int(mouse_point.x())
        row = int(mouse_point.y())
        msg = self.dset.get_coordinates(col, row, idx)
        if msg:
            self.infobar.clear()
            self.infobar.setText(msg)
        return None

    def show_saxs(
        self,
        cmap="jet",
        log=True,
        plot_center=True,
        plot_index=0,
        **kwargs,
    ):
        if self.dset is None:
            return
        self.hdl.clear()
        self.hdl.setImage(self.dset.data_display)
        self.hdl.adjust_viewbox()
        self.hdl.set_colormap(cmap)

        # plot center
        if plot_center:
            t = pg.ScatterPlotItem()
            center = self.get_center(mode="vh")
            logger.info(f"direct beam center is ({center[1]}, {center[0]})")
            t.addPoints(x=[center[1]], y=[center[0]], symbol="+", size=15)
            self.hdl.add_item(t, label="center")

        self.hdl.setCurrentIndex(plot_index)

        return

    def evaluate_drawing(self):
        """
        Apply current ROI drawings to the mask.

        This method uses QPainter to rasterize the ROI shapes into a QImage, which is then
        converted to a numpy boolean mask. This approach correctly handles rotated ROIs
        (e.g., rotated ellipses) by transforming the vector shape of the ROI into the
        image coordinate system before rasterization.
        """
        if self.dset is None:
            return

        # shape is (rows, cols) -> (height, width)
        h, w = self.dset.shape

        # Create QImages for mask accumulation.
        # We use QImage.Format_Grayscale8, where we'll draw with white (255) on black (0).
        # img_e collects "exclusive" ROIs (masked out regions)
        # img_i collects "inclusive" ROIs (regions to keep)
        img_e = QtGui.QImage(w, h, QtGui.QImage.Format.Format_Grayscale8)
        img_e.fill(0)
        img_i = QtGui.QImage(w, h, QtGui.QImage.Format.Format_Grayscale8)
        img_i.fill(0)

        p_e = QtGui.QPainter(img_e)
        p_i = QtGui.QPainter(img_i)

        # Setup painters: no border (NoPen), white fill (SolidPattern)
        for p in (p_e, p_i):
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(QtGui.QBrush(QtCore.Qt.GlobalColor.white))

        has_inclusive = False

        for k, x in self.hdl.roi.items():
            if not k.startswith("roi_"):
                continue

            # Retrieve the ROI's shape as a QPainterPath in its local coordinates.
            # pyqtgraph's EllipseROI.shape() (inherited by CircleROI) returns only a
            # 24-point polygon approximation -- a workaround for a Qt hit-testing bug --
            # which makes large circles/ellipses rasterize with visible straight edges.
            # Build a true ellipse path from the bounding rect instead so the mask is smooth.
            if isinstance(x, pg.EllipseROI):
                path = QtGui.QPainterPath()
                path.addEllipse(x.boundingRect())
            else:
                path = x.shape()
            # Map the path to the ImageItem's coordinate system to handle position, scale, and rotation.
            path = x.mapToItem(self.hdl.imageItem, path)

            if x.sl_mode == "exclusive":
                p_e.drawPath(path)
            elif x.sl_mode == "inclusive":
                has_inclusive = True
                p_i.drawPath(path)

        p_e.end()
        p_i.end()

        def qimage_to_mask(qimg, h, w):
            """Convert QImage to numpy boolean mask."""
            ptr = qimg.constBits()
            stride = qimg.bytesPerLine()
            # If stride != width, the image data has padding at the end of each row.
            # We reshape to (h, stride) first, then crop to (h, w).
            arr_padded = np.array(ptr).reshape(h, stride)
            if stride != w:
                arr = arr_padded[:, :w]
            else:
                arr = arr_padded
            # Pixels drawn with white are True; background is False.
            return arr > 0

        mask_e = qimage_to_mask(img_e, h, w)
        mask_i_raw = qimage_to_mask(img_i, h, w)

        self.hdl.remove_rois(filter_str="roi_")

        # Logic for processing inclusive/exclusive masks:
        # 1. Exclusive masks (mask_e) define pixels to REMOVE (False in final mask).
        # 2. Inclusive masks (mask_i) define pixels to KEEP.
        #    - If NO inclusive masks are drawn, everything is included by default (mask_i=True).
        #    - If ANY inclusive masks are drawn, only those regions are included.

        if not has_inclusive:
            mask_i_final = True
        else:
            mask_i_final = mask_i_raw

        # Final Mask = (NOT Exclusive) AND (Inclusive)
        mask_p = np.logical_not(mask_e) * mask_i_final

        return mask_p

    def add_drawing(
        self,
        num_edges=None,
        radius=60,
        color="r",
        sl_type="Polygon",
        width=3,
        sl_mode="exclusive",
        second_point=None,
        label=None,
        movable=True,
    ):
        # label: label of roi; default is None, which is for roi-draw
        if label is not None and label in self.hdl.roi:
            self.hdl.remove_item(label)

        cen = self.get_center(mode="xy")
        if cen[0] < 0 or cen[1] < 0 or cen[0] > self.shape[1] or cen[1] > self.shape[0]:
            logger.warning("beam center is out of range, use image center instead")
            cen = (self.shape[1] // 2, self.shape[0] // 2)

        if sl_mode == "inclusive":
            pen = pg.mkPen(color=color, width=width, style=QtCore.Qt.DotLine)
        else:
            pen = pg.mkPen(color=color, width=width)

        handle_pen = pg.mkPen(color=color, width=width)

        kwargs = {
            "pen": pen,
            "removable": True,
            "hoverPen": pen,
            "handlePen": handle_pen,
            "movable": movable,
        }
        if sl_type == "Ellipse":
            size = (120, 160)
            new_roi = pg.EllipseROI(
                (cen[0] - size[0] // 2, cen[1] - size[1] // 2), size, **kwargs
            )
            # add scale handle
            new_roi.addScaleHandle([0.5, 0], [0.5, 0.5])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0.5])
            new_roi.addScaleHandle([0, 0.5], [0.5, 0.5])
            # it's a rotation handle by default
            # new_roi.addScaleHandle([1, 0.5], [0.5, 0.5])

        elif sl_type == "Circle":
            if second_point is not None:
                radius = np.sqrt(
                    (second_point[1] - cen[1]) ** 2 + (second_point[0] - cen[0]) ** 2
                )
            new_roi = pg.CircleROI(
                pos=[cen[0] - radius, cen[1] - radius], radius=radius, **kwargs
            )
            new_roi.addScaleHandle([0.5, 0], [0.5, 0.5])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0.5])

        elif sl_type == "Polygon":
            if num_edges is None:
                num_edges = np.random.random_integers(6, 10)

            # add angle offset so that the new rois don't overlap with each
            # other
            offset = np.random.random_integers(0, 359)
            theta = np.linspace(0, np.pi * 2, num_edges + 1) + offset
            x = radius * np.cos(theta) + cen[0]
            y = radius * np.sin(theta) + cen[1]
            pts = np.vstack([x, y]).T
            new_roi = pg.PolyLineROI(pts, closed=True, **kwargs)

        elif sl_type == "Rectangle":
            new_roi = pg.RectROI(cen, [200, 150], **kwargs)
            # new_roi.addScaleHandle([0, 0], [1, 1])
            new_roi.addScaleHandle([0, 0], [1, 1])
            new_roi.addScaleHandle([0, 0.5], [1, 0.5])
            new_roi.addScaleHandle([0, 1], [1, 0])
            new_roi.addScaleHandle([0.5, 0], [0.5, 1])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0])
            new_roi.addScaleHandle([1, 0], [0, 1])
            new_roi.addScaleHandle([1, 0.5], [0, 0.5])
            new_roi.addScaleHandle([1, 1], [0, 0])

        elif sl_type == "Line":
            if second_point is None:
                return
            width = kwargs.pop("width", 1)
            new_roi = LineROI(cen, second_point, width, **kwargs)
        else:
            raise TypeError("type not implemented. %s" % sl_type)

        new_roi.sl_mode = sl_mode
        roi_key = self.hdl.add_item(new_roi, label)
        new_roi.sigRemoveRequested.connect(lambda: self.remove_roi(roi_key))
        return new_roi

    def remove_roi(self, roi_key):
        self.hdl.remove_item(roi_key)

    def compute_saxs1d(self, method="percentile", cutoff=3.0, num=180):
        t0 = time.perf_counter()
        saxs_pack = generate_partition(
            "q", self.mask, self.qmap["q"], num, style="linear"
        )
        qlist, partition = saxs_pack["v_list"], saxs_pack["partition"]
        saxs1d, zero_loc = outlier_removal_with_saxs(
            qlist, partition, self.dset.scat, method=method, cutoff=cutoff
        )
        t1 = time.perf_counter()
        logger.info(
            "outlier removal with azimuthal average finished in %f seconds" % (t1 - t0)
        )
        return saxs1d, zero_loc

    def compute_partition(self, mode="q-phi", **kwargs):
        if mode == "eq-ephi":
            ellipse_param = find_ellipse_parameters(self.mask)
            rho, phi = compute_ellipse_gradient(
                self.qmap["y"], self.qmap["x"], ellipse_param
            )
            q_rev = self.qmap["q"].copy()
            phi_rev = self.qmap["phi"].copy()
            self.qmap["q"] = rho
            self.qmap["phi"] = phi
            mode = "q-phi"

        map_names = mode.split("-")
        logger.info(f"compute partition with mode {mode}: map_names {map_names}")
        t0 = time.perf_counter()
        flag = self.compute_partition_general(map_names=map_names, **kwargs)
        t1 = time.perf_counter()
        logger.info("compute partition finished in %f seconds" % (t1 - t0))

        if mode == "eq-ephi":
            # copy back
            self.qmap["q"] = q_rev
            self.qmap["phi"] = phi_rev

        return flag

    def compute_partition_general(
        self,
        map_names=("q", "phi"),
        dq_num=10,
        sq_num=100,
        style="linear",
        dp_num=36,
        sp_num=360,
        phi_offset=0.0,
        symmetry_fold=1,
    ):
        if self.dset is None:
            return

        # assert map_names in (("q", "phi"), ("x", "y"))
        name0, name1 = map_names
        #  generate dynamic partition
        pack_dq = generate_partition(
            name0, self.mask, self.qmap[name0], dq_num, style=style, phi_offset=None
        )
        pack_dp = generate_partition(
            name1,
            self.mask,
            self.qmap[name1],
            dp_num,
            style=style,
            phi_offset=phi_offset,
            symmetry_fold=symmetry_fold,
        )
        dynamic_map = combine_partitions(pack_dq, pack_dp, prefix="dynamic")

        # generate static partition
        pack_sq = generate_partition(
            name0, self.mask, self.qmap[name0], sq_num, style=style, phi_offset=None
        )
        pack_sp = generate_partition(
            name1,
            self.mask,
            self.qmap[name1],
            sp_num,
            style=style,
            phi_offset=phi_offset,
            symmetry_fold=symmetry_fold,
        )
        static_map = combine_partitions(pack_sq, pack_sp, prefix="static")

        # dump result to file;
        self.dset.update_partitions(
            dynamic_map["dynamic_roi_map"], static_map["static_roi_map"]
        )
        self.hdl.setCurrentIndex(3)

        flag_consistency = check_consistency(
            dynamic_map["dynamic_roi_map"], static_map["static_roi_map"], self.mask
        )
        logger.info("dqmap/sqmap consistency check: {}".format(flag_consistency))

        center = self.get_center("xy")
        partition = {
            "beam_center_x": center[0],
            "beam_center_y": center[1],
            "pixel_size": self.dset.metadata["pixel_size"],
            "mask": self.mask,
            "blemish": self.mask_kernel.blemish,
            "energy": self.dset.metadata["energy"],
            "detector_distance": self.dset.metadata["detector_distance"],
            "map_names": list(map_names),
            "map_units": [self.qmap_unit[name0], self.qmap_unit[name1]],
            "source_file": os.path.realpath(self.dset.fname),
        }
        partition.update(dynamic_map)
        partition.update(static_map)

        self.new_partition = partition
        return partition

    def update_parameters(self, new_metadata=None):
        self.dset.update_metadata(new_metadata)
        self.qmap, self.qmap_unit, _labels = self.dset.compute_qmap()
        self.mask_kernel.update_qmap(self.qmap)

    def get_center(self, mode="xy"):
        if self.dset is None:
            return (None, None)
        else:
            assert mode in ("xy", "vh"), "mode must be either 'xy' or 'vh'"
            center = self.dset.get_center(mode=mode)
            return center

    def goto_max(self):
        center_vh = self.dset.find_maximal_intensity_center()
        self.dset.set_center_vh(center_vh)
        return center_vh


def test01():
    fname = "../data/H187_D100_att0_Rq0_00001_0001-100000.hdf"
    sm = SimpleMask()
    sm.read_data(fname)


if __name__ == "__main__":
    test01()
