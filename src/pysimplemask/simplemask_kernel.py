import h5py
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from .area_mask import MaskAssemble
from .find_center import find_center
from .pyqtgraph_mod import LineROI
from .file_reader import read_raw_file
import logging
import time
from .utils import (
    hash_numpy_dict,
    optimize_integer_array,
    generate_partition,
    combine_partitions,
    check_consistency,
)
from .outlier_removal import outlier_removal_with_saxs


pg.setConfigOptions(imageAxisOrder="row-major")


logger = logging.getLogger(__name__)


class SimpleMask(object):
    def __init__(self, pg_hdl, infobar):
        self.data_raw = None
        self.shape = None
        self.qmap = None
        self.mask = None
        self.mask_kernel = None
        self.saxs_lin = None
        self.saxs_log = None
        self.saxs_log_min = None
        self.plot_log = True

        self.new_partition = None
        self.meta = None

        self.hdl = pg_hdl
        self.infobar = infobar
        self.extent = None
        self.hdl.scene.sigMouseMoved.connect(self.show_location)
        self.bad_pixel_set = set()
        self.corr_roi = None

        self.idx_map = {
            0: "scattering",
            1: "scattering * mask",
            2: "mask",
            3: "dqmap_partition",
            4: "sqmap_partition",
            5: "preview",
        }

    def is_ready(self):
        if self.meta is None or self.data_raw is None:
            return False
        else:
            return True

    def find_center(self):
        if self.saxs_lin is None:
            return

        center_guess = (self.meta["bcy"], self.meta["bcx"])
        center = find_center(
            self.saxs_lin, mask=self.mask, center_guess=center_guess, scale="log"
        )
        return center

    def mask_evaluate(self, target, **kwargs):
        msg = self.mask_kernel.evaluate(target, **kwargs)
        # preview the mask
        mask = self.mask_kernel.get_one_mask(target)
        self.data_raw[5][:, :] = mask
        return msg

    def mask_action(self, action="undo"):
        self.mask_kernel.redo_undo(action=action)
        self.mask_apply()

    def mask_apply(self, target=None):
        if target == "default_mask":
            self.mask = self.mask_kernel.apply_default_mask()
        else:
            # if target is None, apply will return the current mask
            self.mask = self.mask_kernel.apply(target)

        self.data_raw[1] = self.saxs_log * self.mask
        self.data_raw[2] = self.mask

        if self.plot_log:
            min_mask = (self.saxs_lin > 0) * self.mask
            nz_min = np.min(self.saxs_lin[min_mask > 0])
            self.data_raw[1][np.logical_not(min_mask)] = np.log10(nz_min)
        else:
            lin_min = np.min(self.saxs_lin[self.mask > 0])
            self.data_raw[1][np.logical_not(self.mask)] = lin_min

        self.hdl.setImage(self.data_raw)

    def get_pts_with_similar_intensity(self, cen=None, radius=50, variation=50):
        vmin = max(0, int(cen[0] - radius))
        vmax = min(self.shape[0], int(cen[0] + radius))

        hmin = max(0, int(cen[1] - radius))
        hmax = min(self.shape[1], int(cen[1] + radius))
        crop = self.saxs_lin[vmin:vmax, hmin:hmax]
        val = self.saxs_lin[cen]
        idx = np.abs(crop - val) <= variation / 100.0 * val
        pos = np.array(np.nonzero(idx))
        pos[0] += vmin
        pos[1] += hmin
        pos = np.roll(pos, shift=1, axis=0)
        return pos.T

    def save_partition(self, save_fname, root="/qmap", version="0.1"):
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
            group_handle.attrs["version"] = "0.1"

    # generate 2d saxs
    def read_data(self, fname=None, **kwargs):
        reader = read_raw_file(fname)
        if reader is None:
            logger.error(f"failed to create a dataset handler for {fname}")
            return False
        t0 = time.perf_counter()
        saxs = reader.get_scattering(**kwargs)
        t1 = time.perf_counter()
        logger.info(f"data loaded in {t1 - t0: .1f} seconds")

        if saxs is None:
            logger.error("failed to read scattering signal from the dataset.")
            return False

        self.reader = reader
        self.meta = reader.load_meta()

        self.shape = saxs.shape
        self.mask = np.ones(saxs.shape, dtype=bool)

        saxs_nonzero = saxs[saxs > 0]
        # use percentile instead of min to be robust
        self.saxs_lin_min = np.percentile(saxs_nonzero, 1)
        self.saxs_log_min = np.log10(self.saxs_lin_min)

        self.saxs_lin = saxs.astype(np.float32)
        self.min_val = np.min(saxs[saxs > 0])
        self.saxs_log = np.log10(saxs + self.min_val)

        self.qmap, self.qmap_unit = self.compute_qmap()
        num_qmaps = len(self.qmap)
        self.data_raw = np.zeros(shape=(6 + num_qmaps, *saxs.shape))

        for offset, val in enumerate(self.qmap.values()):
            self.data_raw[6 + offset] = val

        self.mask_kernel = MaskAssemble(self.shape, self.saxs_lin)
        # self.mask_kernel.update_qmap(self.qmap)
        self.extent = self.compute_extent()

        # self.meta['saxs'] = saxs
        self.data_raw[0] = self.saxs_log
        self.data_raw[1] = self.saxs_log * self.mask
        self.data_raw[2] = self.mask

        self.mask_apply(target="default_mask")
        self.mask_kernel.update_qmap(self.qmap)

        return True

    def compute_qmap(self):
        k0 = 2 * np.pi / (12.398 / self.meta["energy"])
        v = np.arange(self.shape[0], dtype=np.uint32) - self.meta["bcy"]
        h = np.arange(self.shape[1], dtype=np.uint32) - self.meta["bcx"]
        vg, hg = np.meshgrid(v, h, indexing="ij")

        r = np.hypot(vg, hg) * self.meta["pix_dim"]
        phi = np.arctan2(vg, hg) * (-1)
        alpha = np.arctan(r / self.meta["det_dist"])

        qr = np.sin(alpha) * k0
        # qr = 2 * np.sin(alpha / 2) * k0
        qx = qr * np.cos(phi)
        qy = qr * np.sin(phi)
        phi = np.rad2deg(phi)

        # keep phi and q as np.float64 to keep the precision.
        qmap = {
            "phi": phi,
            "alpha": alpha.astype(np.float32),
            "q": qr,
            "qx": qx.astype(np.float32),
            "qy": qy.astype(np.float32),
            "x": hg,
            "y": vg,
        }

        qmap_unit = {
            "phi": "deg",
            "alpha": "deg",
            "q": "Å⁻¹",
            "qx": "Å⁻¹",
            "qy": "Å⁻¹",
            "x": "pixel",
            "y": "pixel",
        }

        return qmap, qmap_unit

    def get_qp_value(self, x, y):
        x = int(x)
        y = int(y)
        shape = self.qmap["q"].shape
        if 0 <= x < shape[1] and 0 <= y < shape[0]:
            return self.qmap["q"][y, x], self.qmap["phi"][y, x]
        else:
            return None, None

    def compute_extent(self):
        k0 = 2 * np.pi / (12.3980 / self.meta["energy"])
        x_range = np.array([0, self.shape[1]]) - self.meta["bcx"]
        y_range = np.array([-self.shape[0], 0]) + self.meta["bcy"]
        x_range = x_range * self.meta["pix_dim"] / self.meta["det_dist"] * k0
        y_range = y_range * self.meta["pix_dim"] / self.meta["det_dist"] * k0
        # the extent for matplotlib imshow is:
        # self._extent = xmin, xmax, ymin, ymax = extent
        # convert to a tuple of 4 elements;

        return (*x_range, *y_range)

    def show_location(self, pos):

        if not self.hdl.scene.itemsBoundingRect().contains(pos) or self.shape is None:
            return

        shape = self.shape
        mouse_point = self.hdl.getView().mapSceneToView(pos)
        col = int(mouse_point.x())
        row = int(mouse_point.y())

        if col < 0 or col >= shape[1]:
            return
        if row < 0 or row >= shape[0]:
            return

        qx = self.qmap["qx"][row, col]
        qy = self.qmap["qy"][row, col]
        phi = self.qmap["phi"][row, col]
        val = self.data_raw[self.hdl.currentIndex][row, col]

        # msg = f'{self.idx_map[self.hdl.currentIndex]}: ' + \
        msg = (
            f"[x={col:4d}, y={row:4d}, "
            + f"qx={qx:.3e}Å⁻¹, qy={qy:.3e}Å⁻¹, phi={phi:.1f}deg], "
            + f"val={val:.03e}"
        )

        self.infobar.clear()
        self.infobar.setText(msg)

        return None

    def show_saxs(
        self,
        cmap="jet",
        log=True,
        invert=False,
        plot_center=True,
        plot_index=0,
        **kwargs,
    ):
        if self.meta is None or self.data_raw is None:
            return
        # self.hdl.reset_limits()
        self.hdl.clear()
        # self.data = np.copy(self.data_raw)

        center = (self.meta["bcx"], self.meta["bcy"])

        self.plot_log = log
        if not log:
            self.data_raw[0] = self.saxs_lin
        else:
            self.data_raw[0] = self.saxs_log

        self.hdl.setImage(self.data_raw)
        self.hdl.adjust_viewbox()
        self.hdl.set_colormap(cmap)

        # plot center
        if plot_center:
            t = pg.ScatterPlotItem()
            t.addPoints(x=[center[0]], y=[center[1]], symbol="+", size=15)
            self.hdl.add_item(t, label="center")

        self.hdl.setCurrentIndex(plot_index)

        return

    def apply_drawing(self):
        if self.meta is None or self.data_raw is None:
            return

        # ones = np.ones(self.data_raw[0].shape, dtype=np.bool)
        shape = self.data_raw[0].shape
        ones = np.ones((shape[0] + 1, shape[1] + 1), dtype=bool)
        mask_n = np.zeros_like(ones, dtype=bool)
        mask_e = np.zeros_like(mask_n)
        mask_i = np.zeros_like(mask_n)

        for k, x in self.hdl.roi.items():
            if not k.startswith("roi_"):
                continue

            mask_temp = np.zeros_like(ones, dtype=bool)
            # return slice and transfrom
            sl, _ = x.getArraySlice(self.data_raw[1], self.hdl.imageItem)
            y = x.getArrayRegion(ones, self.hdl.imageItem)

            # sometimes the roi size returned from getArraySlice and
            # getArrayRegion are different;
            nz_idx = np.nonzero(y)

            h_beg = np.min(nz_idx[1])
            h_end = np.max(nz_idx[1]) + 1

            v_beg = np.min(nz_idx[0])
            v_end = np.max(nz_idx[0]) + 1

            sl_v = slice(sl[0].start, sl[0].start + v_end - v_beg)
            sl_h = slice(sl[1].start, sl[1].start + h_end - h_beg)
            mask_temp[sl_v, sl_h] = y[v_beg:v_end, h_beg:h_end]

            if x.sl_mode == "exclusive":
                mask_e[mask_temp] = 1
            elif x.sl_mode == "inclusive":
                mask_i[mask_temp] = 1

        self.hdl.remove_rois(filter_str="roi_")

        if np.sum(mask_i) == 0:
            mask_i = 1

        mask_p = np.logical_not(mask_e) * mask_i
        mask_p = mask_p[:-1, :-1]

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

        cen = (self.meta["bcx"], self.meta["bcy"])
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
            new_roi = pg.EllipseROI(cen, [60, 80], **kwargs)
            # add scale handle
            new_roi.addScaleHandle(
                [0.5, 0],
                [0.5, 1],
            )
            new_roi.addScaleHandle([0.5, 1], [0.5, 0])
            new_roi.addScaleHandle([0, 0.5], [1, 0.5])
            new_roi.addScaleHandle([1, 0.5], [0, 0.5])

        elif sl_type == "Circle":
            if second_point is not None:
                radius = np.sqrt(
                    (second_point[1] - cen[1]) ** 2 + (second_point[0] - cen[0]) ** 2
                )
            new_roi = pg.CircleROI(
                pos=[cen[0] - radius, cen[1] - radius], radius=radius, **kwargs
            )

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
            new_roi = pg.RectROI(cen, [30, 150], **kwargs)
            new_roi.addScaleHandle([0, 0], [1, 1])
            # new_roi.addRotateHandle([0, 1], [0.5, 0.5])

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
            qlist, partition, self.saxs_lin, method=method, cutoff=cutoff
        )
        t1 = time.perf_counter()
        logger.info(
            "outlier removal with azimuthal average finished in %f seconds" % (t1 - t0)
        )
        return saxs1d, zero_loc

    def compute_partition(self, mode="q-phi", **kwargs):
        map_names = {
            "q-phi": ("q", "phi"),
            "xy-mesh": ("x", "y"),
        }[mode]
        t0 = time.perf_counter()
        flag = self.compute_partition_general(map_names=map_names, **kwargs)
        t1 = time.perf_counter()
        logger.info("compute partition finished in %f seconds" % (t1 - t0))
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
        if self.meta is None or self.data_raw is None:
            return

        assert map_names in (("q", "phi"), ("x", "y"))
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
        self.data_raw[3] = dynamic_map["dynamic_roi_map"]
        self.data_raw[4] = static_map["static_roi_map"]
        self.hdl.setCurrentIndex(3)

        flag_consistency = check_consistency(
            dynamic_map["dynamic_roi_map"], static_map["static_roi_map"], self.mask
        )
        logger.info("dqmap/sqmap consistency check: {}".format(flag_consistency))

        partition = {
            "beam_center_x": self.meta["bcx"],
            "beam_center_y": self.meta["bcy"],
            "pixel_size": self.meta["pix_dim"],
            "mask": self.mask,
            "energy": self.meta["energy"],
            "detector_distance": self.meta["det_dist"],
            "map_names": list(map_names),
            "map_units": [self.qmap_unit[name0], self.qmap_unit[name1]],
        }
        partition.update(dynamic_map)
        partition.update(static_map)

        self.new_partition = partition
        return partition

    def update_parameters(self, val):
        assert len(val) == 5
        for idx, key in enumerate(["bcx", "bcy", "energy", "pix_dim", "det_dist"]):
            self.meta[key] = val[idx]
        self.qmap, self.qmap_unit = self.compute_qmap()

        for offset, val in enumerate(self.qmap.values()):
            self.data_raw[6 + offset] = val

        self.mask_kernel.update_qmap(self.qmap)

    def get_parameters(self):
        val = []
        for key in ["bcx", "bcy", "energy", "pix_dim", "det_dist"]:
            val.append(self.meta[key])
        return val

    def set_corr_roi(self, roi):
        self.corr_roi = roi

    def perform_correlation(self, angle_list):
        pass


def test01():
    fname = "../data/H187_D100_att0_Rq0_00001_0001-100000.hdf"
    sm = SimpleMask()
    sm.read_data(fname)


if __name__ == "__main__":
    test01()
