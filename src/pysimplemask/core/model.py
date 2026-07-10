import logging
import os
import time

import h5py
import numpy as np
import tifffile

from pysimplemask import __version__

from .ellipse_util import compute_ellipse_gradient, find_ellipse_parameters
from .file_handler import get_handler
from .find_center import find_center
from .mask import MaskAssemble
from .outlier_removal import outlier_removal_adjacent_boxes, outlier_removal_with_saxs
from .partition import (
    check_consistency,
    combine_partitions,
    generate_partition,
    hash_numpy_dict,
    optimize_integer_array,
)
from .rasterize import (
    RoiPolygon,
    circle_vertices,
    ellipse_vertices,
    line_vertices,
    rasterize,
    rectangle_vertices,
)

logger = logging.getLogger(__name__)


class SimpleMaskModel(object):
    """Qt-free domain model for masking and q-partition generation.

    Usable directly from Python scripts: load data, build a mask, compute a
    partition, and save results without any GUI. The model produces plain numpy
    (``dset.data_display``, ``mask``, partition dicts) that a view can render.
    """

    def __init__(self):
        self.dset = None
        self.shape = None
        self.qmap = None
        self.qmap_unit = None
        self.mask = None
        self.mask_kernel = None
        self.new_partition = None
        self.draw_rois = []
        self.bad_pixel_set = set()

    def is_ready(self):
        return self.dset is not None

    def find_center(self, max_radius=384, beamstop_diameter=30):
        # Cap the symmetric crop near the beam: the centering signal lives there,
        # and a bounded window keeps the cross-correlation fast on large detectors.
        if self.dset is None:
            return None
        center_guess = self.get_center(mode="vh")
        t0 = time.perf_counter()
        center = find_center(
            self.dset.scat,
            mask=self.mask,
            center_guess=center_guess,
            scale="log",
            max_radius=max_radius,
        )
        logger.info("find center finished in %.3f seconds", time.perf_counter() - t0)

        if beamstop_diameter > 0:
            cy, cx = center[0], center[1]
            yy, xx = np.indices(self.shape)
            beamstop = np.hypot(yy - cy, xx - cx) < (beamstop_diameter / 2.0)
            self.mask_evaluate("mask_draw", arr=beamstop)
            self.mask_apply("mask_draw")
            n_masked = int(beamstop.sum())
            logger.info(
                "beamstop mask applied: diameter=%d px, %d pixels masked",
                beamstop_diameter,
                n_masked,
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

    # ------------------------------------------------------------------ ROIs
    def clear_rois(self):
        self.draw_rois = []

    def add_polygon(self, vertices, mode="exclusive"):
        self.draw_rois.append(RoiPolygon(np.asarray(vertices, dtype=float), mode))

    def add_circle(self, center, radius, mode="exclusive"):
        self.draw_rois.append(RoiPolygon(circle_vertices(center, radius), mode))

    def add_ellipse(self, center, axes, angle_deg=0.0, mode="exclusive"):
        self.draw_rois.append(
            RoiPolygon(ellipse_vertices(center, axes, angle_deg), mode)
        )

    def add_rectangle(self, center, size, angle_deg=0.0, mode="exclusive"):
        self.draw_rois.append(
            RoiPolygon(rectangle_vertices(center, size, angle_deg), mode)
        )

    def add_line(self, p0, p1, width, mode="exclusive"):
        self.draw_rois.append(RoiPolygon(line_vertices(p0, p1, width), mode))

    def set_draw_rois(self, rois):
        """Replace the current draw ROIs with a list of RoiPolygon (used by the GUI)."""
        self.draw_rois = list(rois)

    def evaluate_draw_mask(self):
        """Rasterize the current draw ROIs to a keep-mask (True = keep)."""
        if self.dset is None:
            return None
        return rasterize(self.dset.shape, self.draw_rois)

    def evaluate_draw(self):
        """Evaluate the 'mask_draw' worker from the current draw ROIs."""
        keep = self.evaluate_draw_mask()
        return self.mask_evaluate("mask_draw", arr=np.logical_not(keep))

    # ----------------------------------------------------------------- saving
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
        logger.info("Hash value of the partition: %s", hash_val)

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

    # ------------------------------------------------------------- data / qmap
    def read_data(self, fname=None, beamline="APS_8IDI", **kwargs):
        self.dset = get_handler(beamline, fname)
        if self.dset is None:
            logger.error("failed to create a dataset handler for %s", fname)
            return False

        t0 = time.perf_counter()
        self.dset.prepare_data(**kwargs)
        t1 = time.perf_counter()
        logger.info("data loaded in %.1f seconds", t1 - t0)

        self.shape = self.dset.shape
        self.mask = np.ones(self.shape, dtype=bool)

        self.qmap, self.qmap_unit, _ = self.dset.compute_qmap()
        self.mask_kernel = MaskAssemble(self.shape, self.dset.scat)
        self.mask_apply(target="default_blemish")
        self.mask_kernel.update_qmap(self.qmap)

        if getattr(self.dset, "saved_partition", None) is not None:
            p = self.dset.saved_partition
            self.dset.update_partitions(p["dynamic_roi_map"], p["static_roi_map"])
            self.new_partition = p

        return True

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
            "outlier removal with azimuthal average finished in %f seconds", t1 - t0
        )
        return saxs1d, zero_loc

    def compute_adjacent_saxs1d(self, method="percentile", cutoff=3.0, box_size=32):
        """Outlier removal by adjacent square boxes instead of q-rings."""
        t0 = time.perf_counter()
        saxs1d, zero_loc = outlier_removal_adjacent_boxes(
            self.dset.scat,
            self.mask,
            box_size=box_size,
            method=method,
            cutoff=cutoff,
        )
        logger.info(
            "adjacent-box outlier removal finished in %f seconds",
            time.perf_counter() - t0,
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
        logger.info("compute partition with mode %s: map_names %s", mode, map_names)
        t0 = time.perf_counter()
        flag = self.compute_partition_general(map_names=map_names, **kwargs)
        t1 = time.perf_counter()
        logger.info("compute partition finished in %f seconds", t1 - t0)

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
            return None

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

        self.dset.update_partitions(
            dynamic_map["dynamic_roi_map"], static_map["static_roi_map"]
        )

        flag_consistency = check_consistency(
            dynamic_map["dynamic_roi_map"], static_map["static_roi_map"], self.mask
        )
        logger.info("dqmap/sqmap consistency check: %s", flag_consistency)

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
        if mode not in ("xy", "vh"):
            raise ValueError(f"mode must be 'xy' or 'vh', got {mode!r}")
        return self.dset.get_center(mode=mode)

    def goto_max(self):
        center_vh = self.dset.find_maximal_intensity_center()
        self.dset.set_center_vh(center_vh)
        return center_vh
