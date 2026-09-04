# Copyright © UChicago Argonne LLC
# See LICENSE file for details
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
    generate_groupindex_partitions,
    generate_partition,
    hash_numpy_dict,
    optimize_integer_array,
)
from .rasterize import (
    RoiPolygon,
    circle_vertices,
    ellipse_vertices,
    group_index_map,
    group_masks,
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
        self._partition_kwargs = None
        # tracks which of mask_parameter/mask_draw most recently produced valid
        # groups, so use_groupindex_for_subpartition can pick "whichever was
        # generated later" when both are currently valid (see get_active_group_source)
        self._group_eval_seq = {}
        self._group_eval_counter = 0

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
        if target in ("mask_parameter", "mask_draw"):
            if self.mask_kernel.workers[target].num_groups:
                self._group_eval_counter += 1
                self._group_eval_seq[target] = self._group_eval_counter
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

    def get_parameter_group_count(self):
        """Number of constraint groups from the last-evaluated parametrization mask
        (see MaskParameter.group_index_map), or 0 if none is available yet."""
        if self.mask_kernel is None:
            return 0
        return self.mask_kernel.workers["mask_parameter"].num_groups

    def get_draw_group_count(self):
        """Number of tracked inclusive-draw groups from the last-evaluated draw
        mask (see MaskDraw.group_index_map), or 0 if none is available yet."""
        if self.mask_kernel is None:
            return 0
        return self.mask_kernel.workers["mask_draw"].num_groups

    def get_active_group_source(self):
        """Which of "mask_parameter"/"mask_draw" use_groupindex_for_subpartition
        should use: whichever currently has valid groups; if both do, whichever was
        (re-)evaluated more recently. None if neither has valid groups."""
        if self.mask_kernel is None:
            return None
        candidates = [
            key
            for key in ("mask_parameter", "mask_draw")
            if self.mask_kernel.workers[key].num_groups
        ]
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        return max(candidates, key=lambda key: self._group_eval_seq.get(key, 0))

    def get_active_group_count(self):
        """num_groups of get_active_group_source()'s worker, or 0 if none."""
        source = self.get_active_group_source()
        if source is None:
            return 0
        return self.mask_kernel.workers[source].num_groups

    def get_pts_with_similar_intensity(self, cen=None, radius=50, variation=50):
        return self.dset.get_pts_with_similar_intensity(cen, radius, variation)

    # ------------------------------------------------------------------ ROIs
    def clear_rois(self):
        self.draw_rois = []

    def add_polygon(self, vertices, mode="exclusive", group_index=0):
        self.draw_rois.append(
            RoiPolygon(np.asarray(vertices, dtype=float), mode, group_index)
        )

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

    def evaluate_draw_group_index_map(self):
        """Label each pixel with the group_index of the tracked inclusive draw
        ROI that covers it (see rasterize.group_index_map); 0 = untracked."""
        if self.dset is None:
            return None
        return group_index_map(self.dset.shape, self.draw_rois)

    def evaluate_draw_group_masks(self):
        """Per-group raw filled masks for tracked inclusive draw ROIs, keyed by
        group_index (see rasterize.group_masks); used by MaskDraw.find_overlaps."""
        if self.dset is None:
            return None
        return group_masks(self.dset.shape, self.draw_rois)

    def evaluate_draw(self):
        """Evaluate the 'mask_draw' worker from the current draw ROIs."""
        keep = self.evaluate_draw_mask()
        gmap = self.evaluate_draw_group_index_map()
        row_masks = self.evaluate_draw_group_masks()
        return self.mask_evaluate(
            "mask_draw",
            arr=np.logical_not(keep),
            group_index_map=gmap,
            row_masks=row_masks,
        )

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
        self._partition_kwargs = {"mode": mode, **kwargs}
        is_ellipse = mode == "eq-ephi"
        if is_ellipse:
            ellipse_param = find_ellipse_parameters(self.mask)
            rho, phi = compute_ellipse_gradient(
                self.qmap["y"], self.qmap["x"], ellipse_param
            )
            if kwargs.get("use_groupindex_for_subpartition"):
                # Refit the ellipse independently per pixel-group instead of
                # reusing this one whole-mask fit for every group — both the
                # radial (dq/sq) and angular (dp/sp) sub-partitions are built
                # per group (see compute_partition_general), so both rho and
                # phi need to come from that same per-group fit to stay
                # geometrically consistent within a group.
                rho, phi = self._per_group_ellipse_geometry(rho, phi)
            q_rev = self.qmap["q"].copy()
            phi_rev = self.qmap["phi"].copy()
            self.qmap["q"] = rho
            self.qmap["phi"] = phi
            mode = "q-phi"

        map_names = mode.split("-")
        logger.info("compute partition with mode %s: map_names %s", mode, map_names)
        t0 = time.perf_counter()
        try:
            flag = self.compute_partition_general(map_names=map_names, **kwargs)
        finally:
            if is_ellipse:
                # copy back, even if compute_partition_general raised
                self.qmap["q"] = q_rev
                self.qmap["phi"] = phi_rev
        t1 = time.perf_counter()
        logger.info("compute partition finished in %f seconds", t1 - t0)

        return flag

    def _per_group_ellipse_geometry(self, fallback_rho, fallback_phi):
        """For ellipse mode combined with group-index sub-partitioning: fit an
        independent ellipse to each pixel-group's own footprint, instead of
        reusing one whole-mask fit for every group, so each group's equivalent-q
        AND phi values reflect its own local geometry. Groups with no active
        group source, or whose own fit fails (e.g. too few pixels), fall back to
        ``fallback_rho``/``fallback_phi``'s whole-mask-fit values for that group.
        """
        source = self.get_active_group_source()
        if source is None:
            return fallback_rho, fallback_phi
        worker = self.mask_kernel.workers[source]
        effective_group_index = worker.group_index_map * self.mask
        rho = fallback_rho.copy()
        phi = fallback_phi.copy()
        for g in range(1, worker.num_groups + 1):
            sub_mask = effective_group_index == g
            if not sub_mask.any():
                continue
            ellipse_param = find_ellipse_parameters(sub_mask)
            if ellipse_param is None:
                continue
            rho_g, phi_g = compute_ellipse_gradient(
                self.qmap["y"], self.qmap["x"], ellipse_param
            )
            rho[sub_mask] = rho_g[sub_mask]
            phi[sub_mask] = phi_g[sub_mask]
        return rho, phi

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
        use_groupindex_for_subpartition=False,
    ):
        if self.dset is None:
            return None

        name0, name1 = map_names

        if use_groupindex_for_subpartition:
            source = self.get_active_group_source()
            if source is None:
                raise RuntimeError(
                    "use_groupindex_for_subpartition requested but no "
                    "parametrization or draw constraint groups are available; "
                    "evaluate/apply the parametrization mask or draw inclusive "
                    "groups first"
                )
            worker = self.mask_kernel.workers[source]
            # the true group count comes from the mask engine, not the caller's
            # dq_num, so a stale/mismatched dq_num can't silently corrupt the result
            num_groups = worker.num_groups
            overlaps = worker.find_overlaps(mask=self.mask)
            if overlaps:
                # group_index_map resolves overlaps via last-write-wins with no
                # indication anything happened; use_groupindex_for_subpartition
                # assumes non-overlapping groups, so surface it instead of silently
                # mis-assigning pixels between groups.
                details = "; ".join(
                    f"{worker.describe_constraint(i)} overlaps "
                    f"{worker.describe_constraint(j)} in {count} pixel(s)"
                    for i, j, count in overlaps
                )
                raise RuntimeError(
                    "use_groupindex_for_subpartition requires non-overlapping "
                    f"constraint groups, but found overlap(s): {details}."
                )
            effective_group_index = worker.group_index_map * self.mask
            present_groups = set(np.unique(effective_group_index).tolist()) - {0}
            empty_groups = [g for g in range(1, num_groups + 1) if g not in present_groups]
            if empty_groups:
                # combine_partitions compacts away empty bins from the roi_map but
                # leaves v_list_dim0 uncompacted, so proceeding would silently
                # misalign the two in the saved file — fail loudly instead.
                raise RuntimeError(
                    "use_groupindex_for_subpartition: constraint group-index "
                    f"{empty_groups} matched 0 pixels (after combining with the "
                    "rest of the mask). Fix or remove the corresponding "
                    "row(s)/ROI(s) before computing the partition."
                )
            # both axes are sub-partitioned independently per group, then combined
            # below exactly like the plain (non-group) path — combine_partitions
            # naturally confines each group's combined bins to their own block,
            # since a pixel's two per-group-offset axis values share that same
            # pixel's group offset.
            pack_dq, pack_sq = generate_groupindex_partitions(
                name0,
                effective_group_index,
                num_groups,
                self.qmap[name0],
                dq_num,
                sq_num,
                style=style,
            )
            pack_dp, pack_sp = generate_groupindex_partitions(
                name1,
                effective_group_index,
                num_groups,
                self.qmap[name1],
                dp_num,
                sp_num,
                style=style,
                phi_offset=phi_offset,
                symmetry_fold=symmetry_fold,
            )
        else:
            #  generate dynamic partition
            pack_dq = generate_partition(
                name0, self.mask, self.qmap[name0], dq_num, style=style, phi_offset=None
            )
            # generate static partition
            pack_sq = generate_partition(
                name0, self.mask, self.qmap[name0], sq_num, style=style, phi_offset=None
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
            pack_sp = generate_partition(
                name1,
                self.mask,
                self.qmap[name1],
                sp_num,
                style=style,
                phi_offset=phi_offset,
                symmetry_fold=symmetry_fold,
            )

        dynamic_map = combine_partitions(pack_dq, pack_dp, prefix="dynamic")
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
        # The dynamic/static partition maps are derived from self.qmap; once the
        # geometry moves, a previously-computed partition is stale, so refresh
        # it with the same settings rather than leave data_display showing a
        # partition from the discarded geometry.
        if self.new_partition is not None and self._partition_kwargs is not None:
            self.compute_partition(**self._partition_kwargs)

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
