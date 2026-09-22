# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""One-page summary report (PDF or PNG) of the hdf→qmap pipeline."""

import logging
import os
from typing import Any, Dict, Optional, Tuple, Union

import h5py
import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

# Number of colors in one tab20 cycle.
_TAB20_N = 20

# Maximum number of partition bins for the cycling colormap.
_CMAP_MAX_BINS = 1000


def _tab20_color(i):
    """Return the i-th tab20 color (cycling)."""
    return matplotlib.colormaps["tab20"](i % _TAB20_N)


def _qmap_cmap(n_bins):
    """ListedColormap: index 0 = white (masked), 1..n_bins = tab20 cycling."""
    colors = [(1.0, 1.0, 1.0, 1.0)]  # index 0 → white (masked)
    for i in range(max(1, n_bins)):
        colors.append(_tab20_color(i))
    return ListedColormap(colors)


def _log_image(scat, mask=None):
    """Return a log10 image of ``scat`` with optional ``mask`` applied.

    Masked-out and non-positive pixels become NaN so matplotlib renders
    them as transparent / background colour.
    """
    img = scat.astype(np.float64).copy()
    if mask is not None and mask.shape == img.shape:
        img[mask == 0] = 0
    out = np.full_like(img, np.nan)
    positive = img > 0
    out[positive] = np.log10(img[positive])
    return out


def _save_figure(fig, output_path: Union[str, os.PathLike], dpi: int = 150) -> str:
    """Save figure to PDF or raster image (PNG, etc.) based on file extension."""
    out_str = str(output_path)
    ext = os.path.splitext(out_str)[1].lower()
    abs_path = os.path.abspath(out_str)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    if ext == ".pdf":
        with PdfPages(abs_path) as pdf:
            pdf.savefig(fig, dpi=dpi)
    else:
        fig.savefig(abs_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Report saved: %s", abs_path)
    return abs_path


def _format_float(val: Any, fmt: str = ".4g", unit: str = "") -> str:
    if val is None:
        return "n/a"
    try:
        fval = float(val)
        return f"{fval:{fmt}}{unit}"
    except (ValueError, TypeError):
        return str(val)


def _render_report_figure(
    fname: str,
    shape: Tuple[int, int],
    center_xy: Tuple[Optional[float], Optional[float]],
    center_vh: Tuple[Optional[float], Optional[float]],
    meta: Dict[str, Any],
    mask: Optional[np.ndarray],
    blemish: Optional[np.ndarray],
    scat: Optional[np.ndarray],
    partition: Optional[Dict[str, Any]],
    crop_half_size: int = 100,
    params: Optional[Dict[str, Any]] = None,
    orientation: str = "landscape",
) -> plt.Figure:
    """Construct the 6-panel summary figure with header and footer parameter box."""
    if scat is not None and scat.ndim == 3:
        scat = scat[0]

    is_landscape = orientation.lower() == "landscape"
    if is_landscape:
        fig = plt.figure(figsize=(11, 8.5))
    else:
        fig = plt.figure(figsize=(8.5, 11))

    # ── Title / metadata header ───────────────────────────────────────────────
    cx, cy = center_xy
    if cx is not None and cy is not None:
        center_str = f"center ({cx:.1f}, {cy:.1f}) px"
    else:
        center_str = "center n/a"

    title = f"{fname}   |   shape {shape[1]}×{shape[0]}   |   {center_str}"

    if mask is not None:
        n_masked = int((~mask.astype(bool)).sum())
        total_px = mask.size
        pct_masked = (n_masked / total_px * 100) if total_px > 0 else 0.0
    else:
        n_masked = 0
        pct_masked = 0.0

    energy_str = _format_float(meta.get("energy"), ".4g", " keV")
    dist_str = _format_float(meta.get("detector_distance"), ".4g", " m")
    pix_val = meta.get("pixel_size")
    if pix_val is not None:
        try:
            pix_str = f"{float(pix_val) * 1e6:.1f} µm"
        except (ValueError, TypeError):
            pix_str = str(pix_val)
    else:
        pix_str = "n/a"

    sub = (
        f"E = {energy_str}   "
        f"dist = {dist_str}   "
        f"pix = {pix_str}   "
        f"masked = {pct_masked:.1f}%"
    )
    if is_landscape:
        fig.text(0.5, 0.980, title, ha="center", va="top", fontsize=9, fontweight="bold")
        fig.text(0.5, 0.955, sub, ha="center", va="top", fontsize=7.5, color="#444444")
    else:
        fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=8, fontweight="bold")
        fig.text(0.5, 0.945, sub, ha="center", va="top", fontsize=7, color="#444444")

    # ── Grid layout ───────────────────────────────────────────────────────────
    if is_landscape:
        gs = GridSpec(
            2, 3, figure=fig,
            top=0.915, bottom=0.175, left=0.04, right=0.98,
            hspace=0.25, wspace=0.20,
            height_ratios=[1.0, 1.0],
        )
        box_rect = [0.03, 0.015, 0.94, 0.14]
        label_kw = dict(fontsize=7.0, va="center", fontfamily="monospace")
    else:
        gs = GridSpec(
            2, 3, figure=fig,
            top=0.93, bottom=0.18, left=0.05, right=0.97,
            hspace=0.35, wspace=0.30,
            height_ratios=[1.4, 1.0],
        )
        box_rect = [0.03, 0.01, 0.94, 0.155]
        label_kw = dict(fontsize=6.5, va="center", fontfamily="monospace")


    def _colorbar(im, ax):
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

    # ── Row 1, panel 1: scattering + blemish (log) ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    if scat is not None:
        img1 = _log_image(scat, mask=blemish)
        im1 = ax1.imshow(img1, cmap="jet", origin="upper", aspect="equal")
        title1 = "Scattering + blemish (log)" if blemish is not None else "Raw scattering (log)"
        ax1.set_title(title1, fontsize=7)
        ax1.axis("off")
        _colorbar(im1, ax1)
    elif blemish is not None:
        im1 = ax1.imshow(blemish.astype(np.uint8), cmap="gray", vmin=0, vmax=1, origin="upper", aspect="equal")
        ax1.set_title("Blemish mask (white = valid)", fontsize=7)
        ax1.axis("off")
    else:
        ax1.text(0.5, 0.5, "Raw scattering / blemish\nnot available", ha="center", va="center", transform=ax1.transAxes, fontsize=8)
        ax1.set_title("Scattering + blemish", fontsize=7)
        ax1.axis("off")

    # ── Row 1, panel 2: scattering × user mask (log) ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if scat is not None:
        img2 = _log_image(scat, mask=mask)
        im2 = ax2.imshow(img2, cmap="jet", origin="upper", aspect="equal")
        ax2.set_title("Scattering × mask (log)", fontsize=7)
        ax2.axis("off")
        _colorbar(im2, ax2)
    else:
        ax2.text(0.5, 0.5, "Raw scattering\nnot available", ha="center", va="center", transform=ax2.transAxes, fontsize=8)
        ax2.set_title("Scattering × mask (log)", fontsize=7)
        ax2.axis("off")

    # ── Row 1, panel 3: combined mask ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    if mask is not None:
        ax3.imshow(
            mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1,
            origin="upper", aspect="equal",
        )
        ax3.set_title("Mask (white = valid)", fontsize=7)
    else:
        ax3.text(0.5, 0.5, "No mask available", ha="center", va="center", transform=ax3.transAxes, fontsize=8)
        ax3.set_title("Mask", fontsize=7)
    ax3.axis("off")

    # ── Row 2, panel 1: beam-center crop ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    cy_vh, cx_vh = center_vh if center_vh is not None else (None, None)
    H, W = shape
    if cy_vh is not None and cx_vh is not None and 0 <= cy_vh < H and 0 <= cx_vh < W:
        r0 = max(0, int(cy_vh) - crop_half_size)
        r1 = min(H, int(cy_vh) + crop_half_size)
        c0 = max(0, int(cx_vh) - crop_half_size)
        c1 = min(W, int(cx_vh) + crop_half_size)
        if scat is not None:
            crop = _log_image(scat, mask=mask)[r0:r1, c0:c1]
            im4 = ax4.imshow(crop, cmap="jet", origin="upper", aspect="equal")
            _colorbar(im4, ax4)
        elif mask is not None:
            crop = mask.astype(np.uint8)[r0:r1, c0:c1]
            ax4.imshow(crop, cmap="gray", vmin=0, vmax=1, origin="upper", aspect="equal")
        local_cy = float(cy_vh) - r0
        local_cx = float(cx_vh) - c0
        # Intertwined black-and-white dashed crosshair for high visibility
        ax4.axhline(local_cy, color="black", linewidth=0.9, linestyle="-", zorder=3)
        ax4.axhline(local_cy, color="white", linewidth=0.9, linestyle=(0, (4, 4)), zorder=4)
        ax4.axvline(local_cx, color="black", linewidth=0.9, linestyle="-", zorder=3)
        ax4.axvline(local_cx, color="white", linewidth=0.9, linestyle=(0, (4, 4)), zorder=4)
        ax4.set_title(f"Beam center crop  ({cx_vh:.0f}, {cy_vh:.0f})", fontsize=7)
    else:
        msg = "Center outside frame" if (cy_vh is not None and cx_vh is not None) else "Center not specified"
        ax4.text(
            0.5, 0.5, msg,
            ha="center", va="center", transform=ax4.transAxes, fontsize=8,
        )
        ax4.set_title("Beam center crop", fontsize=7)
    ax4.axis("off")

    # ── Row 2, panels 2 & 3: qmaps ───────────────────────────────────────────
    map_names = partition.get("map_names", ["q", "phi"]) if partition else ["q", "phi"]

    def _qmap_panel(ax, data, label):
        if data is None:
            ax.text(
                0.5, 0.5, "No partition computed",
                ha="center", va="center", transform=ax.transAxes, fontsize=8,
            )
        else:
            n_bins = int(data.max())
            cmap = _qmap_cmap(n_bins)
            ax.imshow(
                data, cmap=cmap, vmin=0, vmax=max(1, n_bins),
                origin="upper", aspect="equal", interpolation="nearest",
            )
        ax.set_title(label, fontsize=7)
        ax.axis("off")

    if partition and partition.get("static_roi_map") is not None and partition.get("dynamic_roi_map") is not None:
        sq_n = partition.get("static_num_pts")
        dq_n = partition.get("dynamic_num_pts")
        name0 = map_names[0] if len(map_names) > 0 else "dim0"
        name1 = map_names[1] if len(map_names) > 1 else "dim1"
        sq_dim_str = f"{sq_n[0]} {name0} × {sq_n[1]} {name1} bins" if sq_n is not None and len(sq_n) >= 2 else ""
        dq_dim_str = f"{dq_n[0]} {name0} × {dq_n[1]} {name1} bins" if dq_n is not None and len(dq_n) >= 2 else ""
        sq_title = f"Static qmap  ({name0}-{name1})\n{sq_dim_str}".strip()
        dq_title = f"Dynamic qmap  ({name0}-{name1})\n{dq_dim_str}".strip()
        sq_data = partition["static_roi_map"]
        dq_data = partition["dynamic_roi_map"]
    else:
        name0 = map_names[0] if len(map_names) > 0 else "dim0"
        name1 = map_names[1] if len(map_names) > 1 else "dim1"
        sq_title = f"Static qmap  ({name0}-{name1})"
        dq_title = f"Dynamic qmap  ({name0}-{name1})"
        sq_data = dq_data = None

    _qmap_panel(fig.add_subplot(gs[1, 1]), sq_data, sq_title)
    _qmap_panel(fig.add_subplot(gs[1, 2]), dq_data, dq_title)

    # ── Processing parameters footer ─────────────────────────────────────────
    p = params or {}

    # Row 1 — Data loading
    source_name = p.get("source_file", fname)
    if "beamline" in p or "begin_idx" in p:
        beamline = p.get("beamline", "n/a")
        begin_idx = p.get("begin_idx", "n/a")
        num_frames = p.get("num_frames", "n/a")
        nf_str = "all" if num_frames == 0 else ("subset" if num_frames == -1 else str(num_frames))
        data_line = (
            f"Data:      beamline={beamline}   |   begin_idx={begin_idx}"
            f"   |   num_frames={nf_str}"
            f"   |   source: {source_name}"
        )
    else:
        data_line = f"Data:      source: {source_name}"

    # Row 2 — Beam center / beamstop
    pos_str = f"(col={cx_vh:.1f}, row={cy_vh:.1f}) px" if cx_vh is not None and cy_vh is not None else "n/a"
    if "find_center" in p or "beamstop_diameter" in p:
        find_center_used = p.get("find_center", "n/a")
        bs_diam = p.get("beamstop_diameter", "n/a")
        fc_str = "yes" if find_center_used is True else ("no" if find_center_used is False else str(find_center_used))
        center_line = (
            f"Center:    find_center={fc_str}"
            f"   |   position {pos_str}"
            f"   |   beamstop_diameter={bs_diam} px"
            f"   |   max_radius={p.get('max_radius', 'n/a')} px"
        )
    else:
        center_line = f"Center:    position {pos_str}"

    # Row 3 — Mask
    thr = p.get("threshold_high")
    thr_str = f"{thr:.4g}" if thr is not None else "none"
    blemish_str = os.path.basename(str(p["blemish"])) if p.get("blemish") else ("stored" if blemish is not None else "none")
    mask_line = (
        f"Mask:      blemish={blemish_str}"
        f"   |   threshold_high={thr_str}"
        f"   |   masked={n_masked:,} px ({pct_masked:.2f}%)"
    )

    # Row 4 — Partition
    if partition and "map_names" in partition and len(partition["map_names"]) >= 2:
        mode_def = f"{partition['map_names'][0]}-{partition['map_names'][1]}"
    else:
        mode_def = "n/a"
    mode = p.get("mode", mode_def)
    if p.get("use_groupindex_for_subpartition"):
        mode = f"{mode} (group-index sub-partition)"
    if partition and partition.get("dynamic_num_pts") is not None and partition.get("static_num_pts") is not None:
        dq_n = partition["dynamic_num_pts"]
        sq_n = partition["static_num_pts"]
        part_bins = f"dynamic {dq_n[0]}×{dq_n[1]}   |   static {sq_n[0]}×{sq_n[1]}"
    else:
        dq_num, sq_num = p.get("dq_num", "n/a"), p.get("sq_num", "n/a")
        dp_num, sp_num = p.get("dp_num", "n/a"), p.get("sp_num", "n/a")
        part_bins = f"dynamic {dq_num}×{dp_num}   |   static {sq_num}×{sp_num}"

    part_line = (
        f"Partition: mode={mode}   |   {part_bins}"
        f"   |   phi_offset={p.get('phi_offset', 'n/a')}°"
        f"   |   symmetry_fold={p.get('symmetry_fold', 'n/a')}"
        f"   |   style={p.get('style', 'n/a')}"
    )

    # Draw the four lines in a light-grey box
    box_ax = fig.add_axes(box_rect)
    box_ax.set_xlim(0, 1)
    box_ax.set_ylim(0, 1)
    box_ax.patch.set_facecolor("#f5f5f5")
    box_ax.patch.set_edgecolor("#cccccc")
    box_ax.patch.set_linewidth(0.5)
    for spine in box_ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#cccccc")
        spine.set_linewidth(0.5)
    box_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    box_ax.text(0.01, 0.82, data_line, transform=box_ax.transAxes, **label_kw)
    box_ax.text(0.01, 0.57, center_line, transform=box_ax.transAxes, **label_kw)
    box_ax.text(0.01, 0.32, mask_line, transform=box_ax.transAxes, **label_kw)
    box_ax.text(0.01, 0.10, part_line, transform=box_ax.transAxes, **label_kw)

    return fig


def generate_report(
    model,
    output_path,
    crop_half_size=100,
    params=None,
    orientation="landscape",
):
    """Write a one-page summary report (PDF or PNG) of the qmap generation.

    Layout (landscape default, 11 × 8.5 in; or portrait, 8.5 × 11 in):
    - Header: filename, shape, beam center, energy, distance, mask %.
    - Row 1: blemish-only log scattering | scattering × mask (log) | mask.
    - Row 2: beam-center crop with crosshair | static qmap | dynamic qmap.
    - Footer: processing parameters table (4 labelled rows).

    Parameters
    ----------
    model : SimpleMaskModel
        A loaded (and optionally partitioned) model.
    output_path : str or Path
        Destination report path (.pdf, .png, etc.).
    crop_half_size : int, optional
        Half-size in pixels of the beam-center crop panel (default 100).
    params : dict, optional
        Processing parameters to list in the footer.
    orientation : {"landscape", "portrait"}, optional
        Page orientation (default: "landscape").
    """
    fname = os.path.basename(model.dset.fname) if (model.dset and getattr(model.dset, "fname", None)) else "dataset"
    meta = model.dset.metadata if model.dset else {}
    center_xy = model.get_center(mode="xy")
    center_vh = model.get_center(mode="vh")
    shape = model.shape
    mask = model.mask
    blemish = model.mask_kernel.blemish if (model.mask_kernel and hasattr(model.mask_kernel, "blemish")) else None
    scat = model.dset.scat if model.dset else None
    partition = model.new_partition

    # Default beamline into params if not present
    p = dict(params) if params else {}
    if "beamline" not in p and model.dset and hasattr(model.dset, "ftype"):
        p["beamline"] = model.dset.ftype

    fig = _render_report_figure(
        fname=fname,
        shape=shape,
        center_xy=center_xy,
        center_vh=center_vh,
        meta=meta,
        mask=mask,
        blemish=blemish,
        scat=scat,
        partition=partition,
        crop_half_size=crop_half_size,
        params=p,
        orientation=orientation,
    )
    return _save_figure(fig, output_path)


def _load_qmap_data(
    qmap_file: Union[str, os.PathLike],
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[str], Optional[np.ndarray]]:
    """Read partition arrays, scalar metadata, source_file, and embedded scattering."""
    qmap_path = str(qmap_file)
    if not os.path.exists(qmap_path):
        raise FileNotFoundError(f"qmap file not found: {qmap_path}")
    if not h5py.is_hdf5(qmap_path):
        raise ValueError(f"File is not a valid HDF5 file: {qmap_path}")

    with h5py.File(qmap_path, "r") as f:
        grp = None
        for path in ("/qmap", "/xpcs/qmap", "/"):
            if path in f:
                g = f[path]
                if any(k in g for k in ("mask", "static_roi_map", "dynamic_roi_map")):
                    grp = g
                    break
        if grp is None:
            raise ValueError(f"No qmap partition group found in {qmap_path}")

        data: Dict[str, Any] = {}
        for key in (
            "mask", "blemish", "static_roi_map", "dynamic_roi_map",
            "static_num_pts", "dynamic_num_pts",
        ):
            if key in grp:
                data[key] = grp[key][()]

        for key in ("map_names", "map_units"):
            if key in grp:
                val = grp[key][()]
                data[key] = [
                    v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    for v in val
                ]

        meta: Dict[str, Any] = {}
        for k in ("energy", "detector_distance", "pixel_size", "beam_center_x", "beam_center_y"):
            if k in grp:
                meta[k] = float(grp[k][()])

        # Fallbacks for metadata from /entry/instrument if missing
        if "beam_center_x" not in meta and "/entry/instrument/detector_1/beam_center_x" in f:
            meta["beam_center_x"] = float(f["/entry/instrument/detector_1/beam_center_x"][()])
        if "beam_center_y" not in meta and "/entry/instrument/detector_1/beam_center_y" in f:
            meta["beam_center_y"] = float(f["/entry/instrument/detector_1/beam_center_y"][()])
        if "energy" not in meta and "/entry/instrument/incident_beam/incident_energy" in f:
            meta["energy"] = float(f["/entry/instrument/incident_beam/incident_energy"][()])
        if "detector_distance" not in meta and "/entry/instrument/detector_1/distance" in f:
            meta["detector_distance"] = float(f["/entry/instrument/detector_1/distance"][()])
        if "pixel_size" not in meta and "/entry/instrument/detector_1/x_pixel_size" in f:
            meta["pixel_size"] = float(f["/entry/instrument/detector_1/x_pixel_size"][()])

        source_file = None
        if "source_file" in grp:
            val = grp["source_file"][()]
            if isinstance(val, bytes):
                source_file = val.decode("utf-8", errors="replace")
            elif isinstance(val, np.ndarray):
                source_file = str(val.item()) if val.size == 1 else str(val)
            else:
                source_file = str(val)

        embedded_scat = None
        if "/xpcs/temporal_mean/scattering_2d" in f:
            arr = f["/xpcs/temporal_mean/scattering_2d"][()]
            embedded_scat = arr[0] if arr.ndim == 3 else arr
        elif "scattering" in grp:
            embedded_scat = grp["scattering"][()]
        elif "scat" in grp:
            embedded_scat = grp["scat"][()]

        return data, meta, source_file, embedded_scat


def generate_report_from_qmap(
    qmap_file: Union[str, os.PathLike],
    output_path: Optional[Union[str, os.PathLike]] = None,
    raw_data: Optional[Union[str, os.PathLike]] = None,
    crop_half_size: int = 100,
    beamline: str = "APS_8IDI",
    begin_idx: int = 0,
    num_frames: int = -1,
    params: Optional[Dict[str, Any]] = None,
    orientation: str = "landscape",
) -> str:
    """Generate a PDF or PNG summary report directly from a qmap HDF5 file.

    Plots:
    - Row 1: blemish used (or blemish-masked log scattering) | scattering × mask (log) | mask
    - Row 2: beam-center crop with crosshair | static qmap | dynamic qmap
    - Header & footer with metadata, beam center, detector info, and partition settings.

    Parameters
    ----------
    qmap_file : str or Path
        Path to the qmap HDF5 file (.hdf, .h5).
    output_path : str or Path, optional
        Destination file path (.pdf, .png, etc.). If omitted, defaults to
        the same stem as ``qmap_file`` with a ``.pdf`` extension.
    raw_data : str or Path, optional
        Path to raw scattering dataset file. If omitted, attempts to load
        from the ``source_file`` path stored in the qmap file, or from
        scattering arrays embedded within the qmap file itself.
    crop_half_size : int, optional
        Half-size in pixels of the beam-center crop panel (default 100).
    beamline : str, optional
        Beamline reader used if loading external raw data (default "APS_8IDI").
    begin_idx : int, optional
        First frame index to include when loading raw scattering data (default 0).
    num_frames : int, optional
        Frames to average when loading raw scattering data:
        0 = all, >0 = exact count, -1 = representative subset (default -1).
    params : dict, optional
        Additional parameter overrides to display in the footer box.
    orientation : {"landscape", "portrait"}, optional
        Page orientation (default: "landscape").

    Returns
    -------
    str
        The absolute path to the generated report file.
    """
    qmap_str = str(qmap_file)
    data, meta, source_file, embedded_scat = _load_qmap_data(qmap_str)

    # Determine detector shape
    mask = data.get("mask")
    blemish = data.get("blemish")
    static_roi = data.get("static_roi_map")
    dynamic_roi = data.get("dynamic_roi_map")

    if mask is not None:
        shape = mask.shape
    elif static_roi is not None:
        shape = static_roi.shape
    elif embedded_scat is not None:
        shape = embedded_scat.shape
    else:
        shape = (512, 512)

    # Resolve scattering image
    scat = embedded_scat
    scat_source = None
    if raw_data is not None:
        raw_path = str(raw_data)
        if os.path.exists(raw_path):
            from pysimplemask.core.file_handler import get_handler

            h = get_handler(beamline, raw_path)
            if h is not None:
                scat = h.get_scattering(num_frames=num_frames, begin_idx=begin_idx)
                scat_source = raw_path
        else:
            logger.warning("Specified raw dataset does not exist: %s", raw_path)
    elif scat is None and source_file:
        candidate = source_file
        if not os.path.exists(candidate) and not os.path.isabs(candidate):
            rel = os.path.join(os.path.dirname(os.path.abspath(qmap_str)), candidate)
            if os.path.exists(rel):
                candidate = rel
        if os.path.exists(candidate):
            from pysimplemask.core.file_handler import get_handler

            h = get_handler(beamline, candidate)
            if h is not None:
                scat = h.get_scattering(num_frames=num_frames, begin_idx=begin_idx)
                scat_source = candidate
        else:
            logger.info("Referenced source_file not found: %s", source_file)

    # Beam center
    cx = meta.get("beam_center_x")
    cy = meta.get("beam_center_y")
    center_xy = (cx, cy)
    center_vh = (cy, cx) if (cx is not None and cy is not None) else (None, None)

    # Destination output path
    if output_path is None:
        output_path = os.path.splitext(qmap_str)[0] + ".pdf"

    # Assemble partition dictionary
    partition = {
        "static_roi_map": static_roi,
        "dynamic_roi_map": dynamic_roi,
        "static_num_pts": data.get("static_num_pts"),
        "dynamic_num_pts": data.get("dynamic_num_pts"),
        "map_names": data.get("map_names", ["q", "phi"]),
        "map_units": data.get("map_units", ["", ""]),
    }

    # Construct footer parameters
    p: Dict[str, Any] = {
        "source_file": os.path.basename(scat_source or source_file or qmap_str),
        "beamline": beamline,
        "begin_idx": begin_idx,
        "num_frames": num_frames,
    }
    if params:
        p.update(params)

    fig = _render_report_figure(
        fname=os.path.basename(qmap_str),
        shape=shape,
        center_xy=center_xy,
        center_vh=center_vh,
        meta=meta,
        mask=mask,
        blemish=blemish,
        scat=scat,
        partition=partition,
        crop_half_size=crop_half_size,
        params=p,
        orientation=orientation,
    )
    return _save_figure(fig, output_path)


report_from_qmap = generate_report_from_qmap
