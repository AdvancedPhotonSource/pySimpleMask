"""One-page letter-size PDF summary of the hdf→qmap pipeline."""

import logging
import os

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
    if mask is not None:
        img[mask == 0] = 0
    out = np.full_like(img, np.nan)
    positive = img > 0
    out[positive] = np.log10(img[positive])
    return out


def generate_report(model, output_path, crop_half_size=100):
    """Write a one-page letter-size PDF summary of the qmap generation.

    Layout (portrait, 8.5 × 11 in):
    - Header: filename, shape, beam center, energy, distance, mask %.
    - Row 1: blemish-only log scattering | scattering × mask (log) | mask.
    - Row 2: beam-center crop with crosshair | static qmap | dynamic qmap.

    Qmap panels use a tab20 qualitative colormap cycling across bins; bin 0
    (masked pixels) is always white.

    Parameters
    ----------
    model : SimpleMaskModel
        A loaded (and optionally partitioned) model.
    output_path : str
        Destination PDF path (created or overwritten).
    crop_half_size : int
        Half-size in pixels of the beam-center crop panel (default 100).
    """
    fig = plt.figure(figsize=(8.5, 11))

    # ── Title / metadata header ───────────────────────────────────────────────
    fname = os.path.basename(model.dset.fname)
    meta = model.dset.metadata
    center_xy = model.get_center(mode="xy")
    title = (
        f"{fname}   |   shape {model.shape[1]}×{model.shape[0]}"
        f"   |   center ({center_xy[0]:.1f}, {center_xy[1]:.1f}) px"
    )
    n_masked = int((~model.mask.astype(bool)).sum())
    sub = (
        f"E = {meta.get('energy', 'n/a'):.4g} keV   "
        f"dist = {meta.get('detector_distance', 'n/a'):.4g} m   "
        f"pix = {meta.get('pixel_size', 'n/a') * 1e6:.1f} µm   "
        f"masked = {n_masked / model.mask.size * 100:.1f}%"
    )
    fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=8, fontweight="bold")
    fig.text(0.5, 0.945, sub, ha="center", va="top", fontsize=7, color="#444444")

    # ── Grid layout ───────────────────────────────────────────────────────────
    gs = GridSpec(
        2, 3, figure=fig,
        top=0.93, bottom=0.04, left=0.05, right=0.97,
        hspace=0.35, wspace=0.30,
        height_ratios=[1.4, 1.0],
    )

    def _colorbar(im, ax):
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

    # ── Row 1, panel 1: scattering + blemish (log) ───────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    img1 = _log_image(model.dset.scat, mask=model.mask_kernel.blemish)
    im1 = ax1.imshow(img1, cmap="jet", origin="upper", aspect="equal")
    ax1.set_title("Scattering + blemish (log)", fontsize=7)
    ax1.axis("off")
    _colorbar(im1, ax1)

    # ── Row 1, panel 2: scattering × user mask (log) ─────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    img2 = _log_image(model.dset.scat, mask=model.mask)
    im2 = ax2.imshow(img2, cmap="jet", origin="upper", aspect="equal")
    ax2.set_title("Scattering × mask (log)", fontsize=7)
    ax2.axis("off")
    _colorbar(im2, ax2)

    # ── Row 1, panel 3: combined mask ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(
        model.mask.astype(np.uint8), cmap="gray", vmin=0, vmax=1,
        origin="upper", aspect="equal",
    )
    ax3.set_title("Mask (white = valid)", fontsize=7)
    ax3.axis("off")

    # ── Row 2, panel 1: beam-center crop ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    cy, cx = model.get_center(mode="vh")
    H, W = model.shape
    if 0 <= cy < H and 0 <= cx < W:
        r0 = max(0, int(cy) - crop_half_size)
        r1 = min(H, int(cy) + crop_half_size)
        c0 = max(0, int(cx) - crop_half_size)
        c1 = min(W, int(cx) + crop_half_size)
        crop = _log_image(model.dset.scat, mask=model.mask)[r0:r1, c0:c1]
        im4 = ax4.imshow(crop, cmap="jet", origin="upper", aspect="equal")
        _colorbar(im4, ax4)
        local_cy = float(cy) - r0
        local_cx = float(cx) - c0
        ax4.axhline(local_cy, color="cyan", linewidth=0.6, alpha=0.8)
        ax4.axvline(local_cx, color="cyan", linewidth=0.6, alpha=0.8)
        ax4.set_title(f"Beam center crop  ({cx:.0f}, {cy:.0f})", fontsize=7)
    else:
        ax4.text(
            0.5, 0.5, "Center outside frame",
            ha="center", va="center", transform=ax4.transAxes, fontsize=8,
        )
        ax4.set_title("Beam center crop", fontsize=7)
    ax4.axis("off")

    # ── Row 2, panels 2 & 3: qmaps ───────────────────────────────────────────
    partition = model.new_partition
    map_names = partition["map_names"] if partition else ["q", "phi"]
    map_units = partition["map_units"] if partition else ["", ""]  # noqa: F841

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

    if partition:
        sq_n = partition["static_num_pts"]
        dq_n = partition["dynamic_num_pts"]
        sq_title = (
            f"Static qmap  ({map_names[0]}-{map_names[1]})\n"
            f"{sq_n[0]} {map_names[0]} × {sq_n[1]} {map_names[1]} bins"
        )
        dq_title = (
            f"Dynamic qmap  ({map_names[0]}-{map_names[1]})\n"
            f"{dq_n[0]} {map_names[0]} × {dq_n[1]} {map_names[1]} bins"
        )
        sq_data = partition["static_roi_map"]
        dq_data = partition["dynamic_roi_map"]
    else:
        sq_title = f"Static qmap  ({map_names[0]}-{map_names[1]})"
        dq_title = f"Dynamic qmap  ({map_names[0]}-{map_names[1]})"
        sq_data = dq_data = None

    _qmap_panel(fig.add_subplot(gs[1, 1]), sq_data, sq_title)
    _qmap_panel(fig.add_subplot(gs[1, 2]), dq_data, dq_title)

    # ── Save ─────────────────────────────────────────────────────────────────
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=150)
    plt.close(fig)
    logger.info("Report saved: %s", output_path)
