# qmap Summary Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a one-page letter-size PDF summary of the hdf→qmap pipeline, showing scattering, mask, beam-center crop, and partition maps as images with a metadata header.

**Architecture:** A new Qt-free `core/report.py` module with a single public function `generate_report(model, output_path, crop_half_size=100)` that uses matplotlib (already a dependency) to compose a fixed 2-row × 3-column figure on a letter-size page and writes it as a PDF. The CLI gets an optional `--report FILE` flag; the Python API is callable directly on any loaded `SimpleMaskModel`.

**Tech Stack:** Python 3.12, matplotlib (figure, GridSpec, PdfPages, ListedColormap), numpy. No Qt. Env: `/local/MQICHU/envs/l2606_simplemask_refact/bin`. Repo root: `/home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact`.

---

## Layout (letter = 8.5 × 11 in, portrait)

```
┌─────────────────────────────────────────────────────────────┐
│  Title: filename │ beamline │ shape │ center │ energy │ dist │
├──────────┬──────────┬──────────────────────────────────────┤
│ Scat +   │ Scat ×   │ Mask                                 │
│ blemish  │ mask     │ (binary, combined)                   │
│ (log)    │ (log)    │                                      │
├──────────┼──────────┴──────────────────────────────────────┤
│ Beam     │ Static qmap          │ Dynamic qmap             │
│ crop *   │ (tab20, 0=white)     │ (tab20, 0=white)         │
└──────────┴──────────────────────────────────────────────────┘
* if center is within the image; otherwise panel is labelled "center out of frame"
```

Row 1 (taller): 3 columns — blemish-only log scattering | scattering×mask (log) | mask.
Row 2 (shorter): 3 columns — beam-center crop | static qmap | dynamic qmap.

---

## Data sources on `SimpleMaskModel`

| What | Where |
|------|-------|
| Raw scattering (linear) | `model.dset.scat` |
| Scattering log | `model.dset.data_display[0]` ("scattering") |
| Scattering × mask (log) | `model.dset.data_display[1]` ("scattering * mask") |
| Mask (bool, 0=bad) | `model.dset.data_display[2]` ("mask") |
| dqmap partition | `model.dset.data_display[3]` ("dqmap_partition") |
| sqmap partition | `model.dset.data_display[4]` ("sqmap_partition") |
| Default blemish | `model.mask_kernel.blemish` |
| Beam center (row, col) | `model.get_center(mode="vh")` |
| Image shape | `model.shape` |
| Metadata dict | `model.dset.metadata` (energy, detector_distance, pixel_size) |
| Source file | `model.dset.fname` |
| Partition maps | `model.new_partition["dynamic_roi_map"]`, `["static_roi_map"]` |
| Partition axis names/units | `model.new_partition["map_names"]`, `["map_units"]` |

**Blemish-only scattering** (panel 1): apply `model.mask_kernel.blemish` to `model.dset.scat`; take log10 of positive pixels.

**Beam crop** (panel 4): center = `(row, col)` from `get_center("vh")`; extract `[row-crop_half_size : row+crop_half_size, col-crop_half_size : col+crop_half_size]` from the log scattering; draw a `+` crosshair at the center; skip (label "center outside frame") if center is outside `model.shape`.

**qmap colormap**: `ListedColormap` with index 0 = white (masked pixels, value 0 in partition), indices 1-20 = tab20 colors cycling for larger maps.

---

## File structure

| File | Change |
|------|--------|
| `src/pysimplemask/core/report.py` | New — `generate_report`, `_qmap_cmap`, `_log_image` |
| `src/pysimplemask/cli.py` | Add `--report FILE` flag to `_build_qmap_args`; call in `_run_build_qmap` |
| `tests/core/test_report.py` | New — synthetic model + PDF existence/content checks |

---

## Task 1: Core report module

**Files:**
- Create: `src/pysimplemask/core/report.py`
- Create: `tests/core/test_report.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_report.py`:

```python
"""Tests for the qmap summary report generator."""

import os

import h5py
import numpy as np
import pytest

from pysimplemask.core import SimpleMaskModel
from pysimplemask.core.report import generate_report


@pytest.fixture
def loaded_model(tmp_path):
    """A SimpleMaskModel with data loaded and a partition computed."""
    p = tmp_path / "scan.h5"
    rng = np.random.default_rng(42)
    frames = rng.integers(1, 100, size=(3, 64, 60)).astype(np.uint16)
    with h5py.File(p, "w") as h:
        h["/entry/data/data"] = frames
    m = SimpleMaskModel()
    m.read_data(str(p), beamline="APS_8IDI", num_frames=0)
    m.compute_partition(
        mode="q-phi",
        dq_num=2, sq_num=4, dp_num=1, sp_num=1,
    )
    return m


def test_generate_report_creates_pdf(loaded_model, tmp_path):
    out = tmp_path / "report.pdf"
    generate_report(loaded_model, str(out))
    assert out.exists()
    assert out.stat().st_size > 1024  # a real PDF is bigger than 1 KB


def test_generate_report_overwrite(loaded_model, tmp_path):
    out = tmp_path / "report.pdf"
    generate_report(loaded_model, str(out))
    first_size = out.stat().st_size
    generate_report(loaded_model, str(out))
    assert out.exists()
    assert out.stat().st_size > 0


def test_report_with_no_partition(tmp_path):
    """Report should not crash when no partition has been computed yet."""
    p = tmp_path / "scan.h5"
    rng = np.random.default_rng(0)
    with h5py.File(p, "w") as h:
        h["/entry/data/data"] = rng.integers(1, 50, size=(3, 32, 30)).astype(np.uint16)
    m = SimpleMaskModel()
    m.read_data(str(p), beamline="APS_8IDI", num_frames=0)
    out = tmp_path / "report.pdf"
    generate_report(m, str(out))
    assert out.exists()
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m pytest tests/core/test_report.py -q 2>&1 | tail -5
```
Expected: `ImportError` — `pysimplemask.core.report` doesn't exist yet.

- [ ] **Step 3: Implement `src/pysimplemask/core/report.py`**

```python
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


def _qmap_cmap():
    """ListedColormap: index 0 = white (masked), 1-20 = tab20 cycling."""
    tab20 = plt.cm.get_cmap("tab20", _TAB20_N)
    # Build enough colors to cover up to 1000 partition bins (cycling tab20).
    colors = [(1.0, 1.0, 1.0, 1.0)]  # index 0 → white (masked)
    for i in range(1000):
        colors.append(tab20(i % _TAB20_N))
    return ListedColormap(colors)


def _log_image(scat, mask=None):
    """Return a log10 image of ``scat`` with optional ``mask`` applied.

    Masked-out pixels and non-positive values are set to NaN so matplotlib
    renders them as white/transparent.
    """
    img = scat.astype(np.float64).copy()
    if mask is not None:
        img[mask == 0] = 0
    positive = img > 0
    out = np.full_like(img, np.nan)
    out[positive] = np.log10(img[positive])
    return out


def generate_report(model, output_path, crop_half_size=100):
    """Write a one-page letter-size PDF summary of the qmap generation.

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

    # ── Title / metadata ─────────────────────────────────────────────────────
    fname = os.path.basename(model.dset.fname)
    meta = model.dset.metadata
    center_xy = model.get_center(mode="xy")
    title = (
        f"{fname}   |   shape {model.shape[1]}×{model.shape[0]}"
        f"   |   center ({center_xy[0]:.1f}, {center_xy[1]:.1f}) px"
    )
    sub = (
        f"E = {meta.get('energy', 'n/a'):.4g} keV   "
        f"dist = {meta.get('detector_distance', 'n/a'):.4g} m   "
        f"pix = {meta.get('pixel_size', 'n/a') * 1e6:.1f} µm   "
        f"masked = {(~model.mask.astype(bool)).sum() / model.mask.size * 100:.1f}%"
    )
    fig.text(0.5, 0.97, title, ha="center", va="top", fontsize=8, fontweight="bold")
    fig.text(0.5, 0.945, sub, ha="center", va="top", fontsize=7, color="#444444")

    # ── Grid ─────────────────────────────────────────────────────────────────
    gs = GridSpec(
        2, 3, figure=fig,
        top=0.93, bottom=0.04, left=0.05, right=0.97,
        hspace=0.35, wspace=0.25,
        height_ratios=[1.4, 1.0],
    )

    cmap_scat = "jet"
    cmap_mask = "gray"
    cmap_qmap = _qmap_cmap()

    # ── Row 1 panel 1: scattering with blemish only ──────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    scat_blemish = _log_image(model.dset.scat, mask=model.mask_kernel.blemish)
    im1 = ax1.imshow(scat_blemish, cmap=cmap_scat, origin="upper", aspect="equal")
    ax1.set_title("Scattering + blemish (log)", fontsize=7)
    ax1.axis("off")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

    # ── Row 1 panel 2: scattering × user mask ────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    scat_mask = _log_image(model.dset.scat, mask=model.mask)
    im2 = ax2.imshow(scat_mask, cmap=cmap_scat, origin="upper", aspect="equal")
    ax2.set_title("Scattering × mask (log)", fontsize=7)
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)

    # ── Row 1 panel 3: mask ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(model.mask.astype(np.uint8), cmap=cmap_mask, vmin=0, vmax=1,
               origin="upper", aspect="equal")
    ax3.set_title("Mask (white = valid)", fontsize=7)
    ax3.axis("off")

    # ── Row 2 panel 1: beam-center crop ──────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    cy, cx = model.get_center(mode="vh")
    H, W = model.shape
    if 0 <= cy < H and 0 <= cx < W:
        r0 = max(0, int(cy) - crop_half_size)
        r1 = min(H, int(cy) + crop_half_size)
        c0 = max(0, int(cx) - crop_half_size)
        c1 = min(W, int(cx) + crop_half_size)
        crop = _log_image(model.dset.scat, mask=model.mask)[r0:r1, c0:c1]
        im4 = ax4.imshow(crop, cmap=cmap_scat, origin="upper", aspect="equal")
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04).ax.tick_params(labelsize=6)
        # crosshair at the center within the crop
        local_cy = cy - r0
        local_cx = cx - c0
        ax4.axhline(local_cy, color="cyan", linewidth=0.6, alpha=0.8)
        ax4.axvline(local_cx, color="cyan", linewidth=0.6, alpha=0.8)
        ax4.set_title(
            f"Beam center crop  ({cx:.0f}, {cy:.0f})", fontsize=7
        )
    else:
        ax4.text(0.5, 0.5, "Center outside frame",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=8)
        ax4.set_title("Beam center crop", fontsize=7)
    ax4.axis("off")

    # ── Row 2 panels 2 & 3: qmaps ────────────────────────────────────────────
    partition = model.new_partition

    def _qmap_panel(ax, data, title):
        if data is None:
            ax.text(0.5, 0.5, "No partition computed",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8)
            ax.set_title(title, fontsize=7)
            ax.axis("off")
            return
        n_bins = int(data.max())
        cmap = ListedColormap(
            [(1.0, 1.0, 1.0, 1.0)]
            + [plt.cm.get_cmap("tab20", _TAB20_N)(i % _TAB20_N) for i in range(max(1, n_bins))]
        )
        ax.imshow(data, cmap=cmap, vmin=0, vmax=max(1, n_bins),
                  origin="upper", aspect="equal", interpolation="nearest")
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    sq_data = partition["static_roi_map"] if partition else None
    dq_data = partition["dynamic_roi_map"] if partition else None
    map_names = partition["map_names"] if partition else ["q", "phi"]
    map_units = partition["map_units"] if partition else ["", ""]
    sq_title = f"Static qmap  ({map_names[0]}-{map_names[1]})"
    dq_title = f"Dynamic qmap  ({map_names[0]}-{map_names[1]})"
    if partition:
        sq_n = partition[f"static_num_pts"]
        dq_n = partition[f"dynamic_num_pts"]
        sq_title += f"\n{sq_n[0]} {map_names[0]} × {sq_n[1]} {map_names[1]} bins"
        dq_title += f"\n{dq_n[0]} {map_names[0]} × {dq_n[1]} {map_names[1]} bins"

    _qmap_panel(fig.add_subplot(gs[1, 1]), sq_data, sq_title)
    _qmap_panel(fig.add_subplot(gs[1, 2]), dq_data, dq_title)

    # ── Save ─────────────────────────────────────────────────────────────────
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig, dpi=150)
    plt.close(fig)
    logger.info("Report saved: %s", output_path)
```

- [ ] **Step 4: Run tests — all must pass**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m pytest tests/core/test_report.py -q 2>&1 | tail -5
```
Expected: `3 passed`.

- [ ] **Step 5: Lint**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m ruff check src/pysimplemask/core/report.py tests/core/test_report.py
```
Expected: `All checks passed!`

- [ ] **Step 6: Full suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m pytest tests -q 2>&1 | tail -2
```
Expected: all pass (previous 103 + 3 new).

- [ ] **Step 7: Commit**

```bash
git add src/pysimplemask/core/report.py tests/core/test_report.py
git commit -m "feat(core): generate_report — one-page PDF summary of hdf→qmap pipeline

6-panel letter-size PDF: blemish-only scattering, scattering×mask,
combined mask, beam-center crop with crosshair, static qmap, dynamic qmap.
Qualitative tab20 colormap (index 0 = white for masked pixels).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Wire into CLI

**Files:**
- Modify: `src/pysimplemask/cli.py`

- [ ] **Step 1: Add the `--report` flag**

In `_build_qmap_args`, find the `# ── output` group and add one argument after `--output-mask`:

```python
    grp_out.add_argument(
        "--report",
        default=None,
        metavar="FILE",
        help=(
            "Write a one-page PDF summary report to FILE. "
            "Pass empty string to skip. Default: same stem as --output-qmap with .pdf extension."
        ),
    )
```

- [ ] **Step 2: Call `generate_report` in `_run_build_qmap`**

In `_run_build_qmap`, after the `save_partition` / `save_mask` block (step 5), add:

```python
    # ── 6. Report ────────────────────────────────────────────────────────────
    report_path = args.report
    if report_path is None:
        # default: same stem as the qmap output, .pdf extension
        report_path = os.path.splitext(args.output_qmap)[0] + ".pdf"
    if report_path:
        from pysimplemask.core.report import generate_report
        generate_report(m, report_path)
        logging.info("Report saved: %s", report_path)
```

- [ ] **Step 3: Update `test_default_args_parsed` to assert the new default**

In `tests/cli/test_build_qmap.py`, add to `test_default_args_parsed`:
```python
    assert args.report is None
```

- [ ] **Step 4: Run CLI tests**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m pytest tests/cli/ -q 2>&1 | tail -3
```
Expected: all 6 pass.

- [ ] **Step 5: Verify report is auto-generated in the pipeline test**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -c "
import tempfile, os, h5py, numpy as np
from pysimplemask.cli import _build_qmap_args, _run_build_qmap
d = tempfile.mkdtemp(); os.chdir(d)
p = os.path.join(d, 's.h5')
with h5py.File(p, 'w') as h: h['/entry/data/data'] = np.random.default_rng(0).integers(1, 50, (3,32,30)).astype('uint16')
args = _build_qmap_args([p, '--num-frames','0','--no-find-center',
    '--dq-num','2','--sq-num','4','--dp-num','1','--sp-num','1',
    '--output-qmap','qmap.hdf','--output-mask','mask.tif'])
_run_build_qmap(args)
print('qmap.pdf exists:', os.path.exists('qmap.pdf'))
print('size:', os.path.getsize('qmap.pdf'), 'bytes')
"
```
Expected: `qmap.pdf exists: True` with size > 1024.

- [ ] **Step 6: Lint**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m ruff check src/pysimplemask/cli.py tests/cli/
```

- [ ] **Step 7: Full suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m pytest tests -q 2>&1 | tail -2
```

- [ ] **Step 8: Commit**

```bash
git add src/pysimplemask/cli.py tests/cli/test_build_qmap.py
git commit -m "feat(cli): auto-generate PDF report after build-qmap

--report FILE (default: same stem as --output-qmap + .pdf) passes through
to generate_report. Set --report '' to disable.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review — spec coverage

| Requirement | Task |
|---|---|
| One-page letter size | T1: `figsize=(8.5, 11)` |
| Scattering with blemish (log) | T1: `_log_image(scat, blemish)`, panel gs[0,0] |
| User mask + scattering with mask | T1: `_log_image(scat, mask)` gs[0,1]; mask gs[0,2] |
| Beam-center crop with position; skip if outside | T1: gs[1,0] with `0 <= cy < H and 0 <= cx < W` guard |
| Crosshair at beam center | T1: `ax4.axhline / axvline` at local crop coords |
| Static qmap with tab20 | T1: `_qmap_panel(…, sq_data, …)` gs[1,1] |
| Dynamic qmap with tab20 | T1: `_qmap_panel(…, dq_data, …)` gs[1,2] |
| Masked pixels = white (index 0) | T1: `ListedColormap` white at index 0 |
| No crash when partition not computed | T1: `test_report_with_no_partition`; None guard in `_qmap_panel` |
| CLI auto-generates report | T2: `--report` flag with default stem+.pdf |
| Skip report when `--report ''` | T2: `if report_path:` guard |
