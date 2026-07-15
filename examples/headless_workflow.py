"""
pySimpleMask headless example
==============================
Replicates the typical GUI workflow entirely from Python — no display needed.

Steps:
  1. Load a raw scattering file
  2. Set beam center to the intensity maximum, then refine with find_center
  3. Apply a blemish (bad-pixel) file
  4. Compute a q-phi partition (dynamic + static)
  5. Save the mask (TIFF) and the full partition (HDF5)

Run:
  python examples/headless_workflow.py
"""

import logging

from pysimplemask.core import SimpleMaskModel

# Optional: see timing logs from find_center, partition, etc.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s][%(module)s]: %(message)s",
)

# ── 1. Load data ──────────────────────────────────────────────────────────────
m = SimpleMaskModel()
m.read_data(
    "scan.hdf",
    beamline="APS_8IDI",  # or "APS_9IDD"
    begin_idx=0,           # first frame to include
    num_frames=-1,         # -1 = representative subset; 0 = all frames
)
print(f"Loaded: detector shape {m.shape}")

# ── 2. Set center to the intensity maximum, then refine ───────────────────────
# goto_max: finds the smoothed brightness peak and writes it into the metadata.
center_vh = m.goto_max()
print(f"Initial center (goto_max): row={center_vh[0]:.1f}, col={center_vh[1]:.1f}")

# find_center: iterative centro-symmetry cross-correlation (1-2 passes typical).
# max_radius caps the crop window for speed on large detectors (default 384 px).
refined_vh = m.find_center(max_radius=384)
print(f"Refined center:            row={refined_vh[0]:.1f}, col={refined_vh[1]:.1f}")

# Push the refined center back into the metadata and recompute the q-map
# channels (qmap["q"], qmap["phi"], …). Must be called before compute_partition.
m.dset.set_center_vh(refined_vh)
m.update_parameters()
cx, cy = m.get_center(mode="xy")
print(f"Active beam center:        x={cx:.2f}, y={cy:.2f}  (pixel, col/row)")

# ── 3. Apply a blemish (bad-pixel) file ───────────────────────────────────────
# Supported formats: single-channel TIFF or HDF5 (pass key= for the dataset path).
m.mask_evaluate("mask_blemish", fname="blemish.tif")
# For HDF5: m.mask_evaluate("mask_blemish", fname="blemish.h5", key="/data/mask")
m.mask_apply("mask_blemish")
bad = int(m.mask.size - m.mask.sum())
print(f"After blemish: {bad} pixels masked ({bad / m.mask.size * 100:.2f}%)")

# ── Optional extra mask layers ─────────────────────────────────────────────────
# Threshold: exclude pixels outside [low, high)
# m.mask_evaluate("mask_threshold", low=0, high=65535,
#                 low_enable=False, high_enable=True)
# m.mask_apply("mask_threshold")

# Polygon ROI (exclusive = masked out; inclusive = keep only this region)
# m.add_polygon([(100, 50), (100, 200), (300, 200), (300, 50)], mode="exclusive")
# m.evaluate_draw()
# m.mask_apply("mask_draw")

# ── 4. Compute q-phi partition ────────────────────────────────────────────────
# dq_num / dp_num : dynamic (coarse) resolution — used by XPCS correlation.
# sq_num / sp_num : static  (fine)   resolution — must be multiples of dq/dp.
# least_multiple() is applied automatically so sq is always a multiple of dq.
m.compute_partition(
    mode="q-phi",
    dq_num=10,    # dynamic q bins
    sq_num=100,   # static  q bins  (must be a multiple of dq_num)
    dp_num=36,    # dynamic phi bins
    sp_num=360,   # static  phi bins (must be a multiple of dp_num)
    phi_offset=0.0,
    symmetry_fold=1,
)
print("Partition computed.")

# ── 5. Save results ───────────────────────────────────────────────────────────
m.save_mask("mask.tif")       # TIFF, LZW-compressed, uint8
m.save_partition("qmap.hdf")  # NeXus-XPCS HDF5 under /qmap
print("Saved: mask.tif  qmap.hdf")
