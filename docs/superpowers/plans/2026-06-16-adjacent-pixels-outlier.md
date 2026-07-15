# Adjacent-Pixels Outlier Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second outlier-removal strategy ("AdjacentPixels") that divides the detector into fixed-size boxes, detects outlier pixels within each box using the existing percentile/MAD metrics, and plots the boxes sorted by mean intensity — matching the existing CircularRings interface.

**Architecture:** A new pure function `outlier_removal_adjacent_boxes` in `core/outlier_removal.py` (mirrors `outlier_removal_with_saxs`; returns the same `(saxs1d, bad_pixel_all)` shape). A new model method `compute_adjacent_saxs1d` wraps it. The controller's `mask_outlier` branch dispatches on `comboBox_outlier_target`, updates UI labels on combo-box change, and plots with "box index" as the x-axis. The `comboBox_outlier_target` widget already exists in `mask.ui` (with items "CircularRings" and "AdjacentPixels"); `label_outlier_target_info` and `outlier_num_roi` already exist.

**Tech Stack:** Python 3.12, numpy, pyqtgraph, PySide6. Env: `/local/MQICHU/envs/l2606_simplemask_refact/bin`. Repo root: `/home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact`.

---

## File structure

| File | Change |
|------|--------|
| `src/pysimplemask/core/outlier_removal.py` | Add `outlier_removal_adjacent_boxes(saxs_lin, mask, box_size, method, cutoff)` |
| `src/pysimplemask/core/model.py` | Add `compute_adjacent_saxs1d(self, method, cutoff, box_size)` |
| `src/pysimplemask/gui/control/main_window.py` | Wire `comboBox_outlier_target` → label/default; dispatch in `mask_evaluate` |
| `tests/core/test_outlier_removal.py` | New — tests for `outlier_removal_adjacent_boxes` |

---

## Background — existing data structures

`outlier_removal_with_saxs` (the CircularRings path) returns:
- `saxs1d`: `np.ndarray` shape `(5, k)` — rows are `[x, reference, threshold, max_val, raw_avg]`; columns are filtered for `reference > 0`.
- `bad_pixel_all`: `np.ndarray` shape `(2, M)` — 2D row/col indices of outlier pixels.

`outlier_removal_adjacent_boxes` returns the **same** shapes. `saxs1d[0]` is `box_index` (the sorted box rank, 0-based float) instead of a q-value.

The controller plots `saxs1d[0]` on the x-axis and `saxs1d[1]`, `saxs1d[2]`, `saxs1d[3]` as three lines. The x-label changes from `"q (Å⁻¹)"` to `"box index (sorted by mean)"`.

---

## Task 1: Core function `outlier_removal_adjacent_boxes`

**Files:**
- Modify: `src/pysimplemask/core/outlier_removal.py` (append after the last function)
- Create: `tests/core/test_outlier_removal.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/core/test_outlier_removal.py`:

```python
"""Tests for adjacent-box outlier removal."""

import numpy as np
import pytest

from pysimplemask.core.outlier_removal import outlier_removal_adjacent_boxes


def _uniform_image(shape=(64, 64), value=100.0):
    return np.full(shape, value, dtype=np.float32)


def test_returns_correct_shapes():
    img = _uniform_image()
    mask = np.ones(img.shape, dtype=bool)
    saxs1d, bad_pixels = outlier_removal_adjacent_boxes(img, mask, box_size=16)
    # 64/16 = 4 boxes per dim -> 16 boxes total; saxs1d has 5 rows
    assert saxs1d.ndim == 2
    assert saxs1d.shape[0] == 5
    assert bad_pixels.shape[0] == 2


def test_uniform_image_no_outliers():
    img = _uniform_image((64, 64), value=50.0)
    mask = np.ones(img.shape, dtype=bool)
    _, bad = outlier_removal_adjacent_boxes(img, mask, box_size=16)
    assert bad.shape[1] == 0


def test_single_hot_pixel_detected():
    img = _uniform_image((64, 64), value=10.0)
    mask = np.ones(img.shape, dtype=bool)
    img[5, 5] = 10000.0  # hot pixel in top-left box (box_size=16)
    _, bad = outlier_removal_adjacent_boxes(
        img, mask, box_size=16, method="percentile", cutoff=3.0
    )
    assert bad.shape[1] >= 1
    # The hot pixel coords must appear
    locs = set(zip(bad[0].tolist(), bad[1].tolist()))
    assert (5, 5) in locs


def test_masked_pixels_ignored():
    img = _uniform_image((64, 64), value=10.0)
    mask = np.ones(img.shape, dtype=bool)
    mask[0:16, 0:16] = False  # mask out first box entirely
    saxs1d, _ = outlier_removal_adjacent_boxes(img, mask, box_size=16)
    # 15 boxes remain (16 total minus the fully masked one)
    assert saxs1d.shape[1] == 15


def test_x_axis_is_sorted_box_index():
    rng = np.random.default_rng(42)
    img = rng.uniform(1, 100, size=(64, 64)).astype(np.float32)
    mask = np.ones(img.shape, dtype=bool)
    saxs1d, _ = outlier_removal_adjacent_boxes(img, mask, box_size=16)
    # x values must be strictly increasing (sorted rank)
    assert np.all(np.diff(saxs1d[0]) > 0)


def test_mad_method_works():
    img = _uniform_image((64, 64), value=20.0)
    mask = np.ones(img.shape, dtype=bool)
    img[10, 10] = 9999.0
    _, bad = outlier_removal_adjacent_boxes(
        img, mask, box_size=16, method="mad", cutoff=3.0
    )
    assert bad.shape[1] >= 1


def test_non_divisible_shape_uses_floor():
    # 70 // 16 = 4 boxes per dim -> 16 boxes
    img = _uniform_image((70, 70), value=5.0)
    mask = np.ones(img.shape, dtype=bool)
    saxs1d, _ = outlier_removal_adjacent_boxes(img, mask, box_size=16)
    assert saxs1d.shape[1] == 16  # floor(70/16)^2 = 4^2
```

- [ ] **Step 2: Run to verify they fail**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m pytest tests/core/test_outlier_removal.py -q
```
Expected: `ImportError` — `outlier_removal_adjacent_boxes` does not exist yet.

- [ ] **Step 3: Implement `outlier_removal_adjacent_boxes`**

Append to `src/pysimplemask/core/outlier_removal.py`:

```python
def outlier_removal_adjacent_boxes(
    saxs_lin,
    mask,
    box_size=32,
    method="percentile",
    cutoff=3.0,
    percentile=(5, 95),
    eps=1e-16,
):
    """Outlier removal by dividing the detector into adjacent square boxes.

    The image is split into non-overlapping ``box_size × box_size`` tiles
    (only complete tiles; edge remainder is ignored). Masked or non-positive
    pixels are excluded. Within each box the chosen metric (percentile or MAD)
    flags outlier pixels. Boxes are then sorted by their mean valid-pixel
    intensity so the result plots as a 1-D curve analogous to the q-ring curve.

    Parameters
    ----------
    saxs_lin : np.ndarray, shape (H, W)
        Raw scattering image (linear, not log).
    mask : np.ndarray, shape (H, W), dtype bool
        True = valid pixel.
    box_size : int
        Side length of each square box in pixels.
    method : {'percentile', 'mad'}
        Outlier detection strategy.
    cutoff : float
        Multiplier for the outlier threshold.
    percentile : tuple of two floats
        (low, high) percentile for clipping when method='percentile'.
    eps : float
        Guard against near-zero division.

    Returns
    -------
    saxs1d : np.ndarray, shape (5, k)
        Rows: [box_index_sorted, reference, threshold, max_val, raw_avg].
        Only boxes with at least one valid pixel and reference > 0 are kept.
        ``box_index_sorted`` is a monotonically increasing float rank (0, 1, 2, …).
    bad_pixel_all : np.ndarray, shape (2, M)
        Row/col indices of all detected outlier pixels.
    """
    H, W = saxs_lin.shape
    n_row = H // box_size
    n_col = W // box_size
    n_boxes = n_row * n_col

    if n_boxes == 0:
        return np.zeros((5, 0)), np.zeros((2, 0), dtype=int)

    records = []   # (raw_avg, ref, thr, max_val, bad_coords)

    for br in range(n_row):
        for bc in range(n_col):
            r0, r1 = br * box_size, (br + 1) * box_size
            c0, c1 = bc * box_size, (bc + 1) * box_size
            box_mask = mask[r0:r1, c0:c1]
            box_data = saxs_lin[r0:r1, c0:c1]

            valid = box_mask & (box_data > 0)
            if not np.any(valid):
                continue

            rows_v, cols_v = np.nonzero(valid)
            values = box_data[rows_v, cols_v]
            raw_avg = float(values.mean())

            if method.lower() == "percentile":
                ref, thr, om = compute_outlier_percentile(
                    values, cutoff=cutoff, percentiles=percentile, eps=eps
                )
            elif method.lower() == "mad":
                ref, thr, om = compute_outlier_mad(values, cutoff=cutoff, eps=eps)
            else:
                raise ValueError(
                    f"Unknown method '{method}'. Use 'percentile' or 'mad'."
                )

            if ref <= 0:
                continue

            # Convert local box coords back to global image coords
            global_rows = rows_v + r0
            global_cols = cols_v + c0
            if np.any(om):
                bad_coords = np.array(
                    [global_rows[om], global_cols[om]], dtype=int
                )
            else:
                bad_coords = np.zeros((2, 0), dtype=int)

            records.append((raw_avg, ref, thr, float(values.max()), bad_coords))

    if not records:
        return np.zeros((5, 0)), np.zeros((2, 0), dtype=int)

    # Sort boxes by raw mean intensity (ascending) so x-axis is meaningful
    records.sort(key=lambda r: r[0])

    k = len(records)
    saxs1d = np.zeros((5, k), dtype=np.float64)
    bad_list = []
    for i, (raw_avg, ref, thr, max_val, bad_coords) in enumerate(records):
        saxs1d[0, i] = float(i)      # sorted box rank (x-axis)
        saxs1d[1, i] = ref           # reference
        saxs1d[2, i] = thr           # threshold
        saxs1d[3, i] = max_val       # max value
        saxs1d[4, i] = raw_avg       # raw average
        if bad_coords.shape[1] > 0:
            bad_list.append(bad_coords)

    bad_pixel_all = np.hstack(bad_list) if bad_list else np.zeros((2, 0), dtype=int)
    return saxs1d, bad_pixel_all
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m pytest tests/core/test_outlier_removal.py -q
```
Expected: `7 passed`.

- [ ] **Step 5: Lint**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m ruff check src/pysimplemask/core/outlier_removal.py tests/core/test_outlier_removal.py
```
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add src/pysimplemask/core/outlier_removal.py tests/core/test_outlier_removal.py
git commit -m "feat(core): add outlier_removal_adjacent_boxes

Divides the detector into box_size×box_size tiles, detects outlier pixels
within each box using the existing percentile/MAD helpers, and returns the
same (saxs1d, bad_pixel_all) interface as outlier_removal_with_saxs.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Model method `compute_adjacent_saxs1d`

**Files:**
- Modify: `src/pysimplemask/core/model.py`

The existing `compute_saxs1d` (circular rings) is at line ~193. Add the new method directly below it.

- [ ] **Step 1: Add the import at the top of model.py**

In `src/pysimplemask/core/model.py`, the `outlier_removal_with_saxs` import line is:

```python
from .outlier_removal import outlier_removal_with_saxs
```

Change it to:

```python
from .outlier_removal import outlier_removal_adjacent_boxes, outlier_removal_with_saxs
```

- [ ] **Step 2: Add the method**

After the closing `return saxs1d, zero_loc` of `compute_saxs1d`, add:

```python
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
            "adjacent-box outlier removal finished in %f seconds", time.perf_counter() - t0
        )
        return saxs1d, zero_loc
```

- [ ] **Step 3: Verify headless import**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -c "
from pysimplemask.core.model import SimpleMaskModel
m = SimpleMaskModel()
print('compute_adjacent_saxs1d' in dir(m))
import sys; print('Qt-free:', 'PySide6' not in sys.modules)
"
```
Expected:
```
True
Qt-free: True
```

- [ ] **Step 4: Lint**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m ruff check src/pysimplemask/core/model.py
```
Expected: `All checks passed!`

- [ ] **Step 5: Commit**

```bash
git add src/pysimplemask/core/model.py
git commit -m "feat(core): add SimpleMaskModel.compute_adjacent_saxs1d

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Controller — wire the target combo-box and dispatch

**Files:**
- Modify: `src/pysimplemask/gui/control/main_window.py`

There are three changes: (a) connect the target combo-box to a slot that updates the label and default num_roi; (b) add that slot; (c) dispatch in `mask_evaluate`.

- [ ] **Step 1: Connect the signal in `__init__`**

In `main_window.py`, after line 168 (`self.mask_outlier_hdl.setBackground((255, 255, 255))`), add:

```python
        self.comboBox_outlier_target.currentIndexChanged.connect(
            self._on_outlier_target_changed
        )
```

- [ ] **Step 2: Add the slot method**

Add this method to `SimpleMaskGUI`, just before `mask_evaluate`:

```python
    def _on_outlier_target_changed(self):
        target = self.comboBox_outlier_target.currentText()
        if target == "CircularRings":
            self.label_outlier_target_info.setText("num. circular ROI:")
            self.outlier_num_roi.setValue(180)
        elif target == "AdjacentPixels":
            self.label_outlier_target_info.setText("adjacent box size (pixel):")
            self.outlier_num_roi.setValue(32)
```

- [ ] **Step 3: Dispatch inside `mask_evaluate`'s `mask_outlier` branch**

The current `mask_outlier` branch begins at line 311. Replace the entire branch:

```python
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
```

- [ ] **Step 4: Offscreen GUI smoke test**

```bash
QT_QPA_PLATFORM=offscreen /local/MQICHU/envs/l2606_simplemask_refact/bin/python -c "
from PySide6.QtWidgets import QApplication
from pysimplemask.gui.control.main_window import SimpleMaskGUI
app = QApplication([])
w = SimpleMaskGUI()
# switch to AdjacentPixels and verify label + num_roi update
idx = w.comboBox_outlier_target.findText('AdjacentPixels')
w.comboBox_outlier_target.setCurrentIndex(idx)
assert w.label_outlier_target_info.text() == 'adjacent box size (pixel):'
assert w.outlier_num_roi.value() == 32
# switch back to CircularRings
idx2 = w.comboBox_outlier_target.findText('CircularRings')
w.comboBox_outlier_target.setCurrentIndex(idx2)
assert w.label_outlier_target_info.text() == 'num. circular ROI:'
assert w.outlier_num_roi.value() == 180
print('combo-box wiring ok')
" 2>&1 | tail -3
```
Expected last line: `combo-box wiring ok`.

- [ ] **Step 5: Lint**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m ruff check src/pysimplemask/gui/control/main_window.py
```
Expected: `All checks passed!`

- [ ] **Step 6: Full test suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m pytest tests -q
```
Expected: all green (previous count + 7 new outlier tests).

- [ ] **Step 7: Commit**

```bash
git add src/pysimplemask/gui/control/main_window.py
git commit -m "feat(gui): wire AdjacentPixels outlier target

- comboBox_outlier_target change updates label + default box size
- mask_evaluate dispatches CircularRings -> compute_saxs1d,
  AdjacentPixels -> compute_adjacent_saxs1d
- x-axis label changes to 'box index (sorted by mean)' for AdjacentPixels

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-review — spec coverage

| Requirement | Task |
|---|---|
| `comboBox_outlier_target` "AdjacentPixels" sets `label_outlier_target_info` to "adjacent box size (pixel)" | T3 Step 2 |
| Default `outlier_num_roi` to 32 on target change | T3 Step 2 |
| Divide image into `box_size × box_size` sub-groups | T1 Step 3 |
| Ignore masked / invalid (non-positive) pixels | T1 Step 3 (`valid = box_mask & (box_data > 0)`) |
| Tag outlier pixels using the specified metric | T1 Step 3 (dispatches to existing `compute_outlier_percentile` / `compute_outlier_mad`) |
| Sort groups by mean and plot as 1-D curve, same format as CircularRings | T1 Step 3 (sort by `raw_avg`; `saxs1d[0]` = sorted rank) |
| Controller plots same 3 lines (ref, cutoff, max) | T3 Step 3 |
| CircularRings still works unchanged | T3 Step 3 (early `if` branch) |
