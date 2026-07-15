# Log Scale Checkbox Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing `plot_log` checkbox so toggling it switches the scattering display between log₁₀ and linear intensity.

**Architecture:** Two targeted edits to `main_window.py` — add `DISPLAY_FIELD` import, connect `plot_log.stateChanged` to `plot`, and prepend a channel-refresh block to `plot()` that calls `dset.get_scat_with_mask(mode=...)` for channels 0 and 1 based on the checkbox state. One headless GUI test verifies the two channels differ between the two modes.

**Tech Stack:** PySide6, pyqtgraph, existing `FileReader.get_scat_with_mask(mask, mode)`

## Global Constraints

- Environment: `/local/MQICHU/envs/l2606_simplemask_refact/bin/`
- GUI tests run headless: `QT_QPA_PLATFORM=offscreen`
- `plot_log` checked (True) → `mode="log"` → `np.log10` applied
- `plot_log` unchecked (False) → `mode="linear"` → raw counts
- Only channels 0 (`"scattering"`) and 1 (`"scattering * mask"`) are affected; mask, partition, and preview channels are unchanged
- `get_scat_with_mask(mask=None, mode=...)` uses an all-True mask when `mask=None`; pass `self.sm.mask` for channel 1
- `plot_log` is checked by default (matches existing log-scaled default behavior)
- Ruff must pass clean

---

## File Map

| Status | File | Change |
|--------|------|--------|
| Modify | `src/pysimplemask/gui/control/main_window.py` | Import `DISPLAY_FIELD`; connect signal; update `plot()` |
| Modify | `tests/gui/test_main_window.py` | Add log/linear switch test |

---

## Task 1: Wire log-scale checkbox

**Files:**
- Modify: `src/pysimplemask/gui/control/main_window.py`
- Modify: `tests/gui/test_main_window.py`

**Interfaces:**
- `FileReader.get_scat_with_mask(mask, mode)` — already exists in `base_reader.py`:
  - `mask`: boolean ndarray or `None` (None → all-True mask)
  - `mode`: `"log"` or `"linear"`
  - Returns float32 ndarray of shape `(H, W)`
- `DISPLAY_FIELD` — list in `base_reader.py`:
  `["scattering", "scattering * mask", "mask", "dqmap_partition", "sqmap_partition", "preview"]`
  - `DISPLAY_FIELD.index("scattering")` == 0
  - `DISPLAY_FIELD.index("scattering * mask")` == 1

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/test_main_window.py`:

```python
def test_log_scale_checkbox_switches_display_mode(qapp, tmp_path):
    """Toggling plot_log changes data_display channels 0 and 1."""
    rng = np.random.default_rng(42)
    frames = rng.integers(1, 200, size=(3, 16, 14)).astype(np.uint16)
    gui = _load_gui(tmp_path, frames)

    # Default: log checked — channel 0 should be log-scaled (all values finite, no very large raw counts)
    gui.plot_log.setChecked(True)
    gui.plot()
    ch0_log = gui.sm.dset.data_display[0].copy()

    # Uncheck: linear — channel 0 should be raw counts (larger values)
    gui.plot_log.setChecked(False)
    gui.plot()
    ch0_lin = gui.sm.dset.data_display[0].copy()

    # Log values are smaller than linear for data > 1 (log10(x) < x for x > 1)
    assert ch0_log.max() < ch0_lin.max(), "log-scaled max should be less than linear max"
    # Channel 0 under log should equal log10 of channel 0 under linear (approximately)
    np.testing.assert_allclose(ch0_log, np.log10(ch0_lin), rtol=1e-5)
```

- [ ] **Step 2: Run the failing test**

```bash
QT_QPA_PLATFORM=offscreen \
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/gui/test_main_window.py::test_log_scale_checkbox_switches_display_mode -v
```

Expected: FAIL — `AssertionError: log-scaled max should be less than linear max` (because `plot_log` is not wired yet, so both calls leave `data_display[0]` unchanged).

- [ ] **Step 3: Add `DISPLAY_FIELD` import to `main_window.py`**

In `src/pysimplemask/gui/control/main_window.py`, add to the existing import block (after the other `pysimplemask` imports):

```python
from pysimplemask.core.reader.base_reader import DISPLAY_FIELD
```

- [ ] **Step 4: Connect `plot_log.stateChanged` to `plot` in `__init__`**

In `src/pysimplemask/gui/control/main_window.py`, locate the line:
```python
        self.checkBox_showxy.setChecked(True)
```

Add immediately after it:
```python
        self.plot_log.setChecked(True)   # log scale on by default
        self.plot_log.stateChanged.connect(lambda _: self.plot())
```

- [ ] **Step 5: Refresh channels 0 and 1 at the start of `plot()`**

In `src/pysimplemask/gui/control/main_window.py`, locate `plot()`:

```python
    def plot(self, reset_view=False):
        if not self.is_ready():
            return
        cmap = self.plot_cmap.currentText()
        plot_center = self.plot_center.isChecked()
```

Replace those opening lines with:

```python
    def plot(self, reset_view=False):
        if not self.is_ready():
            return
        cmap = self.plot_cmap.currentText()
        plot_center = self.plot_center.isChecked()

        # Refresh scattering channels to match the current log/linear selection.
        mode = "log" if self.plot_log.isChecked() else "linear"
        dset = self.sm.dset
        dset.data_display[DISPLAY_FIELD.index("scattering")] = (
            dset.get_scat_with_mask(None, mode=mode)
        )
        dset.data_display[DISPLAY_FIELD.index("scattering * mask")] = (
            dset.get_scat_with_mask(self.sm.mask, mode=mode)
        )
```

- [ ] **Step 6: Run the test to confirm it passes**

```bash
QT_QPA_PLATFORM=offscreen \
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/gui/test_main_window.py::test_log_scale_checkbox_switches_display_mode -v
```

Expected: PASS.

- [ ] **Step 7: Run the full test suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
```

Expected: all 158 tests pass.

- [ ] **Step 8: Ruff check**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check \
    src/pysimplemask/gui/control/main_window.py \
    tests/gui/test_main_window.py
```

Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add \
    src/pysimplemask/gui/control/main_window.py \
    tests/gui/test_main_window.py
git commit -m "feat(gui): wire plot_log checkbox to switch scattering display between log and linear"
```
