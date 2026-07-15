# Rawdata Frame Browser Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a frame-by-frame browser for raw HDF5 data: when the "rawdata" channel is selected and the loaded file contains `/entry/data/data` with >1 frames, the user can scrub through individual frames via a slider and spinbox with 100 ms debounce.

**Architecture:** "rawdata" sits at `plot_index` index 0 (already added to the UI). Non-rawdata channels now map to `data_display[idx - 1]` (offset by 1). A `QTimer` (100 ms, single-shot) debounces rapid slider moves so h5py only opens the file when the user pauses. Frame controls (`label_frame`, `horizontalSlider_frame`, `spinBox_current_frame`) are hidden by default and shown only when rawdata is active.

**Tech Stack:** Python, PySide6 (QTimer, QComboBox model, QSlider), h5py, existing `SimpleMaskGUI`

## Global Constraints

- Environment: `/local/MQICHU/envs/l2606_simplemask_refact/bin/`
- GUI tests run headless: `QT_QPA_PLATFORM=offscreen`
- `_RAWDATA_IDX = 0` — rawdata is always at `plot_index` position 0 (module-level constant)
- HDF dataset path: `/entry/data/data` (hardcoded, matches what the user specified)
- Debounce interval: exactly 100 ms (`QTimer.setInterval(100)`, single-shot)
- Frame controls shown only when `plot_index.currentIndex() == _RAWDATA_IDX`
- Rawdata item disabled (grayed) by default; enabled only when `/entry/data/data` has > 1 frames
- Non-rawdata channel → `data_display[idx - 1]` (idx is the `plot_index` value)
- Ruff must pass clean

---

## Index offset reference (rawdata added at position 0)

| `plot_index` value | Label | `data_display` slot |
|----|----|----|
| 0 | rawdata | *(special — direct h5py read)* |
| 1 | scattering | `data_display[0]` |
| 2 | scattering * mask | `data_display[1]` |
| 3 | mask | `data_display[2]` |
| 4 | dynamic_q_partition | `data_display[3]` |
| 5 | static_q_partition | `data_display[4]` |
| 6 | preview | `data_display[5]` |
| 7+ | qmap keys | `data_display[6+]` |

---

## File Map

| Status | File | Change |
|--------|------|--------|
| Modify | `src/pysimplemask/gui/view/ui_mask.py` | Regenerate (already done; commit it) |
| Modify | `src/pysimplemask/gui/control/main_window.py` | All rawdata logic + index fixes |
| Modify | `tests/gui/test_main_window.py` | Add tests |

---

## Task 1: Fix hardcoded index offsets + scaffold frame controls

All hardcoded `plot_index` values and `data_display` accesses need to be updated because rawdata was inserted at position 0, shifting everything else by 1.

**Files:**
- Modify: `src/pysimplemask/gui/view/ui_mask.py` (commit regenerated file)
- Modify: `src/pysimplemask/gui/control/main_window.py`
- Modify: `tests/gui/test_main_window.py`

**Interfaces:**
- `_RAWDATA_IDX: int = 0` — module-level constant; used by all rawdata logic
- Frame controls: `self.label_frame`, `self.horizontalSlider_frame`, `self.spinBox_current_frame` — always hidden unless rawdata active

- [ ] **Step 1: Write the failing tests**

Append to `tests/gui/test_main_window.py`:

```python
def test_rawdata_item_disabled_by_default(qapp, tmp_path):
    """rawdata plot_index item is grayed out before any data is loaded."""
    gui = SimpleMaskGUI()
    model = gui.plot_index.model()
    item = model.item(0)   # index 0 = rawdata
    assert item is not None
    assert not item.isEnabled(), "rawdata should be disabled before data load"


def test_frame_controls_hidden_by_default(qapp, tmp_path):
    """Frame controls are invisible on startup."""
    gui = SimpleMaskGUI()
    assert not gui.label_frame.isVisible()
    assert not gui.horizontalSlider_frame.isVisible()
    assert not gui.spinBox_current_frame.isVisible()


def test_non_rawdata_channel_uses_correct_data_display_slice(qapp, tmp_path):
    """plot_index=1 (scattering) shows data_display[0], not data_display[1]."""
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16) * 5)
    # scattering is at plot_index=1, maps to data_display[0]
    gui.plot_index.setCurrentIndex(1)
    # mp1.image should be data_display[0] (the scattering channel)
    assert gui.mp1.image is not None
    expected = gui.sm.dset.data_display[0]
    np.testing.assert_array_equal(gui.mp1.image, expected)
```

- [ ] **Step 2: Run failing tests**

```bash
QT_QPA_PLATFORM=offscreen \
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/gui/test_main_window.py::test_rawdata_item_disabled_by_default \
    tests/gui/test_main_window.py::test_frame_controls_hidden_by_default \
    tests/gui/test_main_window.py::test_non_rawdata_channel_uses_correct_data_display_slice \
    -v
```

Expected: all 3 FAIL.

- [ ] **Step 3: Add `_RAWDATA_IDX` constant and commit `ui_mask.py`**

Add at module level in `src/pysimplemask/gui/control/main_window.py` (alongside the other constants near the top):

```python
_RAWDATA_IDX = 0   # "rawdata" is always at plot_index position 0
```

Add `QTimer` to existing PySide6 core imports:

```python
from PySide6.QtCore import QByteArray, QTimer   # add QTimer
```

- [ ] **Step 4: Hide frame controls in `__init__` and disable rawdata item**

In `__init__`, after `self.setupUi(self)`, add:

```python
# Frame controls are hidden until rawdata channel is active
self.label_frame.setVisible(False)
self.horizontalSlider_frame.setVisible(False)
self.spinBox_current_frame.setVisible(False)

# rawdata item starts disabled (grayed) until a compatible file is loaded
self._set_rawdata_enabled(False)
```

Add the helper method (place alongside other helper methods):

```python
def _set_rawdata_enabled(self, enabled: bool) -> None:
    """Enable or gray out the rawdata item in plot_index."""
    model = self.plot_index.model()
    item = model.item(_RAWDATA_IDX)
    if item:
        item.setEnabled(enabled)
    if not enabled and self.plot_index.currentIndex() == _RAWDATA_IDX:
        self.plot_index.setCurrentIndex(_RAWDATA_IDX + 1)
```

- [ ] **Step 5: Fix all hardcoded `plot_index` values shifted by the rawdata offset**

**`mask_action()` (was 1 → scattering*mask now at 2):**
```python
self.plot_index.setCurrentIndex(2)   # was 1
```

**`mask_evaluate()` (was 5 → preview now at 6):**
```python
self.plot_index.setCurrentIndex(6)   # was 5
```

**`mask_apply_current_tab()` and `mask_apply()` (was 1 → 2):**
```python
self.plot_index.setCurrentIndex(2)   # was 1  (appears at lines ~755 and ~1002)
```

**`compute_partition()` (was 3 → dqmap now at 4):**
```python
self.plot_index.setCurrentIndex(4)   # was 3
```

**`load()` cleanup (was `> 6` → now `> 7` to keep all 7 base items):**
```python
while self.plot_index.count() > 7:
    self.plot_index.removeItem(7)
```

**`_on_plot_index_changed()` binary-level indices (mask=3, preview=6):**
```python
if idx in [3, 6]:   # mask and preview: enforce binary 0/1 scale  (was [2, 5])
```

- [ ] **Step 6: Fix `data_display` and `get_coordinates` index access**

In `plot()`:
```python
idx = self.plot_index.currentIndex()
if idx == _RAWDATA_IDX:
    return   # rawdata handled separately; nothing to show until user selects a frame
channel = idx - 1   # rawdata occupies slot 0; shift back
self.mp1.setImage(self.sm.dset.data_display[channel])
```

In `_on_plot_index_changed()`:
```python
def _on_plot_index_changed(self, idx):
    """Update the displayed 2D slice when the channel selector changes."""
    show_frame_controls = (idx == _RAWDATA_IDX)
    self.label_frame.setVisible(show_frame_controls)
    self.horizontalSlider_frame.setVisible(show_frame_controls)
    self.spinBox_current_frame.setVisible(show_frame_controls)

    if idx == _RAWDATA_IDX:
        return   # rawdata display handled by frame browser callbacks

    if not self.is_ready():
        return
    channel = idx - 1
    vb = self.mp1.getView()
    saved_range = vb.viewRange() if self.mp1.image is not None else None
    self.mp1.setImage(self.sm.dset.data_display[channel])
    if saved_range is not None:
        vb.setRange(xRange=saved_range[0], yRange=saved_range[1], padding=0)
    if idx in [3, 6]:   # mask and preview: enforce binary 0/1 scale
        self.mp1.setLevels(0, 1)
```

In `show_location()`:
```python
idx = self.plot_index.currentIndex()
if idx == _RAWDATA_IDX:
    channel = 0   # use scattering channel for coordinates when rawdata active
else:
    channel = idx - 1
msg = self.sm.dset.get_coordinates(col, row, channel)
```

- [ ] **Step 7: Run the tests**

```bash
QT_QPA_PLATFORM=offscreen \
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/gui/test_main_window.py::test_rawdata_item_disabled_by_default \
    tests/gui/test_main_window.py::test_frame_controls_hidden_by_default \
    tests/gui/test_main_window.py::test_non_rawdata_channel_uses_correct_data_display_slice \
    -v
```

Expected: all 3 PASS.

- [ ] **Step 8: Run full suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
```

Expected: all 163 tests pass.

- [ ] **Step 9: Ruff check**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check \
    src/pysimplemask/gui/control/main_window.py \
    tests/gui/test_main_window.py
```

Expected: no output.

- [ ] **Step 10: Commit**

```bash
git add \
    src/pysimplemask/gui/view/ui_mask.py \
    src/pysimplemask/gui/control/main_window.py \
    tests/gui/test_main_window.py
git commit -m "feat(gui): add rawdata frame browser scaffold; fix plot_index offsets"
```

---

## Task 2: Frame browser — detect, enable, read with debounce

**Files:**
- Modify: `src/pysimplemask/gui/control/main_window.py`
- Modify: `tests/gui/test_main_window.py`

**Interfaces:**
- `_detect_rawdata(self) -> tuple[bool, int]` — returns `(has_rawdata, num_frames)`
- `_on_frame_changed(self, value: int) -> None` — called by both slider and spinbox; syncs the other, restarts timer
- `_read_and_show_frame(self) -> None` — timer callback; opens h5py, reads slice, displays
- `self._frame_timer: QTimer` — single-shot, 100 ms
- `self._pending_frame_idx: int` — last requested frame index

- [ ] **Step 1: Write the failing tests**

Append to `tests/gui/test_main_window.py`:

```python
def test_detect_rawdata_returns_false_for_non_hdf(qapp, tmp_path):
    """_detect_rawdata returns False for non-HDF files."""
    gui = SimpleMaskGUI()
    # No data loaded → not ready
    ok, n = gui._detect_rawdata()
    assert not ok
    assert n == 0


def test_detect_rawdata_returns_false_for_single_frame_hdf(qapp, tmp_path):
    """_detect_rawdata returns False when /entry/data/data has only 1 frame."""
    gui = _load_gui(tmp_path, np.ones((1, 20, 24), dtype=np.uint16))
    ok, n = gui._detect_rawdata()
    assert not ok
    assert n == 0


def test_detect_rawdata_returns_true_for_multi_frame_hdf(qapp, tmp_path):
    """_detect_rawdata returns True when /entry/data/data has > 1 frames."""
    gui = _load_gui(tmp_path, np.ones((5, 20, 24), dtype=np.uint16))
    ok, n = gui._detect_rawdata()
    assert ok
    assert n == 5


def test_rawdata_enabled_after_loading_multi_frame_hdf(qapp, tmp_path):
    """rawdata combobox item is enabled after loading a multi-frame HDF."""
    gui = _load_gui(tmp_path, np.ones((5, 20, 24), dtype=np.uint16))
    model = gui.plot_index.model()
    assert model.item(0).isEnabled(), "rawdata should be enabled for multi-frame HDF"


def test_rawdata_disabled_after_loading_single_frame_hdf(qapp, tmp_path):
    """rawdata combobox item is disabled after loading a single-frame HDF."""
    gui = _load_gui(tmp_path, np.ones((1, 20, 24), dtype=np.uint16))
    model = gui.plot_index.model()
    assert not model.item(0).isEnabled()
```

- [ ] **Step 2: Run failing tests**

```bash
QT_QPA_PLATFORM=offscreen \
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/gui/test_main_window.py::test_detect_rawdata_returns_false_for_non_hdf \
    tests/gui/test_main_window.py::test_detect_rawdata_returns_false_for_single_frame_hdf \
    tests/gui/test_main_window.py::test_detect_rawdata_returns_true_for_multi_frame_hdf \
    tests/gui/test_main_window.py::test_rawdata_enabled_after_loading_multi_frame_hdf \
    tests/gui/test_main_window.py::test_rawdata_disabled_after_loading_single_frame_hdf \
    -v
```

Expected: all 5 FAIL.

- [ ] **Step 3: Add `_detect_rawdata` and timer state to `__init__`**

Add to `__init__` (after existing widget setup):

```python
# Rawdata frame browser state
self._frame_timer = QTimer(self)
self._frame_timer.setSingleShot(True)
self._frame_timer.setInterval(100)   # 100 ms debounce
self._frame_timer.timeout.connect(self._read_and_show_frame)
self._pending_frame_idx = 0
```

Connect slider and spinbox (add after the `plot_index.currentIndexChanged` connection):

```python
self.horizontalSlider_frame.valueChanged.connect(self._on_frame_changed)
self.spinBox_current_frame.valueChanged.connect(self._on_frame_changed)
```

Add the helper method (alongside `_set_rawdata_enabled`):

```python
def _detect_rawdata(self) -> tuple:
    """Return (has_rawdata, num_frames) for the currently loaded file.

    Checks for /entry/data/data with ndim==3 and shape[0] > 1.
    """
    import h5py
    if not self.sm.is_ready():
        return False, 0
    fname = self.sm.dset.fname
    try:
        if not h5py.is_hdf5(fname):
            return False, 0
        with h5py.File(fname, "r") as f:
            if "/entry/data/data" not in f:
                return False, 0
            s = f["/entry/data/data"].shape
            if len(s) != 3 or s[0] <= 1:
                return False, 0
            return True, int(s[0])
    except Exception:
        return False, 0
```

- [ ] **Step 4: Update `load()` to enable/configure rawdata after successful load**

In `load()`, after the existing `self.display_metadata()` call and before `self.plot(reset_view=True)`, add:

```python
has_rawdata, num_frames = self._detect_rawdata()
self._set_rawdata_enabled(has_rawdata)
if has_rawdata:
    self.horizontalSlider_frame.setRange(0, num_frames - 1)
    self.horizontalSlider_frame.setValue(0)
    self.spinBox_current_frame.setRange(0, num_frames - 1)
    self.spinBox_current_frame.setValue(0)
    self._pending_frame_idx = 0
```

- [ ] **Step 5: Implement `_on_frame_changed` and `_read_and_show_frame`**

Add these methods to `SimpleMaskGUI`:

```python
def _on_frame_changed(self, value: int) -> None:
    """Called by slider or spinbox; syncs the other widget and restarts debounce."""
    self._pending_frame_idx = value
    # Sync the other control without triggering a second signal
    sender = self.sender()
    if sender is self.horizontalSlider_frame:
        self.spinBox_current_frame.blockSignals(True)
        self.spinBox_current_frame.setValue(value)
        self.spinBox_current_frame.blockSignals(False)
    else:
        self.horizontalSlider_frame.blockSignals(True)
        self.horizontalSlider_frame.setValue(value)
        self.horizontalSlider_frame.blockSignals(False)
    self._frame_timer.start()   # restart: cancel pending read, schedule new one

def _read_and_show_frame(self) -> None:
    """Timer callback: read one raw frame from HDF5 and display it."""
    import h5py
    if not self.sm.is_ready():
        return
    idx = self._pending_frame_idx
    fname = self.sm.dset.fname
    try:
        with h5py.File(fname, "r") as f:
            frame = f["/entry/data/data"][idx].astype(np.float32)
    except Exception as exc:
        logger.error("Failed to read frame %d from %s: %s", idx, fname, exc)
        return
    vb = self.mp1.getView()
    saved_range = vb.viewRange() if self.mp1.image is not None else None
    self.mp1.setImage(frame)
    if saved_range is not None:
        vb.setRange(xRange=saved_range[0], yRange=saved_range[1], padding=0)
```

- [ ] **Step 6: Run the new tests**

```bash
QT_QPA_PLATFORM=offscreen \
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/gui/test_main_window.py::test_detect_rawdata_returns_false_for_non_hdf \
    tests/gui/test_main_window.py::test_detect_rawdata_returns_false_for_single_frame_hdf \
    tests/gui/test_main_window.py::test_detect_rawdata_returns_true_for_multi_frame_hdf \
    tests/gui/test_main_window.py::test_rawdata_enabled_after_loading_multi_frame_hdf \
    tests/gui/test_main_window.py::test_rawdata_disabled_after_loading_single_frame_hdf \
    -v
```

Expected: all 5 PASS.

- [ ] **Step 7: Run the full test suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
```

Expected: all tests pass (163 existing + 8 new).

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
git commit -m "feat(gui): rawdata frame browser with 100ms debounce and slider/spinbox sync"
```
