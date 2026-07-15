# Single Mask Apply Button Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the new shared `btn_mask_apply` button to apply the correct mask for whatever `MaskWidget` tab is currently active, and remove all dead signal connections and tooltip calls for the deleted per-tab apply buttons.

**Architecture:** A module-level dispatch table `_TAB_MASK_TARGETS` maps each `MaskWidget` tab index to one or more mask target strings. `mask_apply_current_tab()` reads the current tab index, looks it up in the table, and calls the existing `mask_apply(target)` for each entry. Dead connections and tooltips for the seven removed buttons (`btn_mask_blemish_apply`, `btn_mask_file_apply`, `btn_mask_draw_apply`, `btn_mask_threshold_apply`, `btn_mask_list_apply`, `btn_mask_outlier_apply`, `btn_mask_param_apply`) are deleted to prevent `AttributeError` at startup.

**Tech Stack:** Python, PySide6, existing `SimpleMaskGUI.mask_apply(target)`

## Global Constraints

- Environment: `/local/MQICHU/envs/l2606_simplemask_refact/bin/`
- GUI tests run headless: `QT_QPA_PLATFORM=offscreen`
- Tab dispatch table (exact):
  - 0 = Blemish/Files → `("mask_blemish", "mask_file")` (both; unevaluated worker is a no-op)
  - 1 = Draw → `("mask_draw",)`
  - 2 = Binary → `("mask_threshold",)`
  - 3 = Manual → `("mask_list",)`
  - 4 = Outlier → `("mask_outlier",)`
  - 5 = Parametrization → `("mask_parameter",)`
- `_TAB_MASK_TARGETS` is a module-level list of tuples defined in `main_window.py`
- `btn_mask_apply` is already declared in `ui_mask.py` — do not add it; only wire it
- All dead connections and tooltip calls for the seven deleted buttons must be removed
- Ruff must pass clean

---

## File Map

| Status | File | Change |
|--------|------|--------|
| Modify | `src/pysimplemask/gui/control/main_window.py` | Remove dead wiring; add dispatch table + `mask_apply_current_tab`; connect `btn_mask_apply`; add tooltip |
| Modify | `tests/gui/test_main_window.py` | Add dispatch test |

---

## Dead code to remove

The following lines reference buttons deleted from the UI and **must be removed** (app crashes at startup otherwise):

**Signal connections in `__init__`** (exact lines as of writing; verify with `grep -n` before editing):

```python
# Remove these 7 lines/blocks:
self.btn_mask_list_apply.clicked.connect(lambda: self.mask_apply("mask_list"))

self.btn_mask_blemish_apply.clicked.connect(
    lambda: self.mask_apply("mask_blemish")
)

self.btn_mask_file_apply.clicked.connect(lambda: self.mask_apply("mask_file"))

self.btn_mask_draw_apply.clicked.connect(lambda: self.mask_apply("mask_draw"))

self.btn_mask_threshold_apply.clicked.connect(
    lambda: self.mask_apply("mask_threshold")
)

self.btn_mask_outlier_apply.clicked.connect(
    lambda: self.mask_apply("mask_outlier")
)

self.btn_mask_param_apply.clicked.connect(
    lambda: self.mask_apply("mask_parameter")
)
```

**Tooltip calls** (also in `__init__`, in the tooltip-setup section):

```python
# Remove these 7 calls:
self.btn_mask_blemish_apply.setToolTip(...)   # "AND the blemish mask..."
self.btn_mask_file_apply.setToolTip(...)       # "AND the imported mask..."
self.btn_mask_draw_apply.setToolTip(...)       # "Apply drawn shapes..."
self.btn_mask_threshold_apply.setToolTip(...)  # "Apply threshold and morphology..."
self.btn_mask_list_apply.setToolTip(...)       # "Apply the pixel list..."
self.btn_mask_outlier_apply.setToolTip(...)    # "Apply outlier mask..."
self.btn_mask_param_apply.setToolTip(...)      # "Apply the parameter constraints..."
```

---

## Task 1: Wire `btn_mask_apply` + remove dead code

**Files:**
- Modify: `src/pysimplemask/gui/control/main_window.py`
- Modify: `tests/gui/test_main_window.py`

**Interfaces:**
- `_TAB_MASK_TARGETS: list[tuple[str, ...]]` — module-level, 6 entries, index matches `MaskWidget.currentIndex()`
- `SimpleMaskGUI.mask_apply_current_tab(self) -> None` — reads current tab index, dispatches to `mask_apply(target)` for each entry

- [ ] **Step 1: Write the failing test**

Append to `tests/gui/test_main_window.py`:

```python
def test_mask_apply_current_tab_dispatches_correctly(qapp, tmp_path):
    """mask_apply_current_tab calls mask_apply with the target for the active tab."""
    from unittest.mock import patch

    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    calls = []

    with patch.object(gui, "mask_apply", side_effect=lambda t: calls.append(t)):
        # Tab 0: Blemish/Files → both mask_blemish and mask_file
        gui.MaskWidget.setCurrentIndex(0)
        gui.mask_apply_current_tab()
        assert calls == ["mask_blemish", "mask_file"], f"Tab 0: got {calls}"
        calls.clear()

        # Tab 2: Binary → mask_threshold
        gui.MaskWidget.setCurrentIndex(2)
        gui.mask_apply_current_tab()
        assert calls == ["mask_threshold"], f"Tab 2: got {calls}"
        calls.clear()

        # Tab 5: Parametrization → mask_parameter
        gui.MaskWidget.setCurrentIndex(5)
        gui.mask_apply_current_tab()
        assert calls == ["mask_parameter"], f"Tab 5: got {calls}"


def test_tab_mask_targets_constant(qapp, tmp_path):
    """_TAB_MASK_TARGETS has one entry per MaskWidget tab (6 total)."""
    from pysimplemask.gui.control.main_window import _TAB_MASK_TARGETS
    gui = _load_gui(tmp_path, np.ones((3, 20, 24), dtype=np.uint16))
    assert len(_TAB_MASK_TARGETS) == gui.MaskWidget.count()
```

- [ ] **Step 2: Run the failing tests**

```bash
QT_QPA_PLATFORM=offscreen \
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/gui/test_main_window.py::test_mask_apply_current_tab_dispatches_correctly \
    tests/gui/test_main_window.py::test_tab_mask_targets_constant \
    -v
```

Expected: both FAIL — `ImportError` or `AttributeError: 'SimpleMaskGUI' object has no attribute 'mask_apply_current_tab'`.

- [ ] **Step 3: Remove dead signal connections from `main_window.py`**

In `src/pysimplemask/gui/control/main_window.py`, remove the seven dead `.clicked.connect(...)` blocks listed in the "Dead code to remove" section above. Leave all other connections (evaluate, load, clear, add, etc.) intact.

After removal the `# mask_list`, `# blemish`, `# mask_file`, `# draw method / array`, `# binary threshold`, `# btn_mask_outlier_evaluate`, and `# xmap constraint` sections should contain only evaluate/non-apply connections.

- [ ] **Step 4: Remove dead tooltip calls from `main_window.py`**

Remove the seven `.setToolTip(...)` calls for the deleted buttons listed in the "Dead code to remove" section above. Leave all other tooltip calls intact.

- [ ] **Step 5: Add `_TAB_MASK_TARGETS` and `mask_apply_current_tab`**

Add the dispatch table at module level in `src/pysimplemask/gui/control/main_window.py`, just before the `class SimpleMaskGUI` definition:

```python
# Maps MaskWidget tab index → mask target(s) applied by btn_mask_apply.
# Tab 0 (Blemish/Files) applies both; an unevaluated worker is a safe no-op.
_TAB_MASK_TARGETS: list = [
    ("mask_blemish", "mask_file"),  # 0: Blemish/Files
    ("mask_draw",),                  # 1: Draw
    ("mask_threshold",),             # 2: Binary
    ("mask_list",),                  # 3: Manual
    ("mask_outlier",),               # 4: Outlier
    ("mask_parameter",),             # 5: Parametrization
]
```

Then add the method to `SimpleMaskGUI` (alongside the other `mask_*` methods, e.g. after `mask_action`):

```python
def mask_apply_current_tab(self):
    """Apply the mask(s) for the currently active MaskWidget tab."""
    if not self.is_ready():
        return
    idx = self.MaskWidget.currentIndex()
    if idx < 0 or idx >= len(_TAB_MASK_TARGETS):
        return
    for target in _TAB_MASK_TARGETS[idx]:
        self.mask_apply(target)
```

- [ ] **Step 6: Connect `btn_mask_apply` and add its tooltip**

In `__init__`, add the connection (near the other mask button connections):

```python
self.btn_mask_apply.clicked.connect(self.mask_apply_current_tab)
```

In the tooltip-setup section, add:

```python
self.btn_mask_apply.setToolTip(
    "Apply the evaluated mask for the currently active tab into the working mask"
)
```

- [ ] **Step 7: Run the new tests**

```bash
QT_QPA_PLATFORM=offscreen \
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/gui/test_main_window.py::test_mask_apply_current_tab_dispatches_correctly \
    tests/gui/test_main_window.py::test_tab_mask_targets_constant \
    -v
```

Expected: both PASS.

- [ ] **Step 8: Run the full test suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
```

Expected: all tests pass (161 + 2 new = 163).

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
    src/pysimplemask/gui/control/main_window.py \
    tests/gui/test_main_window.py
git commit -m "feat(gui): wire btn_mask_apply to active tab; remove deleted per-tab apply buttons"
```
