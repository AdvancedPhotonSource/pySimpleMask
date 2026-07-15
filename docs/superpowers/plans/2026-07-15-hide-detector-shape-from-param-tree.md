# Hide detector_shape from ParameterTree Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent `detector_shape_x` and `detector_shape_y` from appearing in the GUI's metadata ParameterTree, since they are derived from the image shape and are not user-configurable.

**Architecture:** Add a module-level `_HIDDEN_METADATA_KEYS` frozenset to `base_reader.py` and filter those keys out in `get_parametertree_structure()` before the call to `dict_to_params()`. No changes to `dict_to_params` itself — the filtering lives in the one method that controls what the GUI sees.

**Tech Stack:** Python, existing `pysimplemask.core.reader.base_reader`

## Global Constraints

- Environment: `/local/MQICHU/envs/l2606_simplemask_refact/bin/`
- Only `src/pysimplemask/core/reader/base_reader.py` and `tests/core/reader/test_metadata.py` are modified
- `_HIDDEN_METADATA_KEYS = frozenset({"detector_shape_x", "detector_shape_y"})` — exact name and values
- `detector_shape_x` and `detector_shape_y` must remain in `self.metadata` — only hidden from the tree
- `update_metadata_from_changes` must not be affected (it still writes back any key the user edits via tree, but the hidden keys will simply never appear there)
- Ruff must pass clean

---

## File Map

| Status | File | Change |
|--------|------|--------|
| Modify | `src/pysimplemask/core/reader/base_reader.py` | Add `_HIDDEN_METADATA_KEYS`; filter in `get_parametertree_structure` |
| Modify | `tests/core/reader/test_metadata.py` | Add test verifying the two keys are absent from the tree structure |

---

## Task 1: Filter hidden keys from ParameterTree

**Files:**
- Modify: `src/pysimplemask/core/reader/base_reader.py`
- Test: `tests/core/reader/test_metadata.py`

**Interfaces:**
- `get_parametertree_structure(self) -> dict` — returns a pyqtgraph ParameterTree group definition; after the change, the returned `children` list must contain no entry with `name == "detector_shape_x"` or `name == "detector_shape_y"`
- `_HIDDEN_METADATA_KEYS: frozenset[str]` — module-level constant; value must be `frozenset({"detector_shape_x", "detector_shape_y"})`

- [ ] **Step 1: Write the failing test**

Check whether `tests/core/reader/test_metadata.py` exists; if not, create it. Append this test:

```python
"""Tests for base_reader metadata utilities."""

import numpy as np
import pytest

from pysimplemask.core.reader.base_reader import _HIDDEN_METADATA_KEYS, get_fake_metadata


def test_hidden_metadata_keys_constant():
    """_HIDDEN_METADATA_KEYS must contain exactly the two shape keys."""
    assert "detector_shape_x" in _HIDDEN_METADATA_KEYS
    assert "detector_shape_y" in _HIDDEN_METADATA_KEYS


def test_get_parametertree_structure_hides_shape_keys(tmp_path):
    """detector_shape_x/y must not appear in the ParameterTree structure."""
    import tifffile
    from pysimplemask.core.reader.beamlines.native_files import NativeFilesReader

    p = str(tmp_path / "img.tif")
    tifffile.imwrite(p, np.ones((16, 12), dtype=np.float32))

    reader = NativeFilesReader(p)
    reader.prepare_data()  # sets detector_shape_x/y in metadata

    # Confirm the keys ARE in the underlying metadata dict
    assert "detector_shape_x" in reader.metadata
    assert "detector_shape_y" in reader.metadata

    # Confirm they are NOT in the ParameterTree structure
    struct = reader.get_parametertree_structure()
    child_names = {child["name"] for child in struct["children"]}
    assert "detector_shape_x" not in child_names
    assert "detector_shape_y" not in child_names
```

- [ ] **Step 2: Run the failing test**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/core/reader/test_metadata.py -v
```

Expected: `test_hidden_metadata_keys_constant` FAIL with `ImportError: cannot import name '_HIDDEN_METADATA_KEYS'`; `test_get_parametertree_structure_hides_shape_keys` also FAIL for the same reason.

- [ ] **Step 3: Add `_HIDDEN_METADATA_KEYS` and filter in `get_parametertree_structure`**

In `src/pysimplemask/core/reader/base_reader.py`, add the constant immediately before the `FileReader` class definition:

```python
# Keys that are derived from the image shape and must not appear in the
# GUI's metadata editor.  They remain in self.metadata for qmap computation.
_HIDDEN_METADATA_KEYS: frozenset = frozenset({"detector_shape_x", "detector_shape_y"})
```

Then replace the body of `get_parametertree_structure`:

```python
    def get_parametertree_structure(self):
        visible = {
            k: v for k, v in self.metadata.items()
            if k not in _HIDDEN_METADATA_KEYS
        }
        return dict_to_params("metadata", visible, self.meta_units_fmts)
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/core/reader/test_metadata.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Run the full suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
```

Expected: all existing tests pass plus the 2 new ones.

- [ ] **Step 6: Ruff check**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check \
    src/pysimplemask/core/reader/base_reader.py \
    tests/core/reader/test_metadata.py
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add \
    src/pysimplemask/core/reader/base_reader.py \
    tests/core/reader/test_metadata.py
git commit -m "fix(reader): hide detector_shape_x/y from GUI ParameterTree"
```
