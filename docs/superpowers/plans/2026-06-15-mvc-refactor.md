# MVC Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `src/pysimplemask/` into a Qt-free, scriptable `core/` and a Qt-only `gui/` (view/model/control), so the masking/partition pipeline runs headless from Python and the GUI is a thin display+input layer over `core`.

**Architecture:** `core/` holds all domain logic and produces plain numpy. `core/model.py::SimpleMaskModel` is the scriptable entry (no Qt). ROI shapes are rasterized in `core/rasterize.py` with skimage from plain polygon vertices; the GUI extracts those vertices from pyqtgraph ROIs (`gui/model/roi_extract.py`). `gui/control/main_window.py` wires the `Ui_SimpleMask` widgets (generated from `gui/view/mask.ui`) to the model.

**Tech Stack:** Python 3.12, numpy, scipy, scikit-image, h5py, tifffile, PySide6, pyqtgraph. Env: `/local/MQICHU/envs/l2606_simplemask_refact/bin` (alias `PY` below). Tooling: pytest, ruff. `pyside6-uic` regenerates UI.

**Conventions used in this plan:**
- `PY=/local/MQICHU/envs/l2606_simplemask_refact/bin/python`
- Repo root: `/home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact`
- "copy then edit" tasks: use `git mv` to preserve history, then apply the listed edits.
- Spec: `docs/superpowers/specs/2026-06-15-mvc-refactor-design.md`.

---

## File structure (target)

```
src/pysimplemask/
  __init__.py            # Qt-free; __version__ + core.SimpleMaskModel
  cli.py
  core/
    __init__.py          # re-exports SimpleMaskModel, get_handler
    model.py             # SimpleMaskModel (was simplemask_kernel.SimpleMask)
    file_handler.py
    reader/              # moved unchanged
    qmap.py
    mask.py              # was area_mask.py
    partition.py         # was utils.py (+ least_multiple)
    rasterize.py         # NEW
    io.py                # NEW (text_to_array, load_pixel_list)
    find_center.py  outlier_removal.py  ellipse_util.py
  gui/
    __init__.py
    app.py               # main_gui(path)
    view/
      __init__.py
      mask.ui            # was ui/mask.ui  (header .pyqtgraph_mod -> .widgets)
      ui_mask.py         # regenerated (was simplemask_ui.py)
      widgets.py         # was pyqtgraph_mod.py
    model/
      __init__.py
      table_model.py     # was table_model.py
      roi_extract.py     # NEW
    control/
      __init__.py
      main_window.py     # was simplemask_main.py (SimpleMaskGUI)
  resources/logo.svg
tests/
  core/
    __init__.py
    reader/              # moved from tests/reader/
    test_io.py           # NEW
    test_rasterize.py    # NEW
    test_model_headless.py  # NEW
```

---

## Task 1: Create package skeleton

**Files:**
- Create: `src/pysimplemask/core/__init__.py`, `src/pysimplemask/gui/__init__.py`, `src/pysimplemask/gui/view/__init__.py`, `src/pysimplemask/gui/model/__init__.py`, `src/pysimplemask/gui/control/__init__.py`, `tests/core/__init__.py`

- [ ] **Step 1: Create empty package dirs + init files**

```bash
cd /home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact
mkdir -p src/pysimplemask/core src/pysimplemask/gui/view src/pysimplemask/gui/model src/pysimplemask/gui/control tests/core
: > src/pysimplemask/gui/view/__init__.py
: > src/pysimplemask/gui/model/__init__.py
: > src/pysimplemask/gui/control/__init__.py
: > tests/core/__init__.py
printf '"""GUI package (Qt)."""\n' > src/pysimplemask/gui/__init__.py
```

(`core/__init__.py` and `gui/app.py` are written in later tasks once their targets exist.)

- [ ] **Step 2: Commit**

```bash
git add -A
git commit -m "chore: scaffold core/ and gui/ package skeleton"
```

---

## Task 2: Move the Qt-free compute modules into core/

These modules have no intra-package imports except `reader` (uses `..qmap`) and `file_handler` (uses `.reader`); both relationships are preserved by moving them together into `core/`.

**Files:**
- Move: `reader/` → `core/reader/`; `qmap.py` → `core/qmap.py`; `area_mask.py` → `core/mask.py`; `utils.py` → `core/partition.py`; `file_handler.py` → `core/file_handler.py`; `find_center.py`, `outlier_removal.py`, `ellipse_util.py` → `core/`
- Move: `tests/reader/` → `tests/core/reader/`

- [ ] **Step 1: git mv the modules**

```bash
cd /home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact/src/pysimplemask
git mv reader core/reader
git mv qmap.py core/qmap.py
git mv area_mask.py core/mask.py
git mv utils.py core/partition.py
git mv file_handler.py core/file_handler.py
git mv find_center.py core/find_center.py
git mv outlier_removal.py core/outlier_removal.py
git mv ellipse_util.py core/ellipse_util.py
```

- [ ] **Step 2: Move the reader tests and update their import roots**

```bash
cd /home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact
git mv tests/reader tests/core/reader
grep -rl "pysimplemask.reader" tests/core/reader | xargs sed -i 's/pysimplemask\.reader/pysimplemask.core.reader/g'
```

- [ ] **Step 3: Write `core/__init__.py`** (model added in Task 7; for now expose dispatch)

File `src/pysimplemask/core/__init__.py`:

```python
"""Qt-free core: domain model, readers, and compute utilities."""

from .file_handler import get_handler

__all__ = ["get_handler"]
```

- [ ] **Step 4: Verify reader tests still pass from the new location**

Run: `PY=/local/MQICHU/envs/l2606_simplemask_refact/bin/python; $PY -m pytest tests/core/reader -q`
Expected: `52 passed`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: move Qt-free compute modules into core/"
```

---

## Task 3: Add `least_multiple` to `core/partition.py`

**Files:**
- Modify: `src/pysimplemask/core/partition.py` (append a helper)
- Test: `tests/core/test_partition_helpers.py`

- [ ] **Step 1: Write the failing test**

File `tests/core/test_partition_helpers.py`:

```python
import pytest

from pysimplemask.core.partition import least_multiple


@pytest.mark.parametrize(
    "a,b,expected",
    [(10, 100, 100), (10, 95, 100), (36, 360, 360), (36, 350, 360), (7, 1, 7)],
)
def test_least_multiple(a, b, expected):
    assert least_multiple(a, b) == expected
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/core/test_partition_helpers.py -q`
Expected: FAIL with `ImportError: cannot import name 'least_multiple'`.

- [ ] **Step 3: Add the helper**

Append to `src/pysimplemask/core/partition.py`:

```python
def least_multiple(a: int, b: int) -> int:
    """Smallest multiple of ``a`` that is >= ``b`` (used to align static to dynamic bins)."""
    return ((b + a - 1) // a) * a
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/core/test_partition_helpers.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): add least_multiple partition helper"
```

---

## Task 4: Create `core/io.py` (pixel-list parsing)

Extracts `text_to_array` (was in `simplemask_main.py`) and `load_pixel_list` (the json/txt/csv parsing from `simplemask_main.mask_list_load`). Both return arrays of `[x, y]` rows (no row/col roll, no 1-based shift — those stay GUI-side toggles).

**Files:**
- Create: `src/pysimplemask/core/io.py`
- Test: `tests/core/test_io.py`

- [ ] **Step 1: Write the failing test**

File `tests/core/test_io.py`:

```python
import json

import numpy as np

from pysimplemask.core.io import load_pixel_list, text_to_array


def test_text_to_array_ints():
    out = text_to_array("[1, 2], [3, 4]")
    assert out.tolist() == [1, 2, 3, 4]


def test_text_to_array_floats():
    out = text_to_array("1.5 2.5", dtype=np.float64)
    assert np.allclose(out, [1.5, 2.5])


def test_load_pixel_list_csv(tmp_path):
    p = tmp_path / "pts.csv"
    p.write_text("1,2\n3,4\n")
    out = load_pixel_list(str(p))
    assert out.tolist() == [[1, 2], [3, 4]]


def test_load_pixel_list_txt_space(tmp_path):
    p = tmp_path / "pts.txt"
    p.write_text("1 2\n3 4\n")
    out = load_pixel_list(str(p))
    assert out.tolist() == [[1, 2], [3, 4]]


def test_load_pixel_list_json(tmp_path):
    p = tmp_path / "pts.json"
    p.write_text(json.dumps({"Bad pixels": [{"Pixel": [1, 2]}, {"Pixel": [3, 4]}]}))
    out = load_pixel_list(str(p))
    assert out.tolist() == [[1, 2], [3, 4]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/core/test_io.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pysimplemask.core.io'`.

- [ ] **Step 3: Write `core/io.py`**

File `src/pysimplemask/core/io.py`:

```python
"""Parsing helpers for pixel-coordinate input (text and files)."""

import json

import numpy as np


def text_to_array(pts, dtype=np.int64):
    """Parse a free-form string of numbers (brackets/commas ignored) into a 1-D array."""
    for symbol in "[](),":
        pts = pts.replace(symbol, " ")
    tokens = [tok for tok in pts.split(" ") if tok != ""]
    if dtype == np.int64:
        values = [int(tok) for tok in tokens]
    elif dtype == np.float64:
        values = [float(tok) for tok in tokens]
    else:
        values = [dtype(tok) for tok in tokens]
    return np.array(values).astype(dtype)


def load_pixel_list(fname):
    """Load an (N, 2) array of [x, y] pixel coordinates from .json/.txt/.csv.

    JSON format: ``{"Bad pixels": [{"Pixel": [x, y]}, ...]}``.
    Text/CSV: two columns, comma- or whitespace-separated.
    """
    if fname.endswith(".json"):
        with open(fname, "r") as f:
            entries = json.load(f)["Bad pixels"]
        xy = np.array([entry["Pixel"] for entry in entries])
    elif fname.endswith(".txt") or fname.endswith(".csv"):
        try:
            xy = np.loadtxt(fname, delimiter=",")
        except ValueError:
            xy = np.loadtxt(fname)
    else:
        raise ValueError(f"unsupported pixel-list file: {fname}")
    return xy.astype(np.int64).reshape(-1, 2)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/core/test_io.py -q`
Expected: `5 passed`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): add io.py (text_to_array, load_pixel_list)"
```

---

## Task 5: Create `core/rasterize.py` (geometry → mask, skimage)

Produces the **keep-mask** (True = keep) with the same combination logic as the old Qt
`evaluate_drawing`: `keep = (NOT any exclusive) AND (any inclusive, or all-True if none)`.
All shapes reduce to polygon vertices in image space `(row, col)`.

**Files:**
- Create: `src/pysimplemask/core/rasterize.py`
- Test: `tests/core/test_rasterize.py`

- [ ] **Step 1: Write the failing test**

File `tests/core/test_rasterize.py`:

```python
import numpy as np

from pysimplemask.core.rasterize import (
    RoiPolygon,
    circle_vertices,
    rasterize,
    rectangle_vertices,
)


def test_no_rois_keeps_everything():
    keep = rasterize((10, 10), [])
    assert keep.all()


def test_exclusive_rectangle_removes_region():
    verts = rectangle_vertices(center=(5, 5), size=(4, 4), angle_deg=0.0)
    keep = rasterize((10, 10), [RoiPolygon(verts, "exclusive")])
    # center pixel is inside the excluded rectangle -> removed
    assert not keep[5, 5]
    # a far corner stays kept
    assert keep[0, 0]


def test_inclusive_only_keeps_inside():
    verts = rectangle_vertices(center=(5, 5), size=(4, 4), angle_deg=0.0)
    keep = rasterize((10, 10), [RoiPolygon(verts, "inclusive")])
    assert keep[5, 5]
    assert not keep[0, 0]


def test_circle_vertices_form_disk():
    verts = circle_vertices(center=(10, 10), radius=5, n=180)
    keep = rasterize((20, 20), [RoiPolygon(verts, "inclusive")])
    assert keep[10, 10]
    assert not keep[0, 0]
    # roughly pi r^2 pixels kept
    assert abs(keep.sum() - np.pi * 25) < 25
```

- [ ] **Step 2: Run test to verify it fails**

Run: `$PY -m pytest tests/core/test_rasterize.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'pysimplemask.core.rasterize'`.

- [ ] **Step 3: Write `core/rasterize.py`**

File `src/pysimplemask/core/rasterize.py`:

```python
"""Rasterize ROI geometry to a boolean keep-mask (Qt-free, skimage-based)."""

from dataclasses import dataclass

import numpy as np
from skimage.draw import polygon2mask


@dataclass
class RoiPolygon:
    """A polygon ROI in image space.

    vertices: (N, 2) array of (row, col) points.
    mode: "inclusive" (region to keep) or "exclusive" (region to remove).
    """

    vertices: np.ndarray
    mode: str = "exclusive"


def circle_vertices(center, radius, n=180):
    """Vertices of a circle. center=(row, col)."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rows = center[0] + radius * np.sin(theta)
    cols = center[1] + radius * np.cos(theta)
    return np.column_stack([rows, cols])


def ellipse_vertices(center, axes, angle_deg=0.0, n=180):
    """Vertices of a (possibly rotated) ellipse. center=(row, col), axes=(a_row, a_col)."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = axes[0] * np.sin(theta)
    c = axes[1] * np.cos(theta)
    a = np.deg2rad(angle_deg)
    rr = center[0] + r * np.cos(a) - c * np.sin(a)
    cc = center[1] + r * np.sin(a) + c * np.cos(a)
    return np.column_stack([rr, cc])


def rectangle_vertices(center, size, angle_deg=0.0):
    """Four corners of a (possibly rotated) rectangle. center=(row, col), size=(h, w)."""
    h, w = size[0] / 2.0, size[1] / 2.0
    corners = np.array([[-h, -w], [-h, w], [h, w], [h, -w]])
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rotated = corners @ rot.T
    return rotated + np.asarray(center)


def line_vertices(p0, p1, width):
    """A rectangle polygon of given width along the segment p0->p1. Points are (row, col)."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    direction = p1 - p0
    length = np.hypot(*direction)
    if length == 0:
        normal = np.array([0.0, 0.0])
    else:
        normal = np.array([-direction[1], direction[0]]) / length * (width / 2.0)
    return np.array([p0 + normal, p1 + normal, p1 - normal, p0 - normal])


def rasterize(shape, rois):
    """Combine ROI polygons into a boolean keep-mask of the given image shape.

    keep = (NOT any exclusive) AND (any inclusive OR no inclusive present).
    """
    exclusive = np.zeros(shape, dtype=bool)
    inclusive = np.zeros(shape, dtype=bool)
    has_inclusive = False

    for roi in rois:
        filled = polygon2mask(shape, np.asarray(roi.vertices, dtype=float))
        if roi.mode == "inclusive":
            has_inclusive = True
            inclusive |= filled
        else:
            exclusive |= filled

    keep_inclusive = inclusive if has_inclusive else np.ones(shape, dtype=bool)
    return np.logical_and(~exclusive, keep_inclusive)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `$PY -m pytest tests/core/test_rasterize.py -q`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat(core): add rasterize.py (geometry -> keep-mask via skimage)"
```

---

## Task 6: Create the headless `core/model.py`

Move `simplemask_kernel.py` to `core/model.py`, rename `SimpleMask` → `SimpleMaskModel`,
remove all Qt, and add geometry helpers + `_evaluate_draw_mask`.

**Files:**
- Move: `src/pysimplemask/simplemask_kernel.py` → `src/pysimplemask/core/model.py`
- Modify: `src/pysimplemask/core/model.py`, `src/pysimplemask/core/__init__.py`
- Test: `tests/core/test_model_headless.py`

- [ ] **Step 1: Move the file**

```bash
cd /home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact/src/pysimplemask
git mv simplemask_kernel.py core/model.py
```

- [ ] **Step 2: Replace the imports + class header**

In `core/model.py`, replace the import block (top of file through the `logger =` line) with:

```python
import logging
import os
import time

import h5py
import numpy as np
import tifffile

from pysimplemask import __version__

from .mask import MaskAssemble
from .file_handler import get_handler
from .find_center import find_center
from .outlier_removal import outlier_removal_with_saxs
from .partition import (
    check_consistency,
    combine_partitions,
    generate_partition,
    hash_numpy_dict,
    least_multiple,
    optimize_integer_array,
)
from .ellipse_util import compute_ellipse_gradient, find_ellipse_parameters
from .rasterize import (
    RoiPolygon,
    circle_vertices,
    ellipse_vertices,
    line_vertices,
    rasterize,
    rectangle_vertices,
)

logger = logging.getLogger(__name__)
```

(Removed: `pyqtgraph`, `pyqtgraph.Qt`, `pyqtgraph_mod.LineROI`, `pg.setConfigOptions`.)

- [ ] **Step 3: Replace `__init__` and delete Qt methods**

Replace the class definition `class SimpleMask(object):` and its `__init__` with:

```python
class SimpleMaskModel(object):
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
```

Then **delete these methods entirely** (they were Qt/pyqtgraph view code):
`show_location`, `show_saxs`, `evaluate_drawing`, `add_drawing`, `remove_roi`.

- [ ] **Step 4: Remove the view side-effect in `compute_partition_general`**

In `compute_partition_general`, delete the line:

```python
        self.hdl.setCurrentIndex(3)
```

- [ ] **Step 5: Make `read_data` -> `load` and keep get_center robust**

Rename method `read_data` to `load` (signature unchanged). In `get_center`, the body is unchanged but remove the bare `assert` in favor of the model contract:

```python
    def get_center(self, mode="xy"):
        if self.dset is None:
            return (None, None)
        if mode not in ("xy", "vh"):
            raise ValueError(f"mode must be 'xy' or 'vh', got {mode!r}")
        return self.dset.get_center(mode=mode)
```

- [ ] **Step 6: Add ROI geometry helpers + draw rasterization**

Add these methods to `SimpleMaskModel` (replace the body of any old `mask_draw` handling that referenced `evaluate_drawing`):

```python
    def clear_rois(self):
        self.draw_rois = []

    def add_polygon(self, vertices, mode="exclusive"):
        self.draw_rois.append(RoiPolygon(np.asarray(vertices, dtype=float), mode))

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
```

- [ ] **Step 7: Route `mask_draw` through the core rasterizer**

The GUI previously passed `arr=np.logical_not(evaluate_drawing())` into `mask_evaluate`.
Keep `mask_evaluate(target, **kwargs)` as-is, but add a convenience so callers don't pass `arr`
for draw masks. Add this method:

```python
    def evaluate_draw(self):
        """Evaluate the 'mask_draw' worker from the current draw ROIs."""
        keep = self.evaluate_draw_mask()
        return self.mask_evaluate("mask_draw", arr=np.logical_not(keep))
```

- [ ] **Step 8: Delete the `test01` module-main block**

Remove the `def test01():` function and the `if __name__ == "__main__":` block at the bottom.

- [ ] **Step 9: Export from `core/__init__.py`**

Replace `src/pysimplemask/core/__init__.py` with:

```python
"""Qt-free core: domain model, readers, and compute utilities."""

from .file_handler import get_handler
from .model import SimpleMaskModel

__all__ = ["SimpleMaskModel", "get_handler"]
```

- [ ] **Step 10: Write the headless model test**

File `tests/core/test_model_headless.py`:

```python
import subprocess
import sys

import numpy as np

from pysimplemask.core import SimpleMaskModel
from pysimplemask.core.reader.beamlines.aps_8idi import METADATA_KEYMAPS


def _write_dataset(tmp_path, make_hdf):
    # 4 frames of 16x12 with a hot strip so thresholding has an effect
    frames = np.zeros((4, 16, 12), dtype=np.uint16)
    frames[:, 8, :] = 100
    return make_hdf(frames, name="scan.h5")


def test_model_load_threshold_partition_save(tmp_path, make_hdf):
    path = _write_dataset(tmp_path, make_hdf)
    m = SimpleMaskModel()
    assert m.load(path, beamline="APS_8IDI", num_frames=0) is True
    assert m.shape == (16, 12)

    # threshold mask then a polygon mask, fully headless
    m.evaluate_mask("mask_threshold", low=0, high=50,
                    low_enable=False, high_enable=True)
    m.apply_mask("mask_threshold")
    m.add_polygon([(0, 0), (0, 4), (4, 4), (4, 0)], mode="exclusive")
    m.evaluate_draw()
    m.apply_mask("mask_draw")
    assert m.mask.shape == (16, 12)
    assert not m.mask[1, 1]  # inside the excluded polygon

    out_mask = tmp_path / "mask.tif"
    m.save_mask(str(out_mask))
    assert out_mask.exists()

    m.compute_partition(mode="q-phi", dq_num=2, sq_num=4, dp_num=4, sp_num=8)
    out_qmap = tmp_path / "qmap.hdf"
    m.save_partition(str(out_qmap))
    assert out_qmap.exists()


def test_importing_core_does_not_import_qt():
    code = "import pysimplemask.core, sys; print('PySide6' in sys.modules)"
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == "False"
```

Note: this test reuses the `make_hdf` fixture from `tests/core/reader/conftest.py`. Add a
re-export so it is visible to `tests/core/`:

File `tests/core/conftest.py`:

```python
from pysimplemask.core.reader.conftest import *  # noqa: F401,F403
```

If that import path does not resolve (conftest files are not importable as modules), instead
copy the `make_hdf` fixture definition into `tests/core/conftest.py`:

```python
import h5py
import numpy as np
import pytest


@pytest.fixture
def make_hdf(tmp_path):
    def _make(frames, data_path="/entry/data/data", name="data.h5"):
        path = tmp_path / name
        with h5py.File(path, "w") as h:
            h[data_path] = np.asarray(frames)
        return str(path)

    return _make
```

(Use the copy form — conftest modules are not importable. Delete the re-export variant.)

- [ ] **Step 11: Run the headless tests**

Run: `$PY -m pytest tests/core/test_model_headless.py -q`
Expected: `2 passed`. The second test proves `import pysimplemask.core` pulls in no Qt.

- [ ] **Step 12: Commit**

```bash
git add -A
git commit -m "refactor(core): headless SimpleMaskModel decoupled from Qt"
```

---

## Task 7: Move the view layer (widgets + UI)

**Files:**
- Move: `pyqtgraph_mod.py` → `gui/view/widgets.py`; `ui/mask.ui` → `gui/view/mask.ui`; `simplemask_ui.py` → `gui/view/ui_mask.py`
- Modify: `gui/view/mask.ui` (promoted-widget header), then regenerate `ui_mask.py`

- [ ] **Step 1: Move the files**

```bash
cd /home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact/src/pysimplemask
git mv pyqtgraph_mod.py gui/view/widgets.py
git mv ui/mask.ui gui/view/mask.ui
git mv simplemask_ui.py gui/view/ui_mask.py
rmdir ui 2>/dev/null || true
```

- [ ] **Step 2: Update the promoted-widget header in mask.ui**

In `gui/view/mask.ui`, change the `ImageViewROI` custom widget header:

```
   <header>.pyqtgraph_mod</header>
```
to:
```
   <header>.widgets</header>
```

- [ ] **Step 3: Regenerate `ui_mask.py` from the updated .ui**

```bash
cd /home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact
/local/MQICHU/envs/l2606_simplemask_refact/bin/pyside6-uic \
  src/pysimplemask/gui/view/mask.ui -o src/pysimplemask/gui/view/ui_mask.py
```

Verify the generated file now contains `from .widgets import ImageViewROI` (not `.pyqtgraph_mod`):

Run: `grep -n "widgets import ImageViewROI" src/pysimplemask/gui/view/ui_mask.py`
Expected: one match.

- [ ] **Step 4: Smoke-test the view imports (offscreen)**

Run:
```bash
QT_QPA_PLATFORM=offscreen $PY -c "from pysimplemask.gui.view.ui_mask import Ui_SimpleMask; from pysimplemask.gui.view.widgets import ImageViewROI; print('ok')"
```
Expected: `ok`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(gui): move view widgets + regenerate ui_mask from .ui"
```

---

## Task 8: Move table model + create roi_extract

**Files:**
- Move: `table_model.py` → `gui/model/table_model.py`
- Create: `src/pysimplemask/gui/model/roi_extract.py`

- [ ] **Step 1: Move the table model**

```bash
cd /home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact/src/pysimplemask
git mv table_model.py gui/model/table_model.py
```

- [ ] **Step 2: Write `roi_extract.py`**

Converts pyqtgraph ROI items on the image widget into `core.rasterize.RoiPolygon` objects in
image-space `(row, col)` coordinates, mirroring the old Qt path logic (true ellipse path, map to
image item) but emitting vertices instead of rasterizing.

File `src/pysimplemask/gui/model/roi_extract.py`:

```python
"""Extract pyqtgraph ROI geometry into Qt-free core polygons."""

import numpy as np
import pyqtgraph as pg
from PySide6.QtGui import QPainterPath

from pysimplemask.core.rasterize import RoiPolygon


def extract_roi_geometry(roi_dict, image_item):
    """Return a list of RoiPolygon for every 'roi_*' item in ``roi_dict``.

    roi_dict: mapping of key -> pyqtgraph ROI (each having a ``.sl_mode`` attribute).
    image_item: the pyqtgraph ImageItem the ROIs are drawn over.
    """
    rois = []
    for key, roi in roi_dict.items():
        if not key.startswith("roi_"):
            continue

        if isinstance(roi, pg.EllipseROI):
            path = QPainterPath()
            path.addEllipse(roi.boundingRect())
        else:
            path = roi.shape()
        path = roi.mapToItem(image_item, path)

        for polygon in path.toSubpathPolygons():
            verts = np.array([[pt.y(), pt.x()] for pt in polygon], dtype=float)
            if verts.shape[0] >= 3:
                rois.append(RoiPolygon(verts, roi.sl_mode))
    return rois
```

- [ ] **Step 3: Smoke-test the import (offscreen)**

Run: `QT_QPA_PLATFORM=offscreen $PY -c "from pysimplemask.gui.model.roi_extract import extract_roi_geometry; from pysimplemask.gui.model.table_model import XmapConstraintsTableModel; print('ok')"`
Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "refactor(gui): move table_model + add roi_extract"
```

---

## Task 9: Move the controller and wire it to the headless model

This is the largest task. Move `simplemask_main.py` to `gui/control/main_window.py`, repoint imports,
replace the methods that used the kernel's Qt code with view-side rendering + `roi_extract`, route
draw masks and pixel-list parsing through `core`, and delete the dead correlation methods.

**Files:**
- Move: `simplemask_main.py` → `gui/control/main_window.py`
- Create: `src/pysimplemask/gui/app.py`
- Modify: `gui/control/main_window.py`

- [ ] **Step 1: Move the file**

```bash
cd /home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact/src/pysimplemask
git mv simplemask_main.py gui/control/main_window.py
```

- [ ] **Step 2: Repoint the imports**

In `gui/control/main_window.py`, replace the four intra-package imports:

```python
from . import __version__
from .simplemask_kernel import SimpleMask
from .simplemask_ui import Ui_SimpleMask as Ui
from .table_model import XmapConstraintsTableModel
```
with:

```python
import pyqtgraph as pg
from pysimplemask import __version__
from pysimplemask.core.model import SimpleMaskModel
from pysimplemask.core.io import text_to_array, load_pixel_list
from pysimplemask.gui.view.ui_mask import Ui_SimpleMask as Ui
from pysimplemask.gui.view.widgets import ImageViewROI  # noqa: F401 (promoted in .ui)
from pysimplemask.gui.model.table_model import XmapConstraintsTableModel
from pysimplemask.gui.model.roi_extract import extract_roi_geometry
```

Also delete the module-level `text_to_array` function in this file (now imported from `core.io`).

- [ ] **Step 3: Construct the headless model (no Qt handles)**

Replace:

```python
        self.sm = SimpleMask(self.mp1, self.infobar)
```
with:

```python
        self.sm = SimpleMaskModel()
        self.mp1.scene.sigMouseMoved.connect(self.show_location)
```

- [ ] **Step 4: Add the view-side `show_location` (was in the kernel)**

Add this method to `SimpleMaskGUI`:

```python
    def show_location(self, pos):
        if self.sm.shape is None:
            return
        if not self.mp1.scene.itemsBoundingRect().contains(pos):
            return
        mouse_point = self.mp1.getView().mapSceneToView(pos)
        idx = self.mp1.currentIndex
        col = int(mouse_point.x())
        row = int(mouse_point.y())
        msg = self.sm.dset.get_coordinates(col, row, idx)
        if msg:
            self.infobar.clear()
            self.infobar.setText(msg)
```

- [ ] **Step 5: Replace `plot()` to render directly (was `sm.show_saxs`)**

Replace the existing `plot` method with:

```python
    def plot(self):
        if not self.is_ready():
            return
        cmap = self.plot_cmap.currentText()
        plot_center = self.plot_center.isChecked()
        self.mp1.clear()
        self.mp1.setImage(self.sm.dset.data_display)
        self.mp1.adjust_viewbox()
        self.mp1.set_colormap(cmap)
        if plot_center:
            t = pg.ScatterPlotItem()
            center = self.sm.get_center(mode="vh")
            t.addPoints(x=[center[1]], y=[center[0]], symbol="+", size=15)
            self.mp1.add_item(t, label="center")
        self.mp1.setCurrentIndex(1)
        self.plot_index.setCurrentIndex(1)
```

- [ ] **Step 6: Add view-side `add_drawing` (was in the kernel)**

The ROI-creation code moves verbatim from the old `SimpleMask.add_drawing` into the controller,
operating on `self.mp1` and `self.sm.get_center`/`self.sm.shape`. Replace the existing
`add_drawing` controller method with the full implementation:

```python
    def add_drawing(self):
        if not self.is_ready():
            return
        if self.MaskWidget.currentIndex() != 1:
            return
        color = ("r", "g", "y", "b", "c", "m", "k", "w")[
            self.cb_selector_color.currentIndex()
        ]
        sl_type = self.cb_selector_type.currentText()
        sl_mode = self.cb_selector_mode.currentText()
        width = self.plot_width.value()
        num_edges = self.spinBox_num_edges.value()

        cen = self.sm.get_center(mode="xy")
        shape = self.sm.shape
        if cen[0] < 0 or cen[1] < 0 or cen[0] > shape[1] or cen[1] > shape[0]:
            logger.warning("beam center is out of range, use image center instead")
            cen = (shape[1] // 2, shape[0] // 2)

        if sl_mode == "inclusive":
            pen = pg.mkPen(color=color, width=width, style=QtCore.Qt.DotLine)
        else:
            pen = pg.mkPen(color=color, width=width)
        handle_pen = pg.mkPen(color=color, width=width)
        kwargs = {"pen": pen, "removable": True, "hoverPen": pen,
                  "handlePen": handle_pen, "movable": True}

        if sl_type == "Ellipse":
            size = (120, 160)
            new_roi = pg.EllipseROI(
                (cen[0] - size[0] // 2, cen[1] - size[1] // 2), size, **kwargs)
            new_roi.addScaleHandle([0.5, 0], [0.5, 0.5])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0.5])
            new_roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        elif sl_type == "Circle":
            radius = 60
            new_roi = pg.CircleROI(
                pos=[cen[0] - radius, cen[1] - radius], radius=radius, **kwargs)
            new_roi.addScaleHandle([0.5, 0], [0.5, 0.5])
            new_roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        elif sl_type == "Polygon":
            offset = np.random.randint(0, 360)
            theta = np.linspace(0, np.pi * 2, num_edges + 1) + offset
            x = 60 * np.cos(theta) + cen[0]
            y = 60 * np.sin(theta) + cen[1]
            new_roi = pg.PolyLineROI(np.vstack([x, y]).T, closed=True, **kwargs)
        elif sl_type == "Rectangle":
            new_roi = pg.RectROI(cen, [200, 150], **kwargs)
            for h in ([0, 0], [0, 0.5], [0, 1], [0.5, 0], [0.5, 1], [1, 0], [1, 0.5], [1, 1]):
                new_roi.addScaleHandle(h, [1 - h[0], 1 - h[1]])
        else:
            logger.error("unsupported ROI type %s", sl_type)
            return
        new_roi.sl_mode = sl_mode
        roi_key = self.mp1.add_item(new_roi)
        new_roi.sigRemoveRequested.connect(lambda: self.mp1.remove_item(roi_key))
```

Add `from pyqtgraph.Qt import QtCore` to the imports if not already present (it is used above).

- [ ] **Step 7: Route the `mask_draw` branch through core**

In `mask_evaluate`, replace the `mask_draw` branch:

```python
        elif target == "mask_draw":
            kwargs = {"arr": np.logical_not(self.sm.evaluate_drawing())}
```
with:

```python
        elif target == "mask_draw":
            rois = extract_roi_geometry(self.mp1.roi, self.mp1.imageItem)
            self.sm.set_draw_rois(rois)
            self.mp1.remove_rois(filter_str="roi_")
            keep = self.sm.evaluate_draw_mask()
            kwargs = {"arr": np.logical_not(keep)}
```

- [ ] **Step 8: Route pixel-list file loading through core**

In `mask_list_load`, replace the json/txt/csv parsing block (the `if fname.endswith(".json") ...`
through the `xy = np.loadtxt(...)` error handling) with:

```python
        try:
            xy = load_pixel_list(fname)
        except Exception:
            self.statusbar.showMessage("only support json, csv and space separated file", 500)
            return
```

Keep the existing row/col-roll and 1-based toggles that follow.

- [ ] **Step 9: Use `least_multiple` from core in `compute_partition`**

In `compute_partition`, delete the nested `def least_multiple(...)` definition and add to the imports
at the top of the file:

```python
from pysimplemask.core.partition import least_multiple
```

- [ ] **Step 10: Update `read_data` call to `load`**

In `load` (the controller method), replace `self.sm.read_data(fname, **kwargs)` with
`self.sm.load(fname, **kwargs)`.

- [ ] **Step 11: Delete the dead correlation methods**

Delete the controller methods `corr_add_roi`, `update_corr_angle`, and `perform_correlation`
(they call non-existent model methods), and remove any `self.<btn>.clicked.connect(self.corr_*)`
/ `self.angle_*` wiring lines in `__init__` that reference them.

- [ ] **Step 12: Move `main_gui` into `gui/app.py`**

Cut the `main_gui` function (and the `if __name__ == "__main__":` block) from the bottom of
`main_window.py` into a new file `src/pysimplemask/gui/app.py`:

```python
"""GUI application entry point."""

import sys

from PySide6.QtWidgets import QApplication

from pysimplemask.gui.control.main_window import SimpleMaskGUI


def main_gui(path=None):
    app = QApplication(sys.argv)
    window = SimpleMaskGUI(path)  # noqa: F841 (kept alive by the event loop)
    app.exec()


if __name__ == "__main__":
    main_gui()
```

- [ ] **Step 13: Offscreen import-smoke of the controller**

Run:
```bash
QT_QPA_PLATFORM=offscreen $PY -c "from pysimplemask.gui.control.main_window import SimpleMaskGUI; from pysimplemask.gui.app import main_gui; print('ok')"
```
Expected: `ok`.

- [ ] **Step 14: Commit**

```bash
git add -A
git commit -m "refactor(gui): move controller to gui/control, wire to headless core"
```

---

## Task 10: Update package init, CLI, and packaging

**Files:**
- Modify: `src/pysimplemask/__init__.py`, `src/pysimplemask/cli.py`, `pyproject.toml`, `Makefile` (new), `CLAUDE.md`

- [ ] **Step 1: Make `__init__.py` Qt-free**

Replace `src/pysimplemask/__init__.py` with:

```python
"""Top-level package for pySimpleMask."""

from importlib.metadata import PackageNotFoundError, version

__author__ = """Miaoqi Chu"""
__email__ = "mqichu@anl.gov"

try:
    __version__ = version("pysimplemask")
except PackageNotFoundError:
    __version__ = "0.1.0"

from .core.model import SimpleMaskModel  # noqa: E402 (after __version__ is defined)

__all__ = ["SimpleMaskModel", "__version__"]
```

- [ ] **Step 2: Repoint the CLI**

In `src/pysimplemask/cli.py`, change the imports:

```python
from pysimplemask import main_gui, __version__
from pysimplemask.utils import combine_qmap_files
```
to:

```python
from pysimplemask import __version__
from pysimplemask.gui.app import main_gui
from pysimplemask.core.partition import combine_qmap_files
```

- [ ] **Step 3: Update packaging data globs**

In `pyproject.toml`, under `[tool.setuptools.package-data]`, replace the resources line and add the
UI file:

```toml
[tool.setuptools.package-data]
"*" = ["*.*"]
"pysimplemask.resources" = ["*.svg"]
"pysimplemask.gui.view" = ["*.ui"]
```

- [ ] **Step 4: Add a `make ui` target**

Create `Makefile` at repo root:

```makefile
PY ?= /local/MQICHU/envs/l2606_simplemask_refact/bin/python
UIC ?= /local/MQICHU/envs/l2606_simplemask_refact/bin/pyside6-uic

.PHONY: ui test lint
ui:
	$(UIC) src/pysimplemask/gui/view/mask.ui -o src/pysimplemask/gui/view/ui_mask.py

test:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check src tests
```

- [ ] **Step 5: Reinstall (entry points unchanged, package layout changed)**

```bash
cd /home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact
/local/MQICHU/envs/l2606_simplemask_refact/bin/pip install -e . -q
```

- [ ] **Step 6: Verify CLI entry resolves**

Run: `$PY -c "from pysimplemask.cli import main, combine_qmaps; print('ok')"`
Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor: Qt-free package init, repoint CLI, package the .ui + make ui target"
```

---

## Task 11: Final verification + docs

**Files:**
- Modify: `CLAUDE.md` (architecture + commands)

- [ ] **Step 1: Full test suite**

Run: `$PY -m pytest tests -q`
Expected: all green (reader 52 + io 5 + partition 5 + rasterize 4 + model 2 = 68+).

- [ ] **Step 2: Lint**

Run: `$PY -m ruff check src tests`
Expected: `All checks passed!` (fix any unused imports flagged, e.g. drop unused names).

- [ ] **Step 3: GUI launches offscreen without crashing on construction**

Run:
```bash
QT_QPA_PLATFORM=offscreen timeout 15 $PY -c "
from PySide6.QtWidgets import QApplication
from pysimplemask.gui.control.main_window import SimpleMaskGUI
app = QApplication([])
w = SimpleMaskGUI()
print('constructed ok')
"
```
Expected: `constructed ok`.

- [ ] **Step 4: Headless guarantee re-check**

Run: `$PY -c "import pysimplemask, sys; print('PySide6' in sys.modules)"`
Expected: `False`.

- [ ] **Step 5: Update CLAUDE.md**

Update the architecture section to describe `core/` (Qt-free, `SimpleMaskModel`) vs `gui/`
(view/model/control), the headless scripting entry, and the `.ui` workflow (`make ui`). Update the
commands block to use `pytest tests`, and note `import pysimplemask` is Qt-free.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "docs: update CLAUDE.md for MVC core/gui architecture"
```

---

## Self-review notes (coverage map)

- Spec §"core model API" → Tasks 6, 10. §"rasterization" → Tasks 5, 8 (extract), 9 (wire).
- §"heavy-lifting extracted" → Task 3 (least_multiple), Task 4 (io), Task 9 (wiring).
- §"package init / entry points" → Task 10. §".ui workflow" → Tasks 7, 10.
- §"testing" → Tasks 6, 5, 4, 3, 11. §"file move map" → Tasks 2, 6, 7, 8, 9.
- §"risks: hidden Qt in core" → Task 6 Step 11 + Task 11 Step 4 (runtime assertion).
