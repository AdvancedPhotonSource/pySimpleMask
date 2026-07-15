# MVC refactor — design

**Date:** 2026-06-15
**Branch:** refact
**Scope:** Restructure the whole `src/pysimplemask/` package into an MVC-style layout with a
Qt-free, scriptable `core/` and a Qt-only `gui/`. Builds on the completed reader refactor.

## Goal

Reorganize the project into a maintainable MVC architecture:
1. Extract core (domain/compute) logic into a `core/` module.
2. Place all GUI code in a `gui/` module.
3. Move the GUI's heavy-lifting into `core/`; the GUI only displays data and handles user input.
4. Make the core's main functionality callable directly from Python scripts (headless, Qt-free).
5. Keep the `.ui` Designer source and its generated `.py` so new widgets can be added in
   Qt Designer, recompiled, and wired up.

## Decisions (settled with the user)

1. **`gui/model` = thin Qt adapters only** (Qt item-models + ROI-geometry extraction +
   small presentation state). The real domain model lives in `core/` and the GUI calls it directly.
2. **All ROI rasterization moves to `core/`** using `skimage` (Qt-free). The view only extracts
   ROI geometry; scripts build the same geometry directly.
3. **Big-bang reorg** — move everything to the new tree in one pass, then fix imports.
4. **Hard move, no back-compat shims.** Internal imports are updated; old paths are removed.
5. Approved names: `core/model.py::SimpleMaskModel`, `core/mask.py` (was `area_mask.py`),
   `core/partition.py` (was `utils.py`), `gui/view/widgets.py` (was `pyqtgraph_mod.py`),
   `gui/view/ui_mask.py` (was `simplemask_ui.py`, generated from `gui/view/mask.ui`).

## Non-goals

- No change to the scattering/qmap/partition math or to mask semantics (aside from the
  accepted rasterization edge difference below).
- No new features. Dead/broken code encountered during the move is deleted, not fixed-and-kept.
- No external back-compat layer (the package is a GUI app, not a widely-imported library).

## Current coupling (starting point)

- **Already Qt-free → `core/`:** `reader/`, `qmap.py`, `area_mask.py`, `find_center.py`,
  `outlier_removal.py`, `ellipse_util.py`, `utils.py`, `file_handler.py`.
- **View (Qt) → `gui/view/`:** `ui/mask.ui` + generated `simplemask_ui.py` (`Ui_SimpleMask`),
  `pyqtgraph_mod.py`, `table_model.py`.
- **`simplemask_kernel.py::SimpleMask` is mixed:** core logic (`read_data`, `compute_partition`,
  `compute_saxs1d`, `save_*`, `find_center`, `goto_max`, `update_parameters`, `get_center`)
  tangled with Qt (`__init__(pg_hdl, infobar)`, `show_saxs`, `show_location`, `add_drawing`,
  `remove_roi`, `evaluate_drawing` Qt rasterization, and a `self.hdl.setCurrentIndex(3)` buried
  in `compute_partition_general`). It cannot run headless today.
- **`simplemask_main.py::SimpleMaskGUI` holds heavy-lifting:** `least_multiple` partition
  rounding, `mask_list_load` json/txt/csv parsing, `text_to_array`, per-tab→kwargs mapping.
- **Pre-existing broken code (delete):** `corr_add_roi`/`perform_correlation` call
  `self.sm.set_corr_roi`/`perform_correlation`, which do not exist on the kernel.

## Target layout

```
src/pysimplemask/
  __init__.py            # Qt-free; exposes __version__ and core.SimpleMaskModel
  cli.py                 # main -> gui.app.main_gui; combine_qmaps -> core.partition
  core/                  # Qt-FREE, scriptable
    __init__.py          # SimpleMaskModel, get_handler, geometry helpers
    model.py             # SimpleMaskModel (was SimpleMask)
    file_handler.py
    reader/              # moved unchanged from pysimplemask/reader
    qmap.py
    mask.py              # was area_mask.py
    rasterize.py         # NEW: geometry dataclasses + skimage rasterization
    partition.py         # was utils.py
    io.py                # NEW: text_to_array + load_pixel_list (json/txt/csv parsing)
    find_center.py
    outlier_removal.py
    ellipse_util.py
  gui/                   # Qt only
    __init__.py
    app.py               # main_gui(path)
    view/
      mask.ui            # moved from ui/mask.ui
      ui_mask.py         # generated from mask.ui (was simplemask_ui.py)
      widgets.py         # was pyqtgraph_mod.py
    model/
      table_model.py     # was table_model.py
      roi_extract.py     # NEW: pyqtgraph ROI items -> core geometry
    control/
      main_window.py     # SimpleMaskGUI (was simplemask_main.py)
  resources/logo.svg
```

### File move map

| Old | New |
|-----|-----|
| `reader/` | `core/reader/` |
| `file_handler.py` | `core/file_handler.py` |
| `qmap.py` | `core/qmap.py` |
| `area_mask.py` | `core/mask.py` |
| `utils.py` | `core/partition.py` |
| `find_center.py`, `outlier_removal.py`, `ellipse_util.py` | `core/` |
| `simplemask_kernel.py` (`SimpleMask`) | `core/model.py` (`SimpleMaskModel`) |
| `text_to_array` + `mask_list_load` parsing (from `simplemask_main.py`) | `core/io.py` (new) |
| — | `core/rasterize.py` (new) |
| `ui/mask.ui` | `gui/view/mask.ui` |
| `simplemask_ui.py` | `gui/view/ui_mask.py` |
| `pyqtgraph_mod.py` | `gui/view/widgets.py` |
| `table_model.py` | `gui/model/table_model.py` |
| — | `gui/model/roi_extract.py` (new) |
| `simplemask_main.py` (`SimpleMaskGUI` + `main_gui`) | `gui/control/main_window.py` + `gui/app.py` |

## Core model API (`core/model.py::SimpleMaskModel`)

Qt-free. Constructor takes no widget handles. Public methods (script-callable):
`load(fname, beamline, **kwargs)` (was `read_data`), `evaluate_mask(target, **kwargs)`,
`apply_mask(target)`, `mask_action(action)`, `compute_saxs1d(...)`, `compute_partition(mode, **kwargs)`,
`save_mask(path)`, `save_partition(path)`, `find_center()`, `goto_max()`,
`update_parameters(new_metadata=None)`, `get_center(mode)`, `get_pts_with_similar_intensity(...)`,
plus ROI geometry helpers `add_polygon/add_circle/add_ellipse/add_rectangle/add_line(..., mode=...)`,
`clear_rois()`.

Removed from the model (now view responsibilities): `show_saxs`, `show_location`, `add_drawing`
(pyqtgraph item creation), `remove_roi`, the Qt `evaluate_drawing`, and the `setCurrentIndex(3)`
side-effect.

State exposed for the view to render (all plain numpy / Python): `dset.data_display` channel
stack, `qmap`/`qmap_unit`, `mask`, `new_partition`, metadata parameter structure, saxs1d arrays.

`import pysimplemask` and `import pysimplemask.core` must not import Qt. Headless example:

```python
from pysimplemask.core import SimpleMaskModel
m = SimpleMaskModel()
m.load("scan.h5", beamline="APS_8IDI", begin_idx=0, num_frames=-1)
m.evaluate_mask("mask_threshold", low=0, high=1e7, low_enable=True, high_enable=True)
m.apply_mask("mask_threshold")
m.add_polygon([(r0, c0), (r1, c1), (r2, c2)], mode="exclusive")
m.apply_mask("mask_draw")
m.compute_partition(mode="q-phi", dq_num=10, sq_num=100, dp_num=36, sp_num=360)
m.save_partition("out.hdf")
m.save_mask("mask.tif")
```

## Rasterization (`core/rasterize.py`)

Geometry dataclasses: `Polygon(vertices, mode)`, `Circle(center, radius, mode)`,
`Ellipse(center, axes, angle, mode)`, `Rectangle(origin, size, angle, mode)`,
`Line(p0, p1, width, mode)`; `mode in {"inclusive", "exclusive"}`. Coordinates are image-space
`(row, col)`.

`rasterize(shape, rois) -> np.ndarray[bool]` rasterizes each shape with `skimage.draw`
(curves sampled to polygons; rotation applied when computing vertices), then combines:
`final = (NOT any exclusive) AND (any inclusive, or all-True if no inclusive shapes)` — the same
logic as the current Qt code.

- **GUI:** `gui/model/roi_extract.py` converts each pyqtgraph ROI item (applying its transform via
  `mapToItem(imageItem)`) into the matching geometry dataclass.
- **Script:** `SimpleMaskModel.add_*` helpers build the dataclasses directly.
- **Accepted consequence:** `skimage` fill differs from the old `QPainter` fill by up to ~1px at
  shape boundaries. Pinned by a rasterization unit test; documented as intended.

## Heavy-lifting extracted from the controller into core

- `least_multiple` partition-size rounding (sq a multiple of dq, sp of dp) → `core/partition.py`
  helper, applied inside `SimpleMaskModel.compute_partition` so both GUI and scripts get it.
- `mask_list_load` file parsing (json/txt/csv → pixel array) → `core/io.py::load_pixel_list(path)`;
  the controller only opens the dialog and passes the path.
- `text_to_array` → `core/io.py`.

The controller retains: widget↔kwargs marshalling, file dialogs, status-bar messages, plot-index
switching, and config save/load.

## Package init / entry points

- `pysimplemask/__init__.py`: no Qt import. Exposes `__version__` and re-exports
  `core.SimpleMaskModel`. (Removes the current `from .simplemask_main import main_gui`.)
- `cli.py`: `main()` does `from pysimplemask.gui.app import main_gui` (lazy Qt import);
  `combine_qmaps()` imports from `core.partition`.
- `pyproject.toml` `[project.scripts]` entry points unchanged (`pysimplemask`,
  `pysimplemask-combine-qmaps`); only their import targets move.
- Package data: ensure `gui/view/*.ui` and `resources/*.svg` are still included
  (`[tool.setuptools.package-data]`).

## `.ui` workflow

`gui/view/mask.ui` is the Designer source; `gui/view/ui_mask.py` is generated and never
hand-edited. Regeneration command (documented in CLAUDE.md and a `make ui` target):
```
pyside6-uic src/pysimplemask/gui/view/mask.ui -o src/pysimplemask/gui/view/ui_mask.py
```
The controller imports `Ui_SimpleMask` from `gui/view/ui_mask.py`. Any resource/icon paths the UI
references are kept working after the move.

## Testing

- Move `tests/reader/` → `tests/core/reader/`; update imports to `pysimplemask.core.reader...`
  (conftest builders included). Reader suite must stay green.
- `tests/core/test_rasterize.py`: polygon/circle/ellipse/rectangle/line (incl. rotation and
  inclusive/exclusive combination) → expected boolean masks.
- `tests/core/test_model_headless.py`: construct `SimpleMaskModel()` with no Qt, load a synthetic
  HDF (reusing reader fixtures), apply a threshold mask, add a polygon, compute a partition, and
  save mask + partition to temp files — asserting outputs and that **no Qt import is required**.
- GUI import-smoke: import `gui.control.main_window` and construct the window under
  `QT_QPA_PLATFORM=offscreen`.
- `ruff check src/ tests/` clean.

## Risks & mitigations

- **Hidden Qt in "core":** after the move, grep `core/` for `PySide6`/`pyqtgraph`/`from .*Qt`
  imports; any hit is a bug to fix. The headless test enforces this at runtime.
- **Import churn (big-bang):** one grep-driven sweep of every `from .`/`import pysimplemask`;
  reader tests + headless test + GUI smoke catch breakage.
- **Rasterization edge differences:** accepted; pinned by `test_rasterize.py`.
- **`__init__` importing Qt** would break headless use; explicitly removed and asserted by the
  headless test (run in a subprocess/process that imports `pysimplemask` and checks
  `sys.modules` has no `PySide6`).
- **Package data omission:** verify `.ui`/`.svg` ship via an install check.
```
