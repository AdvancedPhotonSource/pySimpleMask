# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

pySimpleMask creates **masks** and **Q-partition maps** for X-ray scattering patterns
(SAXS / WAXS / XPCS data reduction), primarily for APS beamlines 8-ID-I and 9-ID-D. Input
is raw detector data in many formats (HDF5, IMM, Rigaku 500k/3M binary, TIFF, Timepix);
output is a TIFF mask and a Nexus/XPCS-compatible HDF5 partition file.

It is split into a **Qt-free `core/`** (the masking/partition engine, usable headless from
Python scripts) and a **PySide6 + pyqtgraph `gui/`** (view/model/control). `import
pysimplemask` pulls in **no Qt** — the GUI is imported lazily only when launched.

## Environment & commands

Use the project conda environment for all test/development work:

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/<tool>     # python, pytest, ruff, mypy, pysimplemask
# or: conda activate l2606_simplemask_refact
```

This env has the dev tools (pytest, ruff, mypy) and an editable install of the package
bound to *this* checkout. If `import pysimplemask` ever resolves elsewhere (the editable
`.pth` once pointed at a sibling `pySimpleMask-refactor` path), reinstall from the repo root:

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pip install -e ".[dev]"
```

Common commands (prefix each with the env's `bin/` or activate the env first):

```bash
pysimplemask                       # launch the GUI (needs an X11/Wayland display)
pysimplemask --path /data/dir      # launch with a starting directory
pysimplemask-combine-qmaps a.h5 b.h5 out.h5   # merge two qmap HDF5 files

make test                          # pytest tests   (or: pytest tests)
make lint                          # ruff check src tests
make ui                            # regenerate gui/view/ui_mask.py from gui/view/mask.ui
mypy                               # strict type check (config in pyproject.toml; see note)
```

The core runs **headless** — drive the whole pipeline from a script without a display:

```python
from pysimplemask.core import SimpleMaskModel
m = SimpleMaskModel()
m.read_data("scan.h5", beamline="APS_8IDI", begin_idx=0, num_frames=-1)
m.mask_evaluate("mask_threshold", low=0, high=1e7, low_enable=True, high_enable=True)
m.mask_apply("mask_threshold")
m.add_polygon([(r0, c0), (r1, c1), (r2, c2)], mode="exclusive"); m.evaluate_draw(); m.mask_apply("mask_draw")
m.compute_partition(mode="q-phi", dq_num=10, sq_num=100, dp_num=36, sp_num=360)
m.save_partition("out.hdf"); m.save_mask("mask.tif")
```

**Tests** live in `tests/core/` (synthetic fixtures, no external data): `reader/` (52),
plus io, rasterize, partition helpers, and a headless model test that asserts
`import pysimplemask` stays Qt-free. There are no GUI unit tests; verify GUI changes by
launching, or with an offscreen construct smoke (`QT_QPA_PLATFORM=offscreen`).

**mypy `strict` is aspirational, not passing.** The config in `pyproject.toml` sets
`strict = true`, but the codebase is largely untyped (even untouched modules like
`qmap.py` fail it), so a clean `mypy` run is not a realistic gate today. `ruff` is the
enforced linter.

Releases publish to PyPI automatically on pushing a git tag (`.github/workflows/publish.yml`).
Version is derived from git tags via `setuptools_scm` — there is no hardcoded version string.

## Architecture

MVC: **`core/` (Qt-free) ← `gui/` (Qt)**. The GUI talks to one model object
(`SimpleMaskModel`) and renders the numpy it produces; it never reaches into readers or
geometry directly. `core/` imports no Qt and is fully scriptable.

**`core/` — Qt-free engine** (`src/pysimplemask/core/`)
- `model.py` — `SimpleMaskModel`: the central façade/model. Holds the active dataset
  (`dset`), `qmap`, `mask`, `mask_kernel`, and draw-ROI list. Orchestrates
  read_data → mask → qmap → partition → save. No widget handles; produces plain numpy.
- `file_handler.get_handler(beamline, fname)` / `reader/` — see Readers below.
- `qmap.py` — pure, `lru_cache`d geometry. `compute_qmap(stype, metadata)` →
  `compute_transmission_qmap` / `compute_reflection_qmap`, per-pixel `q/phi/x/y` maps via
  3D rotation matrices (swing angles; reflection incident angle + orientation).
- `mask.py` — `MaskAssemble` composes independent mask `workers` keyed by type
  (`mask_blemish`, `mask_file`, `mask_threshold`, `mask_list`, `mask_draw`, `mask_outlier`,
  `mask_parameter`). Each `.evaluate()` proposes a candidate; `.apply(target)` ANDs it into
  the running mask. Undo/redo/reset = `mask_record` stack + `mask_ptr`. A default blemish
  is auto-loaded by detector shape from `~/Documents/areaDetectorBlemish`.
- `rasterize.py` — turns ROI geometry (`RoiPolygon` + `circle/ellipse/rectangle/line`
  vertex helpers) into a boolean keep-mask via `skimage` (no Qt). Used by both the GUI
  (geometry extracted from pyqtgraph ROIs) and scripts (`SimpleMaskModel.add_*`).
- `partition.py` — `generate_partition`, `combine_partitions`, `check_consistency`,
  `optimize_integer_array`, `hash_numpy_dict`, `least_multiple`, `combine_qmap_files`.
- `io.py` — `text_to_array`, `load_pixel_list` (json/txt/csv pixel parsing).
- `find_center.py`, `outlier_removal.py`, `ellipse_util.py` — beam-center finding,
  SAXS-1D outlier masking, ellipse-fit q-correction.

**Readers** (`src/pysimplemask/core/reader/`)
- `get_reader(beamline, fname)` (re-exported via `core/file_handler.get_handler`) →
  `beamlines/aps_8idi.APS8IDIReader` or `beamlines/aps_9idd.APS9IDDReader`.
- Readers subclass `reader/base_reader.FileReader` (owns `metadata`, `scat`, the
  `data_display` channel stack, mask state, centers, `compute_qmap()`) and pick a format
  loader from `reader/formats/` (`get_format_loader` by extension: hdf/imm/rigaku). `stype`
  selects the qmap geometry. All loaders return the per-pixel **mean** over the frame range.

**`gui/` — Qt only** (`src/pysimplemask/gui/`)
- `view/mask.ui` (Designer source) → `view/ui_mask.py` (**generated** by `pyside6-uic`,
  never hand-edited; run `make ui` after editing the `.ui`). `view/widgets.py` is the
  custom pyqtgraph `ImageViewROI` (sets `imageAxisOrder="row-major"`).
- `model/table_model.py` (Qt table model for param constraints) + `model/roi_extract.py`
  (pyqtgraph ROI items → `core.rasterize` polygons).
- `control/main_window.py` — `SimpleMaskGUI(QMainWindow, Ui)`: wires widgets to
  `SimpleMaskModel`, renders `dset.data_display`, handles mouse/ROI input. GUI settings
  persist to `~/.pysimplemask/config.json`. `gui/app.py::main_gui()` is the entry point.

To add a beamline: add a branch in `core/reader/__init__.get_reader` plus a `FileReader`
subclass under `core/reader/beamlines/`.

## Partition model (important domain concept)

`SimpleMaskModel.compute_partition(mode, ...)` builds **two** partition resolutions for the same
axes and saves both: a coarse **dynamic** map (`dq_num` × `dp_num`) and a fine **static**
map (`sq_num` × `sp_num`). `mode` is `"<axis0>-<axis1>"`, e.g. `q-phi`, `x-y`, or
`eq-ephi` (ellipse-corrected q/phi — temporarily swaps `qmap["q"]`/`["phi"]` for the
ellipse gradient, then restores). Results are written to HDF5 under group `/qmap` with a
content `hash` and package `version` attribute; large integer arrays are lzf-compressed.
When changing partition output, keep the dynamic/static pair and the
`check_consistency(dqmap, sqmap, mask)` invariant intact.

## Conventions

- Keep `core/` Qt-free: never import PySide6/pyqtgraph there. The headless test
  (`tests/core/test_model_headless.py`) fails if `import pysimplemask` pulls in Qt.
- pyqtgraph is configured `imageAxisOrder="row-major"` (in `gui/view/widgets.py`); arrays
  are indexed (row=v=y, col=h=x). Centers come in two conventions — `vh` (row, col) and
  `xy` — converted via `get_center(mode=...)`. Mismatching these is a common bug source.
- Metadata flows reader → `qmap`: editing parameters in the GUI ParameterTree calls
  `SimpleMaskModel.update_parameters()`, which recomputes the qmap and refreshes the mask
  kernel's qmap. Don't recompute geometry inline elsewhere.
- `view/ui_mask.py` is generated and excluded from ruff. Edit `view/mask.ui` in Qt
  Designer, run `make ui`, then wire new widgets in `control/main_window.py`.
