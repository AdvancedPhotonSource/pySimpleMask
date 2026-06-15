# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

pySimpleMask is a PySide6 (Qt6) + pyqtgraph desktop GUI for creating **masks** and
**Q-partition maps** for X-ray scattering patterns (SAXS / WAXS / XPCS data reduction),
primarily for APS beamlines 8-ID-I and 9-ID-D. Input is raw detector data in many
formats (HDF5, IMM, Rigaku 500k/3M binary, TIFF, Timepix); output is a TIFF mask and a
Nexus/XPCS-compatible HDF5 partition file.

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

pytest tests/reader/              # run the reader test suite (52 synthetic-fixture tests)
ruff check src/ tests/            # lint
mypy                              # strict type check (config in pyproject.toml; see note)
```

**Test coverage is uneven.** `tests/reader/` has a real synthetic-fixture suite (no
external data needed; builders live in `tests/reader/conftest.py`) covering the format
loaders, beamline metadata, and dispatch. The legacy `tests/test_pysimplemask.py` is a
broken stub (imports a nonexistent `pysimplemask.pysimplemask`) — ignore it. Outside the
reader subsystem there are no tests; verify those changes by running the GUI.

**mypy `strict` is aspirational, not passing.** The config in `pyproject.toml` sets
`strict = true`, but the codebase is largely untyped (even untouched modules like
`qmap.py` fail it), so a clean `mypy` run is not a realistic gate today. `ruff` is the
enforced linter.

Releases publish to PyPI automatically on pushing a git tag (`.github/workflows/publish.yml`).
Version is derived from git tags via `setuptools_scm` — there is no hardcoded version string.

## Architecture

The code is layered View → Kernel → Data/Compute. Keep this separation: the GUI talks to
one façade object (`SimpleMask`), and never to readers or geometry functions directly.

**View layer** (`src/pysimplemask/`)
- `simplemask_ui.py` — large hand-maintained Qt layout class `Ui_SimpleMask` (treat as
  generated; edit deliberately).
- `simplemask_main.py` — `SimpleMaskGUI(QMainWindow, Ui)`: the controller. Wires every
  button/widget to actions and owns the single `SimpleMask` kernel instance
  (`self.sm`). Entry point `main_gui()` lives here. GUI settings persist to
  `~/.pysimplemask/config.json`.
- `pyqtgraph_mod.py` (custom ROI / image widgets), `table_model.py` (Qt table model for
  parameter constraints).

**Kernel layer**
- `simplemask_kernel.py` — `SimpleMask`: the central façade. Holds the active dataset
  (`dset`), `qmap`, `mask`, and `mask_kernel`. Orchestrates load → mask → qmap →
  partition → save and bridges the Qt view to the compute/data code.

**Data layer — readers** (`src/pysimplemask/reader/`)
- `file_handler.get_handler(beamline, fname)` dispatches by beamline string to
  `APS_8IDI/aps_8idi_reader.APS8IDIReader` or `APS_9IDD/aps_9idd_reader.APS9IDDReader`.
  To add a beamline, add a branch here plus a reader subclass.
- All readers subclass `reader/base_reader.FileReader`, which owns `metadata`, the
  scattering image (`scat`), the `data_display` channel stack, mask state, center
  management, and `compute_qmap()`. Each concrete reader selects a low-level *handler* by
  file extension (HDF/IMM/Rigaku/Timepix under `reader/APS_8IDI/`).
- `stype` (`"Transmission"` or `"Reflection"`) on the reader selects the qmap geometry.
- `data_display` is a stacked array of named channels (scattering, mask, scattering×mask,
  dqmap/sqmap partitions, plus per-pixel qmap channels). The GUI's `plot_index` combo
  switches which channel the pyqtgraph ImageView shows.

**Compute layer**
- `qmap.py` — pure, `lru_cache`d geometry. `compute_qmap(stype, metadata)` →
  `compute_transmission_qmap` / `compute_reflection_qmap`, producing per-pixel `q`, `phi`,
  `x`, `y` maps via 3D rotation matrices (handles horizontal/vertical swing angles and, for
  reflection, incident angle + orientation). `E2KCONST` converts energy↔wavelength.
- `area_mask.py` — `MaskAssemble` composes independent mask `workers` keyed by type
  (`mask_blemish`, `mask_file`, `mask_threshold`, `mask_list`, `mask_draw`, `mask_outlier`,
  `mask_parameter`). Each worker `.evaluate()` proposes a candidate mask; `.apply(target)`
  logically ANDs it into the running combined mask. Undo/redo/reset are a `mask_record`
  stack + `mask_ptr`. A default blemish (bad-pixel) map is auto-loaded by detector shape
  from `~/Documents/areaDetectorBlemish`.
- `utils.py` — partition generation and persistence: `generate_partition`,
  `combine_partitions`, `check_consistency`, integer-array compaction
  (`optimize_integer_array`), `hash_numpy_dict`, and `combine_qmap_files` (the
  combine-qmaps CLI).
- `find_center.py`, `outlier_removal.py`, `ellipse_util.py` — beam-center finding,
  SAXS-1D outlier masking, and ellipse-fit q-correction.

## Partition model (important domain concept)

`SimpleMask.compute_partition(mode, ...)` builds **two** partition resolutions for the same
axes and saves both: a coarse **dynamic** map (`dq_num` × `dp_num`) and a fine **static**
map (`sq_num` × `sp_num`). `mode` is `"<axis0>-<axis1>"`, e.g. `q-phi`, `x-y`, or
`eq-ephi` (ellipse-corrected q/phi — temporarily swaps `qmap["q"]`/`["phi"]` for the
ellipse gradient, then restores). Results are written to HDF5 under group `/qmap` with a
content `hash` and package `version` attribute; large integer arrays are lzf-compressed.
When changing partition output, keep the dynamic/static pair and the
`check_consistency(dqmap, sqmap, mask)` invariant intact.

## Conventions

- pyqtgraph is configured `imageAxisOrder="row-major"`; arrays are indexed (row=v=y,
  col=h=x). Centers come in two conventions — `vh` (row, col) and `xy` — converted via
  `get_center(mode=...)`. Mismatching these is a common bug source.
- Metadata flows reader → `qmap`: editing parameters in the GUI ParameterTree calls
  `SimpleMask.update_parameters()`, which recomputes the qmap and refreshes the mask
  kernel's qmap. Don't recompute geometry inline elsewhere.
