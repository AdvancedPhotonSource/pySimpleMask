# pySimpleMask

[![PyPI version](https://img.shields.io/pypi/v/pysimplemask.svg)](https://pypi.python.org/pypi/pysimplemask)

**pySimpleMask** is a tool for creating masks and Q-partition maps for X-ray scattering
patterns, supporting SAXS, WAXS, and XPCS data reduction. It ships both a desktop GUI
and a **headless Python API** that can drive the full pipeline from scripts.

## Features

- **Versatile data support** — HDF5 (NeXus/XPCS), IMM, Rigaku 500k/3M binary, TIFF,
  native TIFF images (NativeFiles beamline with editable placeholder metadata).
  Supported beamlines: APS 8-ID-I (transmission) and APS 9-ID-D (reflection/GISAXS).
  XPCS result HDF5 files (containing an `/xpcs` group) are auto-detected and their
  saved partition is restored on load.
  On free-threaded Python (CPython 3.13+ `--disable-gil`), LZ4-chunked HDF5 files are
  read with a thread-parallel HDF5-bypass path (no GIL, no HDF5 mutex) for faster
  frame averaging; standard Python falls back automatically.
- **Interactive masking**
  - Drawing tools: polygons, circles, ellipses, rectangles, line-width ROIs.
  - Binary threshold (low / high intensity limits with dtype presets).
  - Blemish/bad-pixel maps (TIFF or HDF5).
  - Additional mask file import (TIFF or HDF5).
  - Manual pixel selection by click or coordinate list.
  - **Outlier removal** — two strategies:
    - *CircularRings*: SAXS 1-D azimuthal average comparison per q-ring.
    - *AdjacentPixels*: fixed-size spatial boxes, sorted brightest-first.
    - Both support percentile-clip and MAD metrics.
  - Parametric masking by q-map range (q, phi, x, y, …); each constraint row gets a
    1-based group-index, so a set of non-overlapping ranges can double as an explicit
    dynamic-q grouping for partitioning (see below).
  - Undo / redo / reset mask history.
- **Beam-center finding** — iterative centro-symmetry cross-correlation, converges
  in 1–2 passes; bounded crop for speed on large detectors.
- **Partition generation**
  - Q-Phi (dynamic + static resolution pair).
  - X-Y spatial partitions.
  - Ellipse-corrected Q-Phi (eq-ephi).
  - Custom axis pair from any q-map channel.
  - **Group-index sub-partitioning** — treat the parametrization/draw tab's
    constraint groups as independent pixel groups and compute independent
    sub-partitions per group on both axes (for ellipse mode, each group also gets
    its own ellipse fit, so rho and phi stay geometrically consistent within a
    group), then combine them into one overall dynamic/static partition. Works on
    any partition mode (q-phi, x-y, ellipse, general). Requires non-overlapping
    constraint ranges (validated — overlapping or empty groups raise a descriptive
    error rather than silently mis-binning). GUI: the "Use group-index for
    sub-partitions" checkbox in the Partition groupbox, showing the current active
    group count. Script/CLI: `use_groupindex_for_subpartition=True` /
    `--use-groupindex-for-subpartition`.
- **Visualization** — real-time display of scattering, mask, preview, and partition
  maps; adjustable colormap, log scale, beam-center marker; raw-frame browser with
  per-frame or averaged display for multi-frame HDF5 files.
- **Web viewer** — browser-based interface (Dash/Plotly) exposing the full mask and
  partition workflow; launch with `pysimplemask web`.
- **Output** — TIFF mask, Nexus-compatible HDF5 partition (hash + version stamped),
  one-page PDF pipeline summary, `pysimplemask-combine-qmaps` CLI to merge two partition files.

## Installation

### From PyPI
```bash
pip install pysimplemask
```

### From Source
```bash
git clone https://github.com/AdvancedPhotonSource/pySimpleMask.git
cd pySimpleMask
pip install .
```

## GUI Usage

Launch the desktop application:

```bash
pysimplemask                          # or: pysimplemask gui
pysimplemask --path /path/to/data    # open at a specific directory
```

### Workflow

1. **Load data** — select a raw file and click **Load**. Beam center, energy, detector
   distance and pixel size are read from the NeXus metadata; defaults are used if metadata
   is absent.
2. **Define mask** — use the Mask tabs (Blemish/Files, Draw, Binary, Manual, Outlier,
   Parametrization). Click **Evaluate** to preview, **Apply** to commit each layer.
   Undo/Redo/Reset are always available.
3. **Compute partition** — go to the Partition panel, choose a mode and bin counts,
   click **Compute Partition**.
4. **Save** — export as *Mask-Only* (TIFF) or *Nexus-XPCS* (HDF5, includes mask,
   partition maps, and instrument metadata).

GUI state (splitter positions, beamline selection) is persisted in
`~/.pysimplemask/config.json`.

## Headless / Scripted Usage

`import pysimplemask` is **Qt-free**. The full masking and partition pipeline is
available without a display:

```python
from pysimplemask.core import SimpleMaskModel

m = SimpleMaskModel()
m.read_data("scan.h5", beamline="APS_8IDI", begin_idx=0, num_frames=-1)

# threshold mask
m.mask_evaluate("mask_threshold", low=0, high=65535, low_enable=False, high_enable=True)
m.mask_apply("mask_threshold")

# geometric mask (polygon)
m.add_polygon([(r0, c0), (r1, c1), (r2, c2)], mode="exclusive")
m.evaluate_draw()
m.mask_apply("mask_draw")

# q-phi partition
m.compute_partition(mode="q-phi", dq_num=10, sq_num=100, dp_num=36, sp_num=360)
m.save_partition("qmap.hdf")
m.save_mask("mask.tif")
```

Geometry helpers available on the model: `add_polygon`, `add_circle`, `add_ellipse`,
`add_rectangle`, `add_line` (all accept `mode="exclusive"` or `"inclusive"`).

To treat explicit, non-overlapping q-ranges as independent pixel groups (instead of
a linear/log rebin), evaluate/apply a parametrization mask, then pass
`use_groupindex_for_subpartition=True`; `dq_num`/`sq_num`/`dp_num`/`sp_num` all
become each group's own dynamic/static bin counts (per axis):

```python
m.mask_evaluate("mask_parameter", constraints=[
    ("q", "AND", "A^-1", 0.01, 0.05),   # group 1
    ("q", "OR",  "A^-1", 0.05, 0.10),   # group 2
])
m.mask_apply("mask_parameter")
m.compute_partition(mode="q-phi", use_groupindex_for_subpartition=True,
                     dq_num=1, sq_num=50, dp_num=1, sp_num=9)
```

#### How group-index sub-partitioning combines partitions

- **Both axes, independently.** Each group gets its own dynamic/static bins on
  *both* axis0 (q/x) and axis1 (phi/y) — `dq_num`/`sq_num`/`dp_num`/`sp_num` are all
  **per-group** bin counts, not global totals. A group's bin edges come from that
  group's own pixel value range on each axis, not the whole detector.
- **Ellipse mode fits geometry per group.** For `eq-ephi`, each group also gets its
  own ellipse fit (instead of reusing one whole-mask fit), so a group's rho and phi
  values come from that same local fit and stay geometrically consistent with each
  other. Groups with too few pixels to fit fall back to the whole-mask fit.
  A group's constraint axis doesn't need to match the partition axis — e.g. groups
  defined on `q` can drive a sub-partition on an `x-y` or ellipse-mode partition.
- **How the combine works.** Within each group, bin labels are offset so group 2's
  labels never collide with group 1's, and so on; the two axes are then merged with
  the ordinary two-axis `combine_partitions` product — no group-aware logic is
  needed there, because a pixel's axis0 and axis1 labels are always confined to
  that same pixel's own group block, so groups can never collide in the combined
  index.
- **Keep `sq_num`/`sp_num` a multiple of `dq_num`/`dp_num`** (per axis), so the
  static partition remains a strict refinement of the dynamic one. The GUI enforces
  this automatically when computing a partition; script/CLI callers should pick
  compatible values themselves.
- **Validation.** Requires at least one non-empty, non-overlapping set of
  constraint groups (from whichever of Parametrization/Draw was evaluated more
  recently) — overlapping or empty groups raise a descriptive error instead of
  silently mis-binning.
- **GUI.** The checkbox label shows the number of currently active groups (e.g.
  "Use group-index for sub-partitions (2 groups)") and is only enabled once groups
  exist; checking it defaults all four bin-count spinboxes to 1/9 (both remain
  editable afterward).

## Web Viewer

```bash
pysimplemask web                     # start the Dash web server (default port 8050)
pysimplemask web --port 8080
```

Open `http://localhost:8050` in a browser. The web viewer exposes the same mask
and partition workflow as the desktop GUI, including all six mask tabs, four
partition modes, and partition/mask save.

## CLI Tools

```bash
# Build a qmap from a raw scattering file (full headless pipeline).
# A PDF summary report is written alongside the qmap by default.
pysimplemask-build-qmap scan.hdf --output-qmap qmap.hdf --output-mask mask.tif

# Key options (see --help for all):
pysimplemask-build-qmap scan.hdf \
    --beamline APS_8IDI \
    --num-frames 0 \
    --blemish blemish.tif \
    --threshold-high 65535 \
    --param-constraint q:AND:0.01:0.15 \   # geometry-based mask: keep q in [0.01, 0.15]
    --param-constraint phi:AND:-30:30 \     # and phi in [-30°, 30°]
    --mode q-phi \
    --dq-num 10 --sq-num 100 \
    --dp-num 36 --sp-num 360 \
    --output-qmap qmap.hdf \
    --output-mask mask.tif \
    --report summary.pdf              # omit to auto-name, pass "" to skip

# Group-index sub-partitioning: each --param-constraint becomes an independent
# pixel group; --dq-num/--sq-num/--dp-num/--sp-num all become each group's own
# dynamic/static bin counts (per axis).
pysimplemask-build-qmap scan.hdf \
    --param-constraint q:AND:0.01:0.05 \
    --param-constraint q:OR:0.05:0.10 \
    --use-groupindex-for-subpartition \
    --dq-num 1 --sq-num 50 --dp-num 1 --sp-num 9 \
    --output-qmap qmap.hdf

# Merge two existing qmap files
pysimplemask-combine-qmaps file1.hdf file2.hdf output.hdf
```

## Development

```bash
pip install -e ".[dev]"   # install with ruff, pytest, mypy

make test                 # run tests
make lint                 # ruff check
make ui                   # regenerate gui/view/ui_mask.py from gui/view/mask.ui
# or: python src/pysimplemask/gui/view/compile_ui.py
```

The project follows an MVC layout: `src/pysimplemask/core/` (Qt-free, scriptable engine)
and `src/pysimplemask/gui/` (PySide6 + pyqtgraph view/model/control). See `CLAUDE.md`
for the full architecture reference.

## Docker

Build (Docker or Podman):
```bash
docker build -t pysimplemask .
podman build -t pysimplemask .
```

Run on Linux (requires X11 forwarding for the GUI):
```bash
xhost +local:   # allow local X11 connections

# Docker
docker run -it --rm -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd):/data pysimplemask

# Podman (SELinux systems)
podman run -it --rm -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --security-opt label=type:container_runtime_t \
    -v $(pwd):/data pysimplemask
```

A convenience script that auto-detects Docker/Podman, builds if needed, and
launches with X11 forwarding is provided at `scripts/run_container.sh`.

## Credits

- **Author**: Miaoqi Chu (mqichu@anl.gov)
- **License**: Apache 2.0 — Copyright © UChicago Argonne LLC
