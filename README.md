# pySimpleMask

[![PyPI version](https://img.shields.io/pypi/v/pysimplemask.svg)](https://pypi.python.org/pypi/pysimplemask)

**pySimpleMask** is a tool for creating masks and Q-partition maps for X-ray scattering
patterns, supporting SAXS, WAXS, and XPCS data reduction. It ships both a desktop GUI
and a **headless Python API** that can drive the full pipeline from scripts.

## Features

- **Versatile data support** — HDF5 (NeXus/XPCS), IMM, Rigaku 500k/3M binary, TIFF.
  Supported beamlines: APS 8-ID-I (transmission) and APS 9-ID-D (reflection/GISAXS).
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
  - Parametric masking by q-map range (q, phi, x, y, …).
  - Undo / redo / reset mask history.
- **Beam-center finding** — iterative centro-symmetry cross-correlation, converges
  in 1–2 passes; bounded crop for speed on large detectors.
- **Partition generation**
  - Q-Phi (dynamic + static resolution pair).
  - X-Y spatial partitions.
  - Ellipse-corrected Q-Phi (eq-ephi).
  - Custom axis pair from any q-map channel.
- **Visualization** — real-time display of scattering, mask, preview, and partition
  maps; adjustable colormap, log scale, beam-center marker.
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
pysimplemask
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
- **License**: BSD 3-Clause
