# Web Interface — Core Viewer (Sub-project 1 of N)

**Date:** 2026-07-10  
**Branch:** mc_dev  
**Status:** Approved

## Problem

pySimpleMask's only interactive interface is a PySide6 desktop GUI, which requires a
local display and Qt installation. A browser-based interface lets users access the tool
over the network (e.g., from a laptop pointed at a beamline workstation) and removes the
Qt dependency for interactive use.

## Scope — This Sub-project

Sub-project 1 is the **core viewer only**:

- Launch a web server from the CLI (host + port configurable)
- Load a scattering data file (any format the existing readers support, including XPCS
  result files via auto-detect)
- Display the detector image with colormap, log-scale, and display-channel controls
- Editable metadata panel (beam center, energy, distance, pixel size)
- Find Center, Goto Max, Swap X/Y, Update Parameters actions
- Hover status bar (pixel coordinates + intensity; q/phi once qmap is loaded)

**Not in this sub-project:** masking controls, partition computation, save/export.
Those are future sub-projects that extend this foundation.

## Technology Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| UI framework | Plotly Dash + explicit Flask | Python-only UI, no JS files; Flask base enables future download endpoints |
| Image rendering | `px.imshow` + `go.Figure` | Handles large 2D float arrays, built-in zoom/pan/hover, log-scale |
| Server model | Single-user, module-level singleton | Local tool; no session management needed |
| Dependency packaging | `pysimplemask[web]` optional extra | Core and Qt GUI remain installable without Dash |

## File Structure

```
src/pysimplemask/web/
├── __init__.py
├── server.py        — Flask app, Dash attachment, SimpleMaskModel singleton, main_web()
├── layout.py        — Full Dash component tree (all IDs, initial values)
├── callbacks.py     — All @callback functions; imports model from server.py
└── image_utils.py   — make_figure(arr, colormap, log_scale, center_vh) → go.Figure
```

No other files are added or modified in `core/` or `gui/`.

## Layout

Two-column, matching the Qt GUI's proportions.

### Left sidebar (~30% width)

**File loading section:**
- `dcc.Input` — file path text input (full filesystem path, since server has local access)
- `dcc.Dropdown` — beamline selector: `APS_8IDI`, `APS_9IDD`, `NativeFiles`
  (XPCS result files are auto-detected by `file_handler.get_handler` regardless of this
  selection)
- `dcc.Input` × 2 — `begin_idx` (default `0`) and `num_frames` (default `-1`)
- `html.Button` — **Load**
- Status message area (red on error, green on success)

**Metadata editor section (populated after load):**
- `dcc.Input` fields for: `beam_center_x`, `beam_center_y`, `energy`,
  `detector_distance`, `pixel_size` — each with a unit label
- `html.Button` × 4 — **Find Center**, **Goto Max**, **Swap X/Y**, **Update Parameters**

### Main panel (~70% width)

**Controls row:**
- `dcc.Dropdown` — display channel: `scattering`, `scattering * mask`, `mask`,
  `dqmap_partition`, `sqmap_partition`, `preview`; populated from `DISPLAY_FIELD`
- `dcc.Dropdown` — colormap: `jet`, `viridis`, `gray`, `hot`, `cool`, `RdBu`
- `dcc.Checklist` — log scale toggle

**Image:**
- `dcc.Graph(id="detector-image")` — fills remaining panel height

**Hover status bar:**
- `html.Div` — updated from `hoverData`; shows `x=, y=, intensity=`; after qmap loaded,
  also shows `q=Å⁻¹, phi=°`

## Callbacks

All callbacks are in `callbacks.py` and import the model singleton from `server.py`.
The model is a module-level `SimpleMaskModel()` instance.

### 1. Load (`load-btn` click)

```
Inputs:  file-path, beamline, begin-idx, num-frames
Outputs: detector-image.figure
         beam-center-x.value, beam-center-y.value,
         energy.value, detector-distance.value, pixel-size.value
         status-msg.children
         display-channel.options
```

Calls `model.read_data(fname, beamline, begin_idx=..., num_frames=...)`.
On success: reads metadata into the form fields, calls `make_figure` with channel 0
(scattering), returns updated figure and all metadata values.
On failure: returns `no_update` for figure, error text for status.

### 2. Update Parameters (`update-params-btn` click)

```
Inputs:  beam-center-x, beam-center-y, energy, detector-distance, pixel-size
Outputs: detector-image.figure, status-msg.children
```

Calls `model.update_parameters({...})`, then `make_figure` on the current channel.

### 3. Find Center (`find-center-btn` click)

```
Inputs:  (none — uses model state)
Outputs: beam-center-x.value, beam-center-y.value, status-msg.children
```

Calls `model.find_center()`, returns the new center coordinates.

### 4. Goto Max (`goto-max-btn` click)

```
Inputs:  (none)
Outputs: beam-center-x.value, beam-center-y.value, status-msg.children
```

Calls `model.goto_max()`, returns coordinates.

### 5. Display controls (channel / colormap / log-scale change)

```
Inputs:  display-channel, colormap, log-scale
Outputs: detector-image.figure
```

Reads `model.dset.data_display[index]`, calls `make_figure`. Uses `Patch()` to update `coloraxis.colorscale` when only the colormap dropdown
changes (data unchanged), avoiding re-serialization of the full image array. Channel
or log-scale changes require a full `make_figure` rebuild.

### 6. Hover status bar (hoverData change)

```
Inputs:  detector-image.hoverData, display-channel
Outputs: hover-status.children
```

Extracts `(col, row)` from hoverData, calls `model.dset.get_coordinates(col, row, index)`
to get the formatted coordinate string.

## Image Utilities (`image_utils.py`)

```python
def make_figure(arr, colormap="jet", log_scale=False, center_vh=None) -> go.Figure:
```

- If `log_scale` and array has positive values: `np.log10(np.maximum(arr, floor))` where
  `floor = arr[arr > 0].min()`
- `px.imshow(arr, color_continuous_scale=colormap, origin="upper", aspect="equal")`
- If `center_vh` is not None: add a scatter trace with two orthogonal line segments
  (crosshair) at `(center_vh[1], center_vh[0])` in white, semi-transparent
- Returns `go.Figure` with axis labels suppressed and margins minimized

**Performance note:** `px.imshow` with a 2162×2068 float32 array serializes ~18 MB as
JSON, taking ~1–2 s per callback. This is acceptable for load/update events. Display
control changes (colormap, log-scale) re-call `make_figure` and re-serialize; this is a
known cost for sub-project 1. Future sub-projects may optimize with server-side image
encoding (PNG bytes) if needed.

## CLI Entry Point

### `pyproject.toml` additions

```toml
[project.optional-dependencies]
web = ["dash>=2.14", "plotly>=5.18"]

[project.scripts]
# (added alongside existing scripts)
pysimplemask-web = "pysimplemask.web.server:main_web"
```

### `main_web` signature

```python
def main_web(
    host: str = "127.0.0.1",
    port: int = 8050,
    path: str | None = None,
    debug: bool = False,
) -> None:
```

Parsed via `argparse` (already available, no new dep):

```bash
pysimplemask-web                          # localhost:8050
pysimplemask-web --host 0.0.0.0           # LAN-accessible
pysimplemask-web --port 8888
pysimplemask-web --path /data/scan.h5     # pre-populate file path field
pysimplemask-web --debug                  # Dash hot-reload
```

Prints on startup:
```
pySimpleMask web interface running at http://127.0.0.1:8050
```

## Error Handling

- If `model.read_data` returns `False` or raises: show error in status bar, leave
  image blank, leave metadata fields at previous values.
- If model is not loaded when Display or Find Center callbacks fire: callbacks check
  `model.is_ready()` and return `no_update` with a "Load a file first" status message.
- If `dash` is not installed and user runs `pysimplemask-web`: Python raises
  `ModuleNotFoundError`; this is acceptable (user must `pip install pysimplemask[web]`).

## Testing

- **Smoke test** (`tests/web/test_web_import.py`): `from pysimplemask.web.server import server`
  imports without error; `from pysimplemask.web.image_utils import make_figure` imports
  without error. Skipped if `dash` is not installed (`pytest.importorskip("dash")`).
- **Unit test** (`tests/web/test_image_utils.py`): `make_figure` with a 16×16 synthetic
  array returns a `go.Figure` with correct dimensions; log-scale applies correctly;
  crosshair trace is added when `center_vh` is provided.
- No callback integration tests in sub-project 1 (require a live Dash server).

## Future Sub-projects

| Sub-project | Adds |
|-------------|------|
| 2 | Threshold mask tab (evaluate + apply + undo/redo) |
| 3 | q-phi partition computation + display |
| 4 | Full mask parity (draw ROI, outlier, parametrize) |
| 5 | Save/export (HDF5, TIFF, download via Flask endpoint) |
