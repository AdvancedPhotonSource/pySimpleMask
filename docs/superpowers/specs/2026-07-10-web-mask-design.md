# Web Interface — Mask Section (Sub-project 2 of N)

**Date:** 2026-07-10
**Branch:** mc_web
**Status:** Approved

## Problem

The web viewer (Sub-project 1) provides file loading and image display but no masking capability. Users need all six mask methods from the PySide6 GUI available in the browser so they can create and refine masks without a Qt installation.

## Scope

All 6 mask tabs from the PySide6 GUI, implemented in Dash with the same evaluate → preview → apply flow:

1. Blemish / Files
2. Binary Threshold
3. Draw (Plotly newshape)
4. Manual pixel list
5. Outlier removal
6. Parametrization (geometry constraints)

**Not in this sub-project:** partition computation, save/export.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Mask controls placement | Extend sidebar below metadata | Consistent with current layout; sidebar scrolls |
| Draw interaction | Plotly `newshape` drag mode | No custom JS; handles rect, circle, closed polygon |
| Preview feedback | Auto-switch display channel | Mirrors Qt GUI; user sees what would be masked |
| Undo/redo | Reset only | Simpler; covers the primary recovery use case |
| Code structure | 3 new files (mask_layout, mask_callbacks, shape_utils) | Keeps layout.py and callbacks.py focused |

## File Structure

```
src/pysimplemask/web/
├── mask_layout.py      NEW — build_mask_section() → html.Div
├── mask_callbacks.py   NEW — all mask @callback functions (13 total)
└── shape_utils.py      NEW — plotly_shapes_to_mask() for Draw tab

tests/web/
├── test_shape_utils.py  NEW — 7 unit tests (no model, no Dash)
└── test_mask_layout.py  NEW — smoke tests for layout structure
```

**Modified:**
- `layout.py`: `_sidebar()` adds `build_mask_section()` after action buttons; root div gets two new stores: `dcc.Store(id="drawn-shapes", data=[])` and `dcc.Store(id="param-constraints", data=[])`
- `server.py`: `main_web()` imports `mask_callbacks` (alongside existing `callbacks`)

No changes to `callbacks.py` or `core/`.

## Layout

### Sidebar additions (below existing Swap X/Y / Update Params buttons)

```
html.Hr()
html.H4("Mask")
html.Div — top bar: [Reset button] [mask-status div]
dcc.Tabs(id="mask-tabs") — 6 tabs (see below)
```

`mask-status` (`html.Div(id="mask-status")`) is dedicated to mask feedback and does not share `status-msg` to avoid clobbering load/metadata messages.

### Tab IDs and controls

**`blemish` — Blemish / Files**
- Blemish sub-row: `dcc.Input(id="blemish-path")` + `dcc.Input(id="blemish-key", value="/exchange/data")` + **Evaluate** (`id="blemish-eval-btn"`) + **Apply** (`id="blemish-apply-btn"`)
- File sub-row: `dcc.Input(id="maskfile-path")` + `dcc.Input(id="maskfile-key", value="/exchange/data")` + **Evaluate** (`id="maskfile-eval-btn"`) + **Apply** (`id="maskfile-apply-btn"`)

**`threshold` — Binary Threshold**
- Low: `dcc.Checklist(id="thresh-low-enable")` + `dcc.Input(id="thresh-low", type="number")`
- High: `dcc.Checklist(id="thresh-high-enable")` + `dcc.Input(id="thresh-high", type="number")`
- Morphology row: **Erode** (`id="morph-erode"`) / **Dilate** (`id="morph-dilate"`) / **Open** (`id="morph-open"`) / **Close** (`id="morph-close"`)
- **Evaluate** (`id="thresh-eval-btn"`) + **Apply** (`id="thresh-apply-btn"`)

**`draw` — Draw (Plotly newshape)**
- `dcc.RadioItems(id="draw-shape")` — options: `"drawcircle"` / `"drawrect"` / `"drawclosedpath"`; labels: Circle / Rectangle / Polygon
- `dcc.RadioItems(id="draw-mode")` — Exclusive (mask out pixels inside ROI) / Inclusive (keep pixels inside ROI)
- **Activate Draw** toggle button (`id="draw-activate-btn"`) — switches `detector-image` `dragmode` to the selected shape type; re-click deactivates (returns to `"pan"`)
- **Clear** (`id="draw-clear-btn"`) + **Evaluate** (`id="draw-eval-btn"`) + **Apply** (`id="draw-apply-btn"`)
- (`drawn-shapes` store lives in root div, not in the tab)

**`manual` — Manual Pixel List**
- `dcc.Textarea(id="manual-pixels", placeholder="row col\nrow col\n...")` — whitespace-delimited pairs
- `dcc.Upload(id="manual-upload", accept=".txt,.csv,.json")` — file upload
- **Evaluate** (`id="manual-eval-btn"`) + **Apply** (`id="manual-apply-btn"`)

**`outlier` — Outlier Removal**
- `dcc.Dropdown(id="outlier-target")` — Circular Rings / Adjacent Boxes
- `dcc.Dropdown(id="outlier-method")` — MAD / Percentile
- `dcc.Input(id="outlier-cutoff", type="number", value=3.0)`
- `dcc.Input(id="outlier-param", type="number", value=180)` — label updates: "Num rings" for Circular, "Box size (px)" for Adjacent
- **Evaluate** (`id="outlier-eval-btn"`) + **Apply** (`id="outlier-apply-btn"`)

**`parameterize` — Parametrization**
- `html.Div(id="param-rows")` — dynamically rendered constraint rows
- (`param-constraints` store lives in root div, not in the tab); each row has: xmap `dcc.Dropdown` (options from `model.qmap.keys()` after load), logic `dcc.Dropdown` (AND/OR/NOT), vmin/vmax `dcc.Input`
- **Add row** (`id="param-add-btn"`) / **Remove last** (`id="param-remove-btn"`) — update `param-constraints` store
- **Evaluate** (`id="param-eval-btn"`) + **Apply** (`id="param-apply-btn"`)

## Callback Architecture

### Shared evaluate / apply / reset pattern

Every mask tab follows the same two-step flow. Callbacks output:
- `Output("detector-image", "figure", allow_duplicate=True)`
- `Output("display-channel", "value", allow_duplicate=True)` — 5 (preview) on evaluate, 1 (scattering × mask) on apply, 0 (scattering) on reset
- `Output("mask-status", "children")`

All use `prevent_initial_call=True` and guard with:
```python
if not model.is_ready():
    return no_update, no_update, "Load a file first."
```

**Evaluate flow:**
```python
model.mask_evaluate(target, **kwargs)
fig = make_figure(model.dset.data_display[5], ...)  # preview channel
return fig, 5, "Preview ready — click Apply to commit."
```

**Apply flow:**
```python
model.mask_apply(target)
fig = make_figure(model.dset.data_display[1], ...)  # scattering * mask
return fig, 1, "Applied."
```

**Reset flow (shared button, any tab):**
```python
model.mask_action("reset")
fig = make_figure(model.dset.data_display[0], ...)  # scattering
return fig, 0, "Mask reset."
```

### Model calls per tab

| Tab | Evaluate kwargs | Apply target |
|-----|----------------|--------------|
| Blemish | `mask_evaluate("mask_blemish", fname=..., key=...)` | `"mask_blemish"` |
| File | `mask_evaluate("mask_file", fname=..., key=...)` | `"mask_file"` |
| Threshold | `mask_evaluate("mask_threshold", saxs_lin=model.dset.scat, low=..., high=..., low_enable=..., high_enable=...)` | `"mask_threshold"` |
| Morphology | `model.mask_kernel.workers["mask_threshold"].morphology(action)` then `mask_apply("mask_threshold")` | `"mask_threshold"` |
| Draw | `mask_evaluate("mask_draw", arr=plotly_shapes_to_mask(shapes, model.shape, mode))` | `"mask_draw"` |
| Manual | `mask_evaluate("mask_list", zero_loc=parsed_pixel_array)` | `"mask_list"` |
| Outlier (Rings) | `_, zero_loc = model.compute_saxs1d(method, cutoff, num_rings)` then `mask_evaluate("mask_outlier", zero_loc=zero_loc)` | `"mask_outlier"` |
| Outlier (Boxes) | `_, zero_loc = model.compute_adjacent_saxs1d(method, cutoff, box_size)` then `mask_evaluate("mask_outlier", zero_loc=zero_loc)` | `"mask_outlier"` |
| Parameterize | `mask_evaluate("mask_parameter", constraints=[[xmap, logic, unit, vmin, vmax], ...])` | `"mask_parameter"` |

### Draw tab extra callbacks

**`relayoutData` → `drawn-shapes` store:**
```python
Input("detector-image", "relayoutData") → Output("drawn-shapes", "data")
```
Appends any new `shapes` entries from `relayoutData` to the existing store list.

**Activate Draw toggle:**
```python
Input("draw-activate-btn", "n_clicks"), State("draw-shape", "value")
→ Output("detector-image", "figure", allow_duplicate=True)
```
Updates `fig.layout.dragmode` to the selected shape type (or `"pan"` to deactivate).

**Clear shapes:**
```python
Input("draw-clear-btn", "n_clicks")
→ Output("drawn-shapes", "data"), Output("detector-image", "figure", allow_duplicate=True)
```
Resets store to `[]` and removes shapes from `fig.layout.shapes`.

### Parametrize tab extra callbacks

**Add/Remove constraint row:**
```python
Input("param-add-btn"/"param-remove-btn", "n_clicks")
→ Output("param-constraints", "data")
```

**Render rows from store:**
```python
Input("param-constraints", "data"), Input("model-loaded", "data")
→ Output("param-rows", "children")
```
xmap dropdown options are populated from `list(model.qmap.keys())` after load; safe to return empty list when model not ready.

## `shape_utils.py`

```python
def plotly_shapes_to_mask(
    shapes: list[dict],
    detector_shape: tuple[int, int],
    mode: str = "exclusive",
) -> np.ndarray:
```

**Conversion table:**

| Plotly `type` | Plotly fields | `core.rasterize` call |
|---|---|---|
| `"rect"` | `x0, y0, x1, y1` | `rectangle_vertices(r0=y0, c0=x0, r1=y1, c1=x1)` → `RoiPolygon` |
| `"circle"` | `x0, y0, x1, y1` (bbox) | `circle_vertices(cy=(y0+y1)/2, cx=(x0+x1)/2, r=(x1-x0)/2)` → `RoiPolygon` |
| `"path"` | SVG string `"M x,y L x,y … Z"` | parse `M`/`L` coordinate tokens → vertex list → `RoiPolygon` |

All shapes are OR'd into a combined ROI mask.
- `mode="exclusive"` (default): `keep_mask = ~roi_mask` — masks OUT pixels inside ROIs
- `mode="inclusive"`: `keep_mask = roi_mask` — keeps ONLY pixels inside ROIs

Returns `np.ndarray` of `detector_shape`, dtype `bool`.

**Note on coordinates:** Plotly image coordinates are (col, row) = (x, y) with `origin="upper"`. `x` maps to column, `y` maps to row. All rasterize calls receive `(row, col)` as required by `core.rasterize`.

## Testing

### `tests/web/test_shape_utils.py` (unit, no model, no Dash)

| Test | Asserts |
|------|---------|
| `test_rect_fills_region` | Rect shape sets `False` for the correct rectangular region (exclusive mode) |
| `test_circle_fills_region` | Circle shape correctly masks circular pixels |
| `test_polygon_path_parses` | SVG `M x,y L x,y Z` path produces the correct vertex polygon |
| `test_exclusive_masks_inside` | `mode="exclusive"` → pixels inside ROI are `False` |
| `test_inclusive_keeps_inside` | `mode="inclusive"` → pixels inside ROI are `True`, outside are `False` |
| `test_multiple_shapes_combined` | Two non-overlapping rects → both regions masked |
| `test_out_of_bounds_no_crash` | Shape with coords outside detector bounds returns valid mask without exception |

### `tests/web/test_mask_layout.py` (smoke, importorskip dash)

- `build_mask_section()` returns `html.Div`
- Component tree contains all 6 tab IDs: `blemish`, `threshold`, `draw`, `manual`, `outlier`, `parameterize`
- All evaluate button IDs present: `blemish-eval-btn`, `thresh-eval-btn`, `draw-eval-btn`, `manual-eval-btn`, `outlier-eval-btn`, `param-eval-btn`
- All apply button IDs present: `blemish-apply-btn`, `thresh-apply-btn`, `draw-apply-btn`, `manual-apply-btn`, `outlier-apply-btn`, `param-apply-btn`
- Reset button ID `mask-reset-btn` present
- `mask-status` div present
