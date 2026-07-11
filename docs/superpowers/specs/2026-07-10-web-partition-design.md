# Web Interface — Partition Section (Sub-project 3 of N)

**Date:** 2026-07-10
**Branch:** mc_web
**Status:** Approved

## Problem

The web viewer can load data and apply masks but cannot compute q-partition maps or save results. Users need partition computation so the web app produces XPCS-ready HDF5 output without a Qt installation.

## Scope

All 4 partition modes from the PySide6 GUI:
1. q-phi
2. xy-mesh
3. General (custom 2-axis)
4. eq-ephi (ellipse-corrected)

Plus save functionality: partition HDF5 and mask TIFF, both server-side (local filesystem path input).

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| All 4 modes | Yes | Full parity; scope is bounded |
| Save included | Yes | Needed for XPCS pipeline output |
| Display after compute | Auto-switch to `dqmap_partition` (channel 3) | Immediate visual feedback, mirrors mask evaluate pattern |
| Controls placement | Extend sidebar below mask section | Consistent with existing layout |
| Code structure | 2 new files (`partition_layout.py`, `partition_callbacks.py`) | Mirrors mask pattern; each file has one responsibility |

## File Structure

```
src/pysimplemask/web/
├── partition_layout.py    NEW — build_partition_section() → html.Div
└── partition_callbacks.py NEW — compute + save + axis-repopulate callbacks

tests/web/
└── test_partition_layout.py  NEW — smoke tests
```

**Modified:**
- `layout.py`: import `build_partition_section`; append `*build_partition_section().children` in `_sidebar()`
- `server.py`: guarded import of `partition_callbacks` in `main_web()`

## Layout

### Sidebar additions (below mask section)

```
html.Hr()
html.H4("Partition")
dcc.Tabs(id="partition-tabs", value="q-phi") — 4 tabs
[per section below]
html.Div — compute row: [Compute button (id="partition-compute-btn")]
                          [partition-status div (id="partition-status")]
html.Hr()
html.H4("Save")
dcc.Input(id="save-path", type="text", placeholder="/path/to/output")
html.Div — save row: [Save Partition (HDF5) (id="save-partition-btn")]
                     [Save Mask (TIFF) (id="save-mask-btn")]
html.Div(id="save-status")
```

### Tab controls

**`q-phi` tab (value="q-phi")**

| ID | Component | Default | Label |
|----|-----------|---------|-------|
| `qphi-dq-num` | `dcc.Input(type="number")` | 10 | Dynamic q bins |
| `qphi-dp-num` | `dcc.Input(type="number")` | 36 | Dynamic φ bins |
| `qphi-sq-num` | `dcc.Input(type="number")` | 100 | Static q bins |
| `qphi-sp-num` | `dcc.Input(type="number")` | 360 | Static φ bins |
| `qphi-style` | `dcc.Dropdown` | `"linear"` | Style (linear/log) |
| `qphi-phi-offset` | `dcc.Input(type="number")` | 0.0 | φ offset (°) |
| `qphi-symmetry-fold` | `dcc.Input(type="number", min=1, max=12)` | 1 | Symmetry fold |

**`xy-mesh` tab (value="xy-mesh")**

| ID | Component | Default | Label |
|----|-----------|---------|-------|
| `xy-dq-num` | `dcc.Input(type="number")` | 10 | Dynamic x bins |
| `xy-dp-num` | `dcc.Input(type="number")` | 10 | Dynamic y bins |
| `xy-sq-num` | `dcc.Input(type="number")` | 100 | Static x bins |
| `xy-sp-num` | `dcc.Input(type="number")` | 100 | Static y bins |

**`general` tab (value="general") — custom 2-axis**

| ID | Component | Default | Label |
|----|-----------|---------|-------|
| `gen-map0` | `dcc.Dropdown` | `None` | Axis 0 (populated from qmap) |
| `gen-dq-num` | `dcc.Input(type="number")` | 10 | Dynamic axis-0 bins |
| `gen-sq-num` | `dcc.Input(type="number")` | 100 | Static axis-0 bins |
| `gen-style` | `dcc.Dropdown` | `"linear"` | Style (linear/log, shared by both axes) |
| `gen-map1` | `dcc.Dropdown` | `None` | Axis 1 (populated from qmap) |
| `gen-dp-num` | `dcc.Input(type="number")` | 36 | Dynamic axis-1 bins |
| `gen-sp-num` | `dcc.Input(type="number")` | 360 | Static axis-1 bins |

**`eq-ephi` tab (value="eq-ephi")**

| ID | Component | Default | Label |
|----|-----------|---------|-------|
| `eqephi-dq-num` | `dcc.Input(type="number")` | 10 | Dynamic q bins |
| `eqephi-dp-num` | `dcc.Input(type="number")` | 36 | Dynamic φ bins |
| `eqephi-sq-num` | `dcc.Input(type="number")` | 100 | Static q bins |
| `eqephi-sp-num` | `dcc.Input(type="number")` | 360 | Static φ bins |

## Callbacks

### 1. Compute (single callback for all 4 modes)

```
Input:  partition-compute-btn.n_clicks
State:  partition-tabs.value
State:  all tab-specific inputs (qphi-*, xy-*, gen-*, eqephi-*)
State:  colormap.value, log-scale.value
Outputs:
  Output("detector-image", "figure",        allow_duplicate=True)
  Output("display-channel", "value",        allow_duplicate=True)
  Output("partition-status", "children",    allow_duplicate=True)
```

Model calls by tab:

```python
if tab == "q-phi":
    model.compute_partition(
        "q-phi",
        dq_num=qphi_dq_num, sq_num=qphi_sq_num,
        dp_num=qphi_dp_num, sp_num=qphi_sp_num,
        style=qphi_style or "linear",
        phi_offset=float(qphi_phi_offset or 0),
        symmetry_fold=int(qphi_symmetry_fold or 1),
    )
elif tab == "xy-mesh":
    model.compute_partition(
        "x-y",
        dq_num=xy_dq_num, sq_num=xy_sq_num,
        dp_num=xy_dp_num, sp_num=xy_sp_num,
    )
elif tab == "general":
    # compute_partition splits mode on "-" to derive map_names; e.g. "q-phi" → ("q","phi")
    mode_str = f"{gen_map0}-{gen_map1}"
    model.compute_partition(
        mode_str,
        dq_num=gen_dq_num, sq_num=gen_sq_num,
        dp_num=gen_dp_num, sp_num=gen_sp_num,
        style=gen_style or "linear",
    )
elif tab == "eq-ephi":
    model.compute_partition(
        "eq-ephi",
        dq_num=eqephi_dq_num, sq_num=eqephi_sq_num,
        dp_num=eqephi_dp_num, sp_num=eqephi_sp_num,
    )
```

After compute:
```python
fig = make_figure(model.dset.data_display[3], colormap=..., log_scale=...,
                  center_vh=model.get_center("vh"))
return fig, 3, "Partition computed."
```

**Note on General mode:** `model.compute_partition` splits the mode string on `"-"` to derive `map_names` (e.g. `"q-phi"` → `("q", "phi")`). The General tab constructs the mode string as `f"{gen_map0}-{gen_map1}"` from the selected axis dropdowns. `compute_partition_general` takes a single `style` parameter shared by both axes.

### 2. Save Partition

```
Input:  save-partition-btn.n_clicks
State:  save-path.value
Output: save-status.children  (primary owner)
```

```python
if not model.is_ready() or model.new_partition is None:
    return "Compute a partition first."
if not save_path:
    return "Enter an output path."
model.save_partition(save_path)
return f"Partition saved to {save_path}"
```

### 3. Save Mask

```
Input:  save-mask-btn.n_clicks
State:  save-path.value
Output: save-status.children  (allow_duplicate=True)
```

```python
if not model.is_ready():
    return "Load a file first."
if not save_path:
    return "Enter an output path."
mask_path = save_path if save_path.endswith((".tif", ".tiff")) else save_path + ".tif"
model.save_mask(mask_path)
return f"Mask saved to {mask_path}"
```

### 4. General axis dropdowns — repopulate on load

```
Input:  model-loaded.data
Output: gen-map0.options, gen-map1.options
prevent_initial_call=False
```

```python
if not model.is_ready():
    return [], []
opts = [{"label": k, "value": k} for k in model.qmap.keys()]
return opts, opts
```

## Testing

### `tests/web/test_partition_layout.py` (smoke, importorskip dash)

Required IDs to verify:
- `partition-tabs`, `partition-compute-btn`, `partition-status`
- `save-path`, `save-partition-btn`, `save-mask-btn`, `save-status`
- All q-phi inputs: `qphi-dq-num`, `qphi-dp-num`, `qphi-sq-num`, `qphi-sp-num`, `qphi-style`, `qphi-phi-offset`, `qphi-symmetry-fold`
- All xy inputs: `xy-dq-num`, `xy-dp-num`, `xy-sq-num`, `xy-sp-num`
- All general inputs: `gen-map0`, `gen-dq-num`, `gen-sq-num`, `gen-style`, `gen-map1`, `gen-dp-num`, `gen-sp-num`
- All eq-ephi inputs: `eqephi-dq-num`, `eqephi-dp-num`, `eqephi-sq-num`, `eqephi-sp-num`

Tests:
1. `test_build_partition_section_returns_html_div`
2. `test_all_required_ids_present`
3. `test_four_tabs_present` — tab values: `{"q-phi", "xy-mesh", "general", "eq-ephi"}`

### `tests/web/test_web_import.py` — add one test

```python
def test_partition_callbacks_imports():
    import pysimplemask.web.partition_callbacks  # noqa: F401
```

No callback integration tests (require live Dash server).
