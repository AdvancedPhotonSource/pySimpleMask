# Web Partition Section Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4-mode partition computation and HDF5/TIFF save to the pySimpleMask web viewer sidebar.

**Architecture:** Two new files mirror the mask pattern — `partition_layout.py` (4-tab Dash layout) and `partition_callbacks.py` (compute + save + axis-repopulate callbacks). `layout.py` and `server.py` each get one-line additions. No changes to `core/`.

**Tech Stack:** Python, Plotly Dash 4.x, existing `pysimplemask.core.model.SimpleMaskModel`

## Global Constraints

- Environment: `/local/MQICHU/envs/l2606_simplemask_refact/bin/`
- All web tests use `pytest.importorskip("dash")` — skipped when Dash absent
- `core/` must not be modified
- Compute callback outputs: `detector-image.figure` (`allow_duplicate=True`), `display-channel.value` (`allow_duplicate=True`), `partition-status.children` (`allow_duplicate=True`) — all three are secondary (mask_reset owns the primaries for display-channel and mask-status; partition-status is brand new so its first callback is the primary)
- After compute: switch display channel to 3 (`dqmap_partition`)
- `save-status.children`: `save_partition_cb` is the primary owner (no `allow_duplicate`); `save_mask_cb` uses `allow_duplicate=True`
- General mode: `model.compute_partition(f"{gen_map0}-{gen_map1}", dq_num=..., sq_num=..., dp_num=..., sp_num=..., style=...)`
- Ruff must pass clean on all new files

---

## File Map

| Status | File | Responsibility |
|--------|------|----------------|
| Create | `src/pysimplemask/web/partition_layout.py` | `build_partition_section() → html.Div` |
| Create | `src/pysimplemask/web/partition_callbacks.py` | 4 callbacks: compute, save partition, save mask, repopulate axes |
| Create | `tests/web/test_partition_layout.py` | 3 smoke tests |
| Modify | `src/pysimplemask/web/layout.py` | Import + append `*build_partition_section().children` in `_sidebar()` |
| Modify | `src/pysimplemask/web/server.py` | Guarded import of `partition_callbacks` in `main_web()` |
| Modify | `tests/web/test_web_import.py` | Add `test_partition_callbacks_imports` |

---

## Component ID Reference

| ID | Component | Tab / Section |
|----|-----------|---------------|
| `partition-tabs` | `dcc.Tabs` | global |
| `partition-compute-btn` | Button | global |
| `partition-status` | Div | global |
| `save-path` | `dcc.Input` | Save |
| `save-partition-btn` | Button | Save |
| `save-mask-btn` | Button | Save |
| `save-status` | Div | Save |
| `qphi-dq-num` | `dcc.Input` | q-phi |
| `qphi-dp-num` | `dcc.Input` | q-phi |
| `qphi-sq-num` | `dcc.Input` | q-phi |
| `qphi-sp-num` | `dcc.Input` | q-phi |
| `qphi-style` | `dcc.Dropdown` | q-phi |
| `qphi-phi-offset` | `dcc.Input` | q-phi |
| `qphi-symmetry-fold` | `dcc.Input` | q-phi |
| `xy-dq-num` | `dcc.Input` | xy-mesh |
| `xy-dp-num` | `dcc.Input` | xy-mesh |
| `xy-sq-num` | `dcc.Input` | xy-mesh |
| `xy-sp-num` | `dcc.Input` | xy-mesh |
| `gen-map0` | `dcc.Dropdown` | general |
| `gen-dq-num` | `dcc.Input` | general |
| `gen-sq-num` | `dcc.Input` | general |
| `gen-style` | `dcc.Dropdown` | general |
| `gen-map1` | `dcc.Dropdown` | general |
| `gen-dp-num` | `dcc.Input` | general |
| `gen-sp-num` | `dcc.Input` | general |
| `eqephi-dq-num` | `dcc.Input` | eq-ephi |
| `eqephi-dp-num` | `dcc.Input` | eq-ephi |
| `eqephi-sq-num` | `dcc.Input` | eq-ephi |
| `eqephi-sp-num` | `dcc.Input` | eq-ephi |

---

## Task 1: `partition_layout.py` + smoke tests

**Files:**
- Create: `src/pysimplemask/web/partition_layout.py`
- Create: `tests/web/test_partition_layout.py`

**Interfaces:**
- Produces: `build_partition_section() -> html.Div` — root element containing Hr, H4, Tabs (4 tabs), compute row, Hr, Save section

- [ ] **Step 1: Write failing smoke tests**

Create `tests/web/test_partition_layout.py`:

```python
"""Smoke tests for partition_layout — verify component IDs and structure."""

import pytest

dash = pytest.importorskip("dash")
from dash import dcc, html  # noqa: E402

from pysimplemask.web.partition_layout import build_partition_section  # noqa: E402

REQUIRED_IDS = {
    "partition-tabs", "partition-compute-btn", "partition-status",
    "save-path", "save-partition-btn", "save-mask-btn", "save-status",
    # q-phi
    "qphi-dq-num", "qphi-dp-num", "qphi-sq-num", "qphi-sp-num",
    "qphi-style", "qphi-phi-offset", "qphi-symmetry-fold",
    # xy-mesh
    "xy-dq-num", "xy-dp-num", "xy-sq-num", "xy-sp-num",
    # general
    "gen-map0", "gen-dq-num", "gen-sq-num", "gen-style",
    "gen-map1", "gen-dp-num", "gen-sp-num",
    # eq-ephi
    "eqephi-dq-num", "eqephi-dp-num", "eqephi-sq-num", "eqephi-sp-num",
}


def _collect_ids(component: object, found: set) -> None:
    if hasattr(component, "id") and component.id is not None:
        found.add(component.id)
    children = getattr(component, "children", None)
    if children is None:
        return
    items = children if isinstance(children, list) else [children]
    for child in items:
        _collect_ids(child, found)


def test_build_partition_section_returns_html_div():
    assert isinstance(build_partition_section(), html.Div)


def test_all_required_ids_present():
    section = build_partition_section()
    found: set = set()
    _collect_ids(section, found)
    missing = REQUIRED_IDS - found
    assert not missing, f"Missing IDs: {missing}"


def test_four_tabs_present():
    section = build_partition_section()

    def find_tabs(comp):
        if isinstance(comp, dcc.Tabs):
            return comp
        for child in getattr(comp, "children", None) or []:
            result = find_tabs(child)
            if result is not None:
                return result
        return None

    tabs = find_tabs(section)
    assert tabs is not None, "dcc.Tabs not found"
    values = {t.value for t in tabs.children if hasattr(t, "value")}
    assert values == {"q-phi", "xy-mesh", "general", "eq-ephi"}
```

- [ ] **Step 2: Run failing tests**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/web/test_partition_layout.py -v
```

Expected: all 3 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/pysimplemask/web/partition_layout.py`**

```python
"""Partition section layout — 4-tab Dash component tree for the sidebar."""

from __future__ import annotations

from dash import dcc, html

_BTN = {"padding": "2px 8px", "cursor": "pointer"}
_LBL = {"fontSize": "12px", "color": "#555"}
_INP = {"flex": "1", "fontSize": "12px"}
_PAD = {"padding": "6px"}
_STYLES = ["linear", "log"]


def build_partition_section() -> html.Div:
    """Return the full partition + save section for the sidebar."""
    return html.Div(children=[
        html.Hr(),
        html.H4("Partition", style={"marginBottom": "4px"}),
        dcc.Tabs(
            id="partition-tabs",
            value="q-phi",
            style={"fontSize": "12px"},
            children=[
                dcc.Tab(label="q-phi",    value="q-phi",    children=_tab_qphi()),
                dcc.Tab(label="xy-mesh",  value="xy-mesh",  children=_tab_xymesh()),
                dcc.Tab(label="General",  value="general",  children=_tab_general()),
                dcc.Tab(label="eq-ephi",  value="eq-ephi",  children=_tab_eqephi()),
            ],
        ),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "8px",
                   "marginTop": "6px", "marginBottom": "4px"},
            children=[
                html.Button("Compute", id="partition-compute-btn",
                            style={**_BTN, "flex": "1"}),
                html.Div(id="partition-status",
                         style={"fontSize": "11px", "color": "#555", "flex": "2"}),
            ],
        ),
        html.Hr(),
        html.H4("Save", style={"marginBottom": "4px"}),
        dcc.Input(
            id="save-path",
            type="text",
            placeholder="/path/to/output",
            style={"width": "100%", "boxSizing": "border-box",
                   "fontSize": "12px", "marginBottom": "4px"},
        ),
        html.Div(
            style={"display": "flex", "gap": "4px", "marginBottom": "4px"},
            children=[
                html.Button("Save Partition (HDF5)", id="save-partition-btn",
                            style={**_BTN, "flex": "1"}),
                html.Button("Save Mask (TIFF)", id="save-mask-btn",
                            style={**_BTN, "flex": "1"}),
            ],
        ),
        html.Div(id="save-status", style={"fontSize": "11px", "color": "#555"}),
    ])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _row(label: str, *children) -> html.Div:
    return html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "4px",
               "marginBottom": "3px"},
        children=[
            html.Label(label, style={**_LBL, "width": "110px", "flexShrink": 0}),
            *children,
        ],
    )


# ---------------------------------------------------------------------------
# Tab content functions
# ---------------------------------------------------------------------------


def _tab_qphi() -> list:
    return [html.Div(style=_PAD, children=[
        _row("Dynamic q bins",
             dcc.Input(id="qphi-dq-num", type="number", value=10, style=_INP)),
        _row("Dynamic φ bins",
             dcc.Input(id="qphi-dp-num", type="number", value=36, style=_INP)),
        _row("Static q bins",
             dcc.Input(id="qphi-sq-num", type="number", value=100, style=_INP)),
        _row("Static φ bins",
             dcc.Input(id="qphi-sp-num", type="number", value=360, style=_INP)),
        _row("Style",
             dcc.Dropdown(id="qphi-style", options=_STYLES, value="linear",
                          clearable=False, style={**_INP, "fontSize": "12px"})),
        _row("φ offset (°)",
             dcc.Input(id="qphi-phi-offset", type="number", value=0.0, style=_INP)),
        _row("Symmetry fold",
             dcc.Input(id="qphi-symmetry-fold", type="number", value=1,
                       min=1, max=12, style=_INP)),
    ])]


def _tab_xymesh() -> list:
    return [html.Div(style=_PAD, children=[
        _row("Dynamic x bins",
             dcc.Input(id="xy-dq-num", type="number", value=10, style=_INP)),
        _row("Dynamic y bins",
             dcc.Input(id="xy-dp-num", type="number", value=10, style=_INP)),
        _row("Static x bins",
             dcc.Input(id="xy-sq-num", type="number", value=100, style=_INP)),
        _row("Static y bins",
             dcc.Input(id="xy-sp-num", type="number", value=100, style=_INP)),
    ])]


def _tab_general() -> list:
    return [html.Div(style=_PAD, children=[
        html.Label("Axis 0", style=_LBL),
        dcc.Dropdown(id="gen-map0", options=[], placeholder="select axis",
                     clearable=False,
                     style={"fontSize": "12px", "marginBottom": "3px"}),
        _row("Dynamic bins",
             dcc.Input(id="gen-dq-num", type="number", value=10, style=_INP)),
        _row("Static bins",
             dcc.Input(id="gen-sq-num", type="number", value=100, style=_INP)),
        _row("Style",
             dcc.Dropdown(id="gen-style", options=_STYLES, value="linear",
                          clearable=False,
                          style={**_INP, "fontSize": "12px"})),
        html.Label("Axis 1", style={**_LBL, "marginTop": "6px"}),
        dcc.Dropdown(id="gen-map1", options=[], placeholder="select axis",
                     clearable=False,
                     style={"fontSize": "12px", "marginBottom": "3px"}),
        _row("Dynamic bins",
             dcc.Input(id="gen-dp-num", type="number", value=36, style=_INP)),
        _row("Static bins",
             dcc.Input(id="gen-sp-num", type="number", value=360, style=_INP)),
    ])]


def _tab_eqephi() -> list:
    return [html.Div(style=_PAD, children=[
        _row("Dynamic q bins",
             dcc.Input(id="eqephi-dq-num", type="number", value=10, style=_INP)),
        _row("Dynamic φ bins",
             dcc.Input(id="eqephi-dp-num", type="number", value=36, style=_INP)),
        _row("Static q bins",
             dcc.Input(id="eqephi-sq-num", type="number", value=100, style=_INP)),
        _row("Static φ bins",
             dcc.Input(id="eqephi-sp-num", type="number", value=360, style=_INP)),
    ])]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/web/test_partition_layout.py -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Ruff + full suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check src/pysimplemask/web/partition_layout.py tests/web/test_partition_layout.py
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
```

Expected: ruff clean; all existing tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/pysimplemask/web/partition_layout.py tests/web/test_partition_layout.py
git commit -m "feat(web): add 4-tab partition layout section"
```

---

## Task 2: Wire `layout.py` + `server.py`

**Files:**
- Modify: `src/pysimplemask/web/layout.py`
- Modify: `src/pysimplemask/web/server.py`

**Interfaces:**
- Consumes: `build_partition_section()` from Task 1
- Produces: partition section visible in sidebar; `partition_callbacks` import guarded in server

- [ ] **Step 1: Update `layout.py`**

Add import at top of `src/pysimplemask/web/layout.py` (after the existing mask_layout import):

```python
from pysimplemask.web.partition_layout import build_partition_section
```

At the end of `_sidebar()`'s `children` list (after the existing `*build_mask_section().children` line), append:

```python
            *build_partition_section().children,
```

The complete end of `_sidebar()` children becomes:

```python
        children=[
            # ... existing file/metadata/buttons/mask elements ...
            *build_mask_section().children,
            *build_partition_section().children,   # ← ADD THIS LINE
        ],
```

- [ ] **Step 2: Update `server.py`**

In `main_web()`, add a third guarded import block after the existing mask_callbacks block:

```python
    try:
        from pysimplemask.web import partition_callbacks as _partition_callbacks  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pysimplemask.web.partition_callbacks not found. "
            "Ensure the web package is fully installed."
        ) from exc
```

- [ ] **Step 3: Run full suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
```

Expected: all tests pass. (The guarded import in server.py won't break existing tests because the module level import of `server` only instantiates the Flask/Dash app — `main_web()` is not called by tests.)

- [ ] **Step 4: Commit**

```bash
git add src/pysimplemask/web/layout.py src/pysimplemask/web/server.py
git commit -m "feat(web): wire partition section into sidebar and server"
```

---

## Task 3: `partition_callbacks.py` — all callbacks

**Files:**
- Create: `src/pysimplemask/web/partition_callbacks.py`
- Modify: `tests/web/test_web_import.py`

**Interfaces:**
- Consumes: `model` from `server.py`; `make_figure` from `image_utils.py`
- All component IDs from Task 1
- `model.compute_partition(mode_str, **kwargs)` — core API
- `model.save_partition(path)` — writes HDF5
- `model.save_mask(path)` — writes TIFF
- `model.new_partition` — `None` until `compute_partition` has been called

- [ ] **Step 1: Add smoke test**

Append to `tests/web/test_web_import.py`:

```python
def test_partition_callbacks_imports():
    import pysimplemask.web.partition_callbacks  # noqa: F401
```

- [ ] **Step 2: Run failing test**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/web/test_web_import.py::test_partition_callbacks_imports -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `src/pysimplemask/web/partition_callbacks.py`**

```python
"""Partition section callbacks for the pySimpleMask web viewer."""

from __future__ import annotations

from dash import Input, Output, State, callback, no_update

from pysimplemask.web.image_utils import make_figure
from pysimplemask.web.server import model

# DISPLAY_FIELD index for dqmap_partition
_DQMAP = 3


def _fig(colormap: str | None, log_scale_list: list) -> object:
    arr = model.dset.data_display[_DQMAP]
    return make_figure(
        arr,
        colormap=colormap or "jet",
        log_scale=bool(log_scale_list),
        center_vh=model.get_center("vh"),
    )


# ---------------------------------------------------------------------------
# Compute (single callback dispatching on partition-tabs.value)
# ---------------------------------------------------------------------------


@callback(
    Output("detector-image", "figure",     allow_duplicate=True),
    Output("display-channel", "value",     allow_duplicate=True),
    Output("partition-status", "children", allow_duplicate=True),
    Input("partition-compute-btn", "n_clicks"),
    # which tab
    State("partition-tabs", "value"),
    # q-phi states
    State("qphi-dq-num",       "value"),
    State("qphi-dp-num",       "value"),
    State("qphi-sq-num",       "value"),
    State("qphi-sp-num",       "value"),
    State("qphi-style",        "value"),
    State("qphi-phi-offset",   "value"),
    State("qphi-symmetry-fold","value"),
    # xy-mesh states
    State("xy-dq-num", "value"),
    State("xy-dp-num", "value"),
    State("xy-sq-num", "value"),
    State("xy-sp-num", "value"),
    # general states
    State("gen-map0",   "value"),
    State("gen-map1",   "value"),
    State("gen-dq-num", "value"),
    State("gen-dp-num", "value"),
    State("gen-sq-num", "value"),
    State("gen-sp-num", "value"),
    State("gen-style",  "value"),
    # eq-ephi states
    State("eqephi-dq-num", "value"),
    State("eqephi-dp-num", "value"),
    State("eqephi-sq-num", "value"),
    State("eqephi-sp-num", "value"),
    # display
    State("colormap",   "value"),
    State("log-scale",  "value"),
    prevent_initial_call=True,
)
def compute_partition(
    n_clicks, tab,
    qphi_dq, qphi_dp, qphi_sq, qphi_sp, qphi_style, qphi_offset, qphi_fold,
    xy_dq, xy_dp, xy_sq, xy_sp,
    gen_map0, gen_map1, gen_dq, gen_dp, gen_sq, gen_sp, gen_style,
    eq_dq, eq_dp, eq_sq, eq_sp,
    colormap, log_scale_list,
):
    """Compute partition for the active tab and display dqmap overlay."""
    if not model.is_ready():
        return no_update, no_update, "Load a file first."

    try:
        if tab == "q-phi":
            model.compute_partition(
                "q-phi",
                dq_num=int(qphi_dq or 10),
                sq_num=int(qphi_sq or 100),
                dp_num=int(qphi_dp or 36),
                sp_num=int(qphi_sp or 360),
                style=qphi_style or "linear",
                phi_offset=float(qphi_offset or 0.0),
                symmetry_fold=int(qphi_fold or 1),
            )
        elif tab == "xy-mesh":
            model.compute_partition(
                "x-y",
                dq_num=int(xy_dq or 10),
                sq_num=int(xy_sq or 100),
                dp_num=int(xy_dp or 10),
                sp_num=int(xy_sp or 100),
            )
        elif tab == "general":
            if not gen_map0 or not gen_map1:
                return no_update, no_update, "Select both axes first."
            mode_str = f"{gen_map0}-{gen_map1}"
            model.compute_partition(
                mode_str,
                dq_num=int(gen_dq or 10),
                sq_num=int(gen_sq or 100),
                dp_num=int(gen_dp or 36),
                sp_num=int(gen_sp or 360),
                style=gen_style or "linear",
            )
        elif tab == "eq-ephi":
            model.compute_partition(
                "eq-ephi",
                dq_num=int(eq_dq or 10),
                sq_num=int(eq_sq or 100),
                dp_num=int(eq_dp or 36),
                sp_num=int(eq_sp or 360),
            )
        else:
            return no_update, no_update, f"Unknown tab: {tab}"
    except Exception as exc:
        return no_update, no_update, f"Error: {exc}"

    return _fig(colormap, log_scale_list), _DQMAP, "Partition computed."


# ---------------------------------------------------------------------------
# Save Partition (HDF5) — primary owner of save-status
# ---------------------------------------------------------------------------


@callback(
    Output("save-status", "children"),
    Input("save-partition-btn", "n_clicks"),
    State("save-path", "value"),
    prevent_initial_call=True,
)
def save_partition_cb(n_clicks, save_path):
    if not model.is_ready() or model.new_partition is None:
        return "Compute a partition first."
    if not save_path:
        return "Enter an output path."
    try:
        model.save_partition(save_path)
    except Exception as exc:
        return f"Save error: {exc}"
    return f"Partition saved to {save_path}"


# ---------------------------------------------------------------------------
# Save Mask (TIFF) — secondary output on save-status
# ---------------------------------------------------------------------------


@callback(
    Output("save-status", "children", allow_duplicate=True),
    Input("save-mask-btn", "n_clicks"),
    State("save-path", "value"),
    prevent_initial_call=True,
)
def save_mask_cb(n_clicks, save_path):
    if not model.is_ready():
        return "Load a file first."
    if not save_path:
        return "Enter an output path."
    mask_path = (
        save_path
        if save_path.endswith((".tif", ".tiff"))
        else save_path + ".tif"
    )
    try:
        model.save_mask(mask_path)
    except Exception as exc:
        return f"Save error: {exc}"
    return f"Mask saved to {mask_path}"


# ---------------------------------------------------------------------------
# General tab — repopulate axis dropdowns when model loads
# ---------------------------------------------------------------------------


@callback(
    Output("gen-map0", "options"),
    Output("gen-map1", "options"),
    Input("model-loaded", "data"),
    prevent_initial_call=False,
)
def repopulate_gen_axes(_model_loaded):
    if not model.is_ready():
        return [], []
    opts = [{"label": k, "value": k} for k in model.qmap.keys()]
    return opts, opts
```

- [ ] **Step 4: Run smoke test**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/web/test_web_import.py::test_partition_callbacks_imports -v
```

Expected: PASS.

- [ ] **Step 5: Run full suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 6: Ruff check**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check src/pysimplemask/web/partition_callbacks.py tests/web/test_web_import.py
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add src/pysimplemask/web/partition_callbacks.py tests/web/test_web_import.py
git commit -m "feat(web): add partition compute, save, and axis-repopulate callbacks"
```
