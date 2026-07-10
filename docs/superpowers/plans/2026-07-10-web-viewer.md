# Web Interface — Core Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pysimplemask-web` CLI command that launches a Dash/Flask web server where users can load a scattering file, view the detector image, and edit beam-center metadata in a browser.

**Architecture:** `src/pysimplemask/web/` is a new package that wires a module-level `SimpleMaskModel` singleton to a Plotly Dash app served by Flask. `image_utils.make_figure` converts numpy arrays to Plotly figures. Layout and callbacks are in separate files. No changes to `core/` or `gui/`.

**Tech Stack:** Python, Plotly Dash ≥2.14, Flask (bundled with Dash), NumPy, existing `pysimplemask.core`

## Global Constraints

- Environment: `/local/MQICHU/envs/l2606_simplemask_refact/bin/`
- Run tests with: `/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest`
- `core/` and `gui/` must not be modified
- All web tests use `pytest.importorskip("dash")` — skipped when Dash is absent
- Optional dep group name: `web`; packages: `dash>=2.14`, `plotly>=5.18`
- CLI script name: `pysimplemask-web` → `pysimplemask.web.server:main_web`
- Module-level singleton in `server.py`: `model = SimpleMaskModel()`
- Component IDs must match exactly between `layout.py` and `callbacks.py` (see tables below)
- `find_center` is called with `beamstop_diameter=0` (no side-effect masking in viewer sub-project)
- After `find_center()` or `goto_max()`, `model.dset.set_center_vh()` and `model.update_parameters()` must be called explicitly (the model methods do not do this internally for find_center)
- Ruff must pass clean on all new files

---

## File Map

| Status | File | Responsibility |
|--------|------|----------------|
| Create | `src/pysimplemask/web/__init__.py` | Empty package marker |
| Create | `src/pysimplemask/web/image_utils.py` | `make_figure(arr, colormap, log_scale, center_vh)` |
| Create | `src/pysimplemask/web/server.py` | Flask app, Dash app, model singleton, `main_web()` |
| Create | `src/pysimplemask/web/layout.py` | `build_layout(initial_path)` → `html.Div` |
| Create | `src/pysimplemask/web/callbacks.py` | All 6 `@callback` functions |
| Modify | `pyproject.toml` | Add `[web]` optional deps + `pysimplemask-web` script |
| Create | `tests/web/__init__.py` | Empty package marker |
| Create | `tests/web/test_image_utils.py` | Unit tests for `make_figure` |
| Create | `tests/web/test_web_import.py` | Smoke tests for imports + layout |

---

## Component ID Reference (layout.py ↔ callbacks.py)

| Widget | ID | Type |
|--------|----|------|
| File path input | `file-path` | `dcc.Input` |
| Beamline dropdown | `beamline` | `dcc.Dropdown` |
| Begin index input | `begin-idx` | `dcc.Input` |
| Num frames input | `num-frames` | `dcc.Input` |
| Load button | `load-btn` | `html.Button` |
| Status message | `status-msg` | `html.Div` |
| Beam center X | `meta-beam_center_x` | `dcc.Input` |
| Beam center Y | `meta-beam_center_y` | `dcc.Input` |
| Energy | `meta-energy` | `dcc.Input` |
| Detector distance | `meta-detector_distance` | `dcc.Input` |
| Pixel size | `meta-pixel_size` | `dcc.Input` |
| Find Center button | `find-center-btn` | `html.Button` |
| Goto Max button | `goto-max-btn` | `html.Button` |
| Swap X/Y button | `swap-xy-btn` | `html.Button` |
| Update Parameters button | `update-params-btn` | `html.Button` |
| Display channel dropdown | `display-channel` | `dcc.Dropdown` |
| Colormap dropdown | `colormap` | `dcc.Dropdown` |
| Log scale checklist | `log-scale` | `dcc.Checklist` |
| Detector image graph | `detector-image` | `dcc.Graph` |
| Hover status bar | `hover-status` | `html.Div` |

---

## Task 1: `image_utils.py` + package scaffold + unit tests

**Files:**
- Create: `src/pysimplemask/web/__init__.py`
- Create: `src/pysimplemask/web/image_utils.py`
- Create: `tests/web/__init__.py`
- Create: `tests/web/test_image_utils.py`

**Interfaces:**
- Produces: `make_figure(arr, colormap="jet", log_scale=False, center_vh=None) -> go.Figure`
  - `arr`: 2-D `np.ndarray` (H, W), any float dtype
  - `colormap`: Plotly color scale name string
  - `log_scale`: if True, apply `log10` with positive floor before rendering
  - `center_vh`: `(row, col)` tuple or `None`; adds a white crosshair scatter trace if given
  - Returns a `plotly.graph_objects.Figure`

- [ ] **Step 1: Install Dash into the project environment**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pip install "dash>=2.14" "plotly>=5.18"
```

Expected: `Successfully installed dash-X.Y.Z plotly-X.Y.Z` (versions may vary).

- [ ] **Step 2: Create the empty package markers**

Create `src/pysimplemask/web/__init__.py` (empty file):
```python
```

Create `tests/web/__init__.py` (empty file):
```python
```

- [ ] **Step 3: Write the failing unit tests**

Create `tests/web/test_image_utils.py`:

```python
"""Unit tests for image_utils.make_figure."""

import numpy as np
import pytest

dash = pytest.importorskip("dash")
import plotly.graph_objects as go  # noqa: E402

from pysimplemask.web.image_utils import make_figure  # noqa: E402


def test_make_figure_returns_go_figure():
    arr = np.ones((16, 16), dtype=np.float32) * 5.0
    fig = make_figure(arr)
    assert isinstance(fig, go.Figure)


def test_make_figure_has_one_trace_without_crosshair():
    arr = np.ones((16, 16), dtype=np.float32)
    fig = make_figure(arr, center_vh=None)
    assert len(fig.data) == 1


def test_make_figure_has_two_traces_with_crosshair():
    arr = np.ones((64, 64), dtype=np.float32)
    fig = make_figure(arr, center_vh=(32, 32))
    assert len(fig.data) == 2
    assert fig.data[1].type == "scatter"


def test_make_figure_log_scale_applies_log10():
    arr = np.array([[1.0, 10.0, 100.0], [0.1, 0.01, 1000.0]], dtype=np.float64)
    fig = make_figure(arr, log_scale=True)
    z = fig.data[0].z
    assert z is not None
    np.testing.assert_allclose(z[0][1], np.log10(10.0), rtol=1e-5)
    np.testing.assert_allclose(z[0][2], np.log10(100.0), rtol=1e-5)


def test_make_figure_log_scale_handles_zeros():
    arr = np.array([[0.0, 5.0], [2.0, 0.0]], dtype=np.float32)
    fig = make_figure(arr, log_scale=True)
    assert isinstance(fig, go.Figure)
    z = fig.data[0].z
    # zeros are replaced by floor (min positive value = 2.0)
    floor = 2.0
    np.testing.assert_allclose(z[0][0], np.log10(floor), rtol=1e-4)


def test_make_figure_colormap_accepted():
    arr = np.ones((8, 8), dtype=np.float32)
    fig = make_figure(arr, colormap="viridis")
    assert isinstance(fig, go.Figure)
```

- [ ] **Step 4: Run the failing tests**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/web/test_image_utils.py -v
```

Expected: all 6 tests FAIL with `ModuleNotFoundError: No module named 'pysimplemask.web.image_utils'`.

- [ ] **Step 5: Create `src/pysimplemask/web/image_utils.py`**

```python
"""Array-to-Plotly-figure utilities for the pySimpleMask web viewer."""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def make_figure(
    arr: np.ndarray,
    colormap: str = "jet",
    log_scale: bool = False,
    center_vh: tuple[int, int] | None = None,
) -> go.Figure:
    """Convert a 2-D numpy array to a Plotly figure for ``dcc.Graph``.

    Args:
        arr: 2-D array of shape (H, W).
        colormap: Plotly color scale name (e.g. ``"jet"``, ``"viridis"``).
        log_scale: Apply ``log10`` before rendering.  Zeros are replaced by
            the minimum positive value in the array before taking the log.
        center_vh: Beam center as ``(row, col)``.  Draws a white crosshair
            scatter trace on top of the image when provided.

    Returns:
        :class:`plotly.graph_objects.Figure` ready for ``dcc.Graph``.
    """
    display = arr.astype(np.float64)
    if log_scale:
        positive = display[display > 0]
        floor = float(positive.min()) if positive.size else 1.0
        display = np.log10(np.maximum(display, floor))

    fig = px.imshow(
        display,
        color_continuous_scale=colormap,
        origin="upper",
        aspect="equal",
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        coloraxis_colorbar={"thickness": 12, "len": 0.8},
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    if center_vh is not None:
        row, col = center_vh
        arm = max(arr.shape) * 0.02  # crosshair arm = 2% of image dimension
        fig.add_trace(
            go.Scatter(
                x=[col - arm, col + arm, None, col, col],
                y=[row, row, None, row - arm, row + arm],
                mode="lines",
                line={"color": "white", "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    return fig
```

- [ ] **Step 6: Run the tests to verify they pass**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/web/test_image_utils.py -v
```

Expected:
```
PASSED test_make_figure_returns_go_figure
PASSED test_make_figure_has_one_trace_without_crosshair
PASSED test_make_figure_has_two_traces_with_crosshair
PASSED test_make_figure_log_scale_applies_log10
PASSED test_make_figure_log_scale_handles_zeros
PASSED test_make_figure_colormap_accepted
6 passed
```

- [ ] **Step 7: Ruff check**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check \
    src/pysimplemask/web/image_utils.py \
    tests/web/test_image_utils.py
```

Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add \
    src/pysimplemask/web/__init__.py \
    src/pysimplemask/web/image_utils.py \
    tests/web/__init__.py \
    tests/web/test_image_utils.py
git commit -m "feat(web): add image_utils.make_figure for detector image rendering"
```

---

## Task 2: `server.py` + `layout.py` + `pyproject.toml` + smoke tests

**Files:**
- Create: `src/pysimplemask/web/server.py`
- Create: `src/pysimplemask/web/layout.py`
- Modify: `pyproject.toml`
- Create: `tests/web/test_web_import.py`

**Interfaces:**
- Consumes: `make_figure` from Task 1 (not yet called here — imported in callbacks)
- Produces: `server` — `flask.Flask` instance (module-level in `server.py`)
- Produces: `app` — `dash.Dash` instance (module-level in `server.py`)
- Produces: `model` — `SimpleMaskModel` instance (module-level in `server.py`)
- Produces: `main_web(host, port, path, debug)` → `None` (console entry point)
- Produces: `build_layout(initial_path="") -> html.Div` (in `layout.py`)

- [ ] **Step 1: Write the failing smoke tests**

Create `tests/web/test_web_import.py`:

```python
"""Smoke tests — verify web package imports without error."""

import pytest

dash = pytest.importorskip("dash")
import flask  # noqa: E402
import dash as dash_mod  # noqa: E402


def test_server_is_flask_app():
    from pysimplemask.web.server import server
    assert isinstance(server, flask.Flask)


def test_app_is_dash_app():
    from pysimplemask.web.server import app
    assert isinstance(app, dash_mod.Dash)


def test_model_is_simplemaskmodel():
    from pysimplemask.web.server import model
    from pysimplemask.core.model import SimpleMaskModel
    assert isinstance(model, SimpleMaskModel)


def test_main_web_is_callable():
    from pysimplemask.web.server import main_web
    assert callable(main_web)


def test_build_layout_returns_html_div():
    from pysimplemask.web.layout import build_layout
    from dash import html
    layout = build_layout(initial_path="/test/path.h5")
    assert isinstance(layout, html.Div)


def test_build_layout_file_path_pre_populated():
    from pysimplemask.web.layout import build_layout
    layout = build_layout(initial_path="/data/scan.h5")
    # Recursively search for the file-path Input component
    def find_value(component, target_id):
        if hasattr(component, "id") and component.id == target_id:
            return component.value
        for child in getattr(component, "children", []) or []:
            if isinstance(child, list):
                for c in child:
                    result = find_value(c, target_id)
                    if result is not None:
                        return result
            result = find_value(child, target_id)
            if result is not None:
                return result
        return None

    value = find_value(layout, "file-path")
    assert value == "/data/scan.h5"
```

- [ ] **Step 2: Run the failing tests**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/web/test_web_import.py -v
```

Expected: all 6 tests FAIL with `ModuleNotFoundError: No module named 'pysimplemask.web.server'`.

- [ ] **Step 3: Create `src/pysimplemask/web/server.py`**

```python
"""Flask + Dash application and module-level model singleton."""

from __future__ import annotations

import argparse

import dash
import flask

from pysimplemask.core.model import SimpleMaskModel

# Module-level singletons — single-user, single-process.
server = flask.Flask(__name__)
app = dash.Dash(__name__, server=server, title="pySimpleMask")
model = SimpleMaskModel()


def main_web() -> None:
    """Console-script entry point: ``pysimplemask-web``."""
    parser = argparse.ArgumentParser(
        prog="pysimplemask-web",
        description="Launch the pySimpleMask web interface.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Port number (default: 8050)",
    )
    parser.add_argument(
        "--path", default=None,
        help="Pre-populate the file-path field with this path",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Dash debug/hot-reload mode",
    )
    args = parser.parse_args()

    from pysimplemask.web import layout as _layout  # noqa: F401
    from pysimplemask.web import callbacks as _callbacks  # noqa: F401

    app.layout = _layout.build_layout(initial_path=args.path or "")

    print(f"pySimpleMask web interface running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
```

- [ ] **Step 4: Create `src/pysimplemask/web/layout.py`**

```python
"""Dash component tree for the pySimpleMask web viewer."""

from __future__ import annotations

from dash import dcc, html

from pysimplemask.core.reader.base_reader import DISPLAY_FIELD

COLORMAPS = ["jet", "viridis", "gray", "hot", "cool", "RdBu_r", "plasma", "magma"]
BEAMLINES = ["APS_8IDI", "APS_9IDD", "NativeFiles"]


def build_layout(initial_path: str = "") -> html.Div:
    """Return the root Dash layout component tree."""
    return html.Div(
        style={
            "display": "flex",
            "height": "100vh",
            "fontFamily": "sans-serif",
            "fontSize": "14px",
        },
        children=[
            _sidebar(initial_path),
            _main_panel(),
        ],
    )


def _sidebar(initial_path: str) -> html.Div:
    return html.Div(
        style={
            "width": "30%",
            "overflowY": "auto",
            "padding": "12px",
            "borderRight": "1px solid #ccc",
            "boxSizing": "border-box",
        },
        children=[
            html.H4("File", style={"marginTop": 0}),
            dcc.Input(
                id="file-path",
                value=initial_path,
                type="text",
                placeholder="/path/to/data.h5",
                style={"width": "100%", "marginBottom": "6px", "boxSizing": "border-box"},
            ),
            dcc.Dropdown(
                id="beamline",
                options=BEAMLINES,
                value="APS_8IDI",
                clearable=False,
                style={"marginBottom": "6px"},
            ),
            html.Div(
                style={"display": "flex", "gap": "6px", "marginBottom": "6px"},
                children=[
                    dcc.Input(
                        id="begin-idx",
                        type="number",
                        value=0,
                        placeholder="begin_idx",
                        style={"width": "50%"},
                    ),
                    dcc.Input(
                        id="num-frames",
                        type="number",
                        value=-1,
                        placeholder="num_frames",
                        style={"width": "50%"},
                    ),
                ],
            ),
            html.Button(
                "Load",
                id="load-btn",
                style={"width": "100%", "marginBottom": "8px"},
            ),
            html.Div(
                id="status-msg",
                style={"color": "red", "fontSize": "12px", "marginBottom": "8px"},
            ),
            html.Hr(),
            html.H4("Metadata", style={"marginBottom": "6px"}),
            *_meta_row("beam_center_x", "Beam center X", "px"),
            *_meta_row("beam_center_y", "Beam center Y", "px"),
            *_meta_row("energy", "Energy", "keV"),
            *_meta_row("detector_distance", "Distance", "m"),
            *_meta_row("pixel_size", "Pixel size", "m"),
            html.Div(
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "4px",
                    "marginTop": "8px",
                },
                children=[
                    html.Button("Find Center", id="find-center-btn", style={"flex": "1"}),
                    html.Button("Goto Max", id="goto-max-btn", style={"flex": "1"}),
                    html.Button("Swap X/Y", id="swap-xy-btn", style={"flex": "1"}),
                    html.Button("Update Params", id="update-params-btn", style={"flex": "1"}),
                ],
            ),
        ],
    )


def _main_panel() -> html.Div:
    return html.Div(
        style={
            "width": "70%",
            "display": "flex",
            "flexDirection": "column",
            "padding": "12px",
            "boxSizing": "border-box",
        },
        children=[
            html.Div(
                style={"display": "flex", "gap": "8px", "marginBottom": "8px",
                       "alignItems": "center"},
                children=[
                    dcc.Dropdown(
                        id="display-channel",
                        options=[
                            {"label": v, "value": i}
                            for i, v in enumerate(DISPLAY_FIELD)
                        ],
                        value=0,
                        clearable=False,
                        style={"flex": "2"},
                    ),
                    dcc.Dropdown(
                        id="colormap",
                        options=COLORMAPS,
                        value="jet",
                        clearable=False,
                        style={"flex": "1"},
                    ),
                    dcc.Checklist(
                        id="log-scale",
                        options=[{"label": " Log scale", "value": "log"}],
                        value=[],
                        style={"whiteSpace": "nowrap"},
                    ),
                ],
            ),
            dcc.Graph(
                id="detector-image",
                style={"flex": "1"},
                config={"scrollZoom": True, "displaylogo": False},
            ),
            html.Div(
                id="hover-status",
                style={
                    "fontFamily": "monospace",
                    "fontSize": "12px",
                    "marginTop": "4px",
                    "height": "18px",
                },
            ),
        ],
    )


def _meta_row(field_id: str, label: str, unit: str) -> list:
    return [
        html.Div(
            style={
                "display": "flex",
                "alignItems": "center",
                "marginBottom": "4px",
                "gap": "6px",
            },
            children=[
                html.Label(
                    label,
                    style={"width": "120px", "fontSize": "13px", "flexShrink": 0},
                ),
                dcc.Input(
                    id=f"meta-{field_id}",
                    type="number",
                    debounce=True,
                    style={"flex": "1"},
                ),
                html.Span(
                    unit,
                    style={"fontSize": "12px", "color": "#666", "flexShrink": 0},
                ),
            ],
        )
    ]
```

- [ ] **Step 5: Update `pyproject.toml`**

Add the `web` optional-dependency group and new script. In `pyproject.toml`:

Replace:
```toml
[project.optional-dependencies]
dev = [
  "coverage", # Testing
  "mypy",     # Type checking
  "pytest",   # Testing
  "ruff",     # Linting
]
```

With:
```toml
[project.optional-dependencies]
dev = [
  "coverage", # Testing
  "mypy",     # Type checking
  "pytest",   # Testing
  "ruff",     # Linting
]
web = [
  "dash>=2.14",
  "plotly>=5.18",
]
```

And replace:
```toml
[project.scripts]
pysimplemask = "pysimplemask.cli:main"
pysimplemask-combine-qmaps = "pysimplemask.cli:combine_qmaps"
pysimplemask-build-qmap = "pysimplemask.cli:build_qmap"
```

With:
```toml
[project.scripts]
pysimplemask = "pysimplemask.cli:main"
pysimplemask-combine-qmaps = "pysimplemask.cli:combine_qmaps"
pysimplemask-build-qmap = "pysimplemask.cli:build_qmap"
pysimplemask-web = "pysimplemask.web.server:main_web"
```

Then reinstall to register the new script:
```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pip install -e ".[dev,web]"
```

- [ ] **Step 6: Run the smoke tests**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/web/test_web_import.py -v
```

Expected:
```
PASSED test_server_is_flask_app
PASSED test_app_is_dash_app
PASSED test_model_is_simplemaskmodel
PASSED test_main_web_is_callable
PASSED test_build_layout_returns_html_div
PASSED test_build_layout_file_path_pre_populated
6 passed
```

- [ ] **Step 7: Verify `pysimplemask-web` script is installed**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pysimplemask-web --help
```

Expected:
```
usage: pysimplemask-web [-h] [--host HOST] [--port PORT] [--path PATH] [--debug]
...
```

- [ ] **Step 8: Ruff check**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check \
    src/pysimplemask/web/server.py \
    src/pysimplemask/web/layout.py \
    tests/web/test_web_import.py
```

Expected: no output.

- [ ] **Step 9: Run full test suite to confirm no regressions**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
```

Expected: all previous tests pass, plus the 12 new web tests.

- [ ] **Step 10: Commit**

```bash
git add \
    src/pysimplemask/web/server.py \
    src/pysimplemask/web/layout.py \
    pyproject.toml \
    tests/web/test_web_import.py
git commit -m "feat(web): add Flask+Dash server, layout, and CLI entry point"
```

---

## Task 3: `callbacks.py` + smoke test + launch verification

**Files:**
- Create: `src/pysimplemask/web/callbacks.py`
- Modify: `tests/web/test_web_import.py` (add one test)

**Interfaces:**
- Consumes: `model` from `server.py` (Task 2)
- Consumes: `make_figure` from `image_utils.py` (Task 1)
- Consumes: `build_layout` from `layout.py` (Task 2) — IDs must match exactly
- Consumes: `DISPLAY_FIELD` from `pysimplemask.core.reader.base_reader`
- Produces: 6 registered Dash callbacks (no public API; side-effect of import)

**Callback → model API mapping:**

| Callback | Model calls |
|----------|------------|
| `load_file` | `model.read_data(fname, beamline, begin_idx, num_frames)` |
| `update_params` | `model.update_parameters({...})` |
| `find_center` | `model.find_center(beamstop_diameter=0)` → `model.dset.set_center_vh(center)` → `model.update_parameters()` |
| `goto_max` | `model.goto_max()` → `model.update_parameters()` |
| `swap_xy` | `model.dset.swapxy()` → `model.update_parameters()` |
| `update_display` | reads `model.dset.data_display[idx]`, calls `make_figure` |
| `update_hover` | `model.dset.get_coordinates(col, row, idx)` |

- [ ] **Step 1: Add the callbacks smoke test**

Append to `tests/web/test_web_import.py`:

```python
def test_callbacks_module_imports():
    # Importing callbacks registers all @callback decorators as a side effect.
    # This test ensures no NameError or import cycle at import time.
    import pysimplemask.web.callbacks  # noqa: F401
```

- [ ] **Step 2: Run the failing test**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/web/test_web_import.py::test_callbacks_module_imports -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'pysimplemask.web.callbacks'`.

- [ ] **Step 3: Create `src/pysimplemask/web/callbacks.py`**

```python
"""Dash callbacks for the pySimpleMask web viewer."""

from __future__ import annotations

from dash import Input, Output, Patch, callback, ctx, no_update

from pysimplemask.core.reader.base_reader import DISPLAY_FIELD
from pysimplemask.web.image_utils import make_figure
from pysimplemask.web.server import model


# ---------------------------------------------------------------------------
# Load file
# ---------------------------------------------------------------------------


@callback(
    Output("detector-image", "figure"),
    Output("meta-beam_center_x", "value"),
    Output("meta-beam_center_y", "value"),
    Output("meta-energy", "value"),
    Output("meta-detector_distance", "value"),
    Output("meta-pixel_size", "value"),
    Output("status-msg", "children"),
    Output("display-channel", "options"),
    Input("load-btn", "n_clicks"),
    Input("file-path", "value"),
    Input("beamline", "value"),
    Input("begin-idx", "value"),
    Input("num-frames", "value"),
    prevent_initial_call=True,
)
def load_file(n_clicks, file_path, beamline, begin_idx, num_frames):
    """Load a scattering file and populate the metadata fields."""
    if ctx.triggered_id != "load-btn":
        return (no_update,) * 8
    if not file_path:
        return no_update, no_update, no_update, no_update, no_update, no_update, \
               "Enter a file path.", no_update
    try:
        ok = model.read_data(
            fname=file_path,
            beamline=beamline or "APS_8IDI",
            begin_idx=int(begin_idx or 0),
            num_frames=int(num_frames if num_frames is not None else -1),
        )
    except Exception as exc:
        return no_update, no_update, no_update, no_update, no_update, no_update, \
               f"Error: {exc}", no_update
    if not ok:
        return no_update, no_update, no_update, no_update, no_update, no_update, \
               "Failed to load file.", no_update

    meta = model.dset.metadata
    arr = model.dset.data_display[0]
    center_vh = model.get_center("vh")
    fig = make_figure(arr, log_scale=False, center_vh=center_vh)

    all_channel_labels = list(DISPLAY_FIELD) + list(model.qmap.keys())
    options = [{"label": v, "value": i} for i, v in enumerate(all_channel_labels)]

    return (
        fig,
        round(float(meta["beam_center_x"]), 4),
        round(float(meta["beam_center_y"]), 4),
        round(float(meta["energy"]), 6),
        round(float(meta["detector_distance"]), 6),
        float(meta["pixel_size"]),
        f"Loaded: {file_path}",
        options,
    )


# ---------------------------------------------------------------------------
# Update parameters
# ---------------------------------------------------------------------------


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Output("status-msg", "children", allow_duplicate=True),
    Input("update-params-btn", "n_clicks"),
    Input("meta-beam_center_x", "value"),
    Input("meta-beam_center_y", "value"),
    Input("meta-energy", "value"),
    Input("meta-detector_distance", "value"),
    Input("meta-pixel_size", "value"),
    prevent_initial_call=True,
)
def update_params(n_clicks, bcx, bcy, energy, distance, pixel_size):
    """Recompute qmap from edited metadata fields and refresh the image."""
    if ctx.triggered_id != "update-params-btn":
        return no_update, no_update
    if not model.is_ready():
        return no_update, "Load a file first."
    new_meta: dict[str, float] = {}
    if bcx is not None:
        new_meta["beam_center_x"] = float(bcx)
    if bcy is not None:
        new_meta["beam_center_y"] = float(bcy)
    if energy is not None:
        new_meta["energy"] = float(energy)
    if distance is not None:
        new_meta["detector_distance"] = float(distance)
    if pixel_size is not None:
        new_meta["pixel_size"] = float(pixel_size)
    model.update_parameters(new_meta)
    arr = model.dset.data_display[0]
    center_vh = model.get_center("vh")
    fig = make_figure(arr, center_vh=center_vh)
    return fig, "Parameters updated."


# ---------------------------------------------------------------------------
# Find Center
# ---------------------------------------------------------------------------


@callback(
    Output("meta-beam_center_x", "value", allow_duplicate=True),
    Output("meta-beam_center_y", "value", allow_duplicate=True),
    Output("status-msg", "children", allow_duplicate=True),
    Input("find-center-btn", "n_clicks"),
    prevent_initial_call=True,
)
def find_center_cb(n_clicks):
    """Run the cross-correlation beam-center finder and update metadata fields."""
    if not model.is_ready():
        return no_update, no_update, "Load a file first."
    # beamstop_diameter=0: skip the beamstop masking side-effect
    center_vh = model.find_center(beamstop_diameter=0)
    if center_vh is None:
        return no_update, no_update, "Find center failed."
    model.dset.set_center_vh(center_vh)
    model.update_parameters()
    meta = model.dset.metadata
    return (
        round(float(meta["beam_center_x"]), 4),
        round(float(meta["beam_center_y"]), 4),
        "Center found — click Update Params to apply.",
    )


# ---------------------------------------------------------------------------
# Goto Max
# ---------------------------------------------------------------------------


@callback(
    Output("meta-beam_center_x", "value", allow_duplicate=True),
    Output("meta-beam_center_y", "value", allow_duplicate=True),
    Output("status-msg", "children", allow_duplicate=True),
    Input("goto-max-btn", "n_clicks"),
    prevent_initial_call=True,
)
def goto_max_cb(n_clicks):
    """Move beam center to the pixel of maximum intensity."""
    if not model.is_ready():
        return no_update, no_update, "Load a file first."
    model.goto_max()          # sets center_vh in metadata internally
    model.update_parameters() # recomputes qmap with new center
    meta = model.dset.metadata
    return (
        round(float(meta["beam_center_x"]), 4),
        round(float(meta["beam_center_y"]), 4),
        "Moved to max — click Update Params to apply.",
    )


# ---------------------------------------------------------------------------
# Swap X/Y
# ---------------------------------------------------------------------------


@callback(
    Output("meta-beam_center_x", "value", allow_duplicate=True),
    Output("meta-beam_center_y", "value", allow_duplicate=True),
    Output("status-msg", "children", allow_duplicate=True),
    Input("swap-xy-btn", "n_clicks"),
    prevent_initial_call=True,
)
def swap_xy_cb(n_clicks):
    """Swap beam_center_x and beam_center_y in the metadata."""
    if not model.is_ready():
        return no_update, no_update, "Load a file first."
    model.dset.swapxy()
    model.update_parameters()
    meta = model.dset.metadata
    return (
        round(float(meta["beam_center_x"]), 4),
        round(float(meta["beam_center_y"]), 4),
        "X/Y swapped — click Update Params to apply.",
    )


# ---------------------------------------------------------------------------
# Display controls (channel / colormap / log-scale)
# ---------------------------------------------------------------------------


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Input("display-channel", "value"),
    Input("colormap", "value"),
    Input("log-scale", "value"),
    prevent_initial_call=True,
)
def update_display(channel_idx, colormap, log_scale_list):
    """Re-render the detector image when display controls change."""
    if not model.is_ready():
        return no_update
    colormap = colormap or "jet"
    log_scale = bool(log_scale_list)
    idx = int(channel_idx) if channel_idx is not None else 0

    # Colormap-only change: patch the colorscale without re-serialising the array.
    if ctx.triggered_id == "colormap":
        patch = Patch()
        patch["data"][0]["colorscale"] = colormap
        return patch

    arr = model.dset.data_display[idx]
    center_vh = model.get_center("vh")
    return make_figure(arr, colormap=colormap, log_scale=log_scale, center_vh=center_vh)


# ---------------------------------------------------------------------------
# Hover status bar
# ---------------------------------------------------------------------------


@callback(
    Output("hover-status", "children"),
    Input("detector-image", "hoverData"),
    Input("display-channel", "value"),
    prevent_initial_call=True,
)
def update_hover(hover_data, channel_idx):
    """Show pixel coordinates and data value in the status bar."""
    if hover_data is None or not model.is_ready():
        return ""
    pt = hover_data["points"][0]
    col = int(pt.get("x", 0))
    row = int(pt.get("y", 0))
    idx = int(channel_idx) if channel_idx is not None else 0
    return model.dset.get_coordinates(col, row, idx) or ""
```

- [ ] **Step 4: Run the smoke test**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/web/test_web_import.py::test_callbacks_module_imports -v
```

Expected: PASS.

- [ ] **Step 5: Run the full web test suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/web/ -v
```

Expected: all 13 web tests pass.

- [ ] **Step 6: Run the complete test suite for regressions**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
```

Expected: all tests pass (116 previous + 13 new = 129 total).

- [ ] **Step 7: Ruff check**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check \
    src/pysimplemask/web/callbacks.py \
    tests/web/test_web_import.py
```

Expected: no output.

- [ ] **Step 8: Manual launch verification**

In one terminal, start the server:
```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pysimplemask-web --debug
```

Expected console output:
```
pySimpleMask web interface running at http://127.0.0.1:8050
Dash is running on http://127.0.0.1:8050/
```

Open a browser to `http://127.0.0.1:8050`. Verify:
- [ ] Left sidebar shows File section with path input, beamline dropdown, frame inputs, Load button
- [ ] Metadata section shows 5 editable fields (beam center X/Y, energy, distance, pixel size)
- [ ] Four action buttons visible (Find Center, Goto Max, Swap X/Y, Update Params)
- [ ] Main panel shows display-channel dropdown, colormap dropdown, log-scale checkbox
- [ ] Loading a real file (e.g. type path and click Load) shows the detector image

Stop the server with Ctrl-C.

- [ ] **Step 9: Commit**

```bash
git add \
    src/pysimplemask/web/callbacks.py \
    tests/web/test_web_import.py
git commit -m "feat(web): add Dash callbacks for load, metadata, display, and hover"
```
