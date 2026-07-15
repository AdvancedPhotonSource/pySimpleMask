# Download Partition Button Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Download Partition (HDF5)" button to the web Save section that sends the computed qmap file to the user's browser.

**Architecture:** `dcc.Download` (invisible Dash component) + button trigger. The callback writes the partition to a temp file with `model.save_partition()`, calls `dcc.send_file()` to encode it, then cleans up. No server-side path is required from the user.

**Tech Stack:** Plotly Dash 4.x (`dcc.Download`, `dcc.send_file`), Python `tempfile`, existing `model.save_partition()`

## Global Constraints

- Environment: `/local/MQICHU/envs/l2606_simplemask_refact/bin/`
- Button ID: `download-partition-btn`; Download component ID: `download-partition-data`
- Downloaded filename always `"qmap.hdf"` (regardless of server path)
- Guard: return `no_update` if `model.new_partition is None`
- Temp file must be deleted after `dcc.send_file()` is called (in a `finally` block)
- `save-status.children` is NOT updated by the download callback — it has its own status area (`partition-status`) or returns silently
- Ruff must pass clean

---

## File Map

| Status | File | Change |
|--------|------|--------|
| Modify | `src/pysimplemask/web/partition_layout.py` | Add `dcc.Download` + "Download" button |
| Modify | `src/pysimplemask/web/partition_callbacks.py` | Add `download_partition_cb` callback |
| Modify | `tests/web/test_partition_layout.py` | Add `"download-partition-btn"` and `"download-partition-data"` to `REQUIRED_IDS` |

---

## Task 1: Download button + callback

**Files:**
- Modify: `src/pysimplemask/web/partition_layout.py`
- Modify: `src/pysimplemask/web/partition_callbacks.py`
- Modify: `tests/web/test_partition_layout.py`

**Interfaces:**
- `dcc.Download(id="download-partition-data")` — invisible Dash component; its `data` prop is set by the callback to trigger the browser download
- `download_partition_cb` — `Input("download-partition-btn", "n_clicks")` → `Output("download-partition-data", "data")`; calls `model.save_partition(tmp_path)` then `dcc.send_file(tmp_path, filename="qmap.hdf")`

- [ ] **Step 1: Update `REQUIRED_IDS` in the test (failing)**

In `tests/web/test_partition_layout.py`, add the two new IDs to the `REQUIRED_IDS` set:

```python
REQUIRED_IDS = {
    "partition-tabs", "partition-compute-btn", "partition-status",
    "save-path", "save-partition-btn", "save-mask-btn", "save-status",
    "download-partition-btn", "download-partition-data",   # ← ADD THESE TWO
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
```

- [ ] **Step 2: Run the failing test**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/web/test_partition_layout.py::test_all_required_ids_present -v
```

Expected: FAIL — `Missing IDs: {'download-partition-btn', 'download-partition-data'}`.

- [ ] **Step 3: Update `partition_layout.py`**

In `build_partition_section()`, make two additions:

**3a.** Add `dcc.Download` to the top-level `children` list (before the Hr/Partition heading — it renders as nothing):

```python
def build_partition_section() -> html.Div:
    """Return the full partition + save section for the sidebar."""
    return html.Div(children=[
        dcc.Download(id="download-partition-data"),   # ← ADD THIS LINE
        html.Hr(),
        html.H4("Partition", style={"marginBottom": "4px"}),
        # ... rest unchanged ...
```

**3b.** Add the Download button after the existing Save buttons row. The Save section currently ends with:

```python
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
```

Replace the end of the section with:

```python
        html.Div(
            style={"display": "flex", "gap": "4px", "marginBottom": "4px"},
            children=[
                html.Button("Save Partition (HDF5)", id="save-partition-btn",
                            style={**_BTN, "flex": "1"}),
                html.Button("Save Mask (TIFF)", id="save-mask-btn",
                            style={**_BTN, "flex": "1"}),
            ],
        ),
        html.Button(
            "Download Partition (HDF5)",
            id="download-partition-btn",
            style={**_BTN, "width": "100%", "marginBottom": "4px"},
        ),
        html.Div(id="save-status", style={"fontSize": "11px", "color": "#555"}),
    ])
```

Also add `dcc` to the existing `from dash import dcc, html` import (it is already imported — no change needed).

- [ ] **Step 4: Run the layout test to confirm it passes**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/web/test_partition_layout.py -v
```

Expected: all 3 layout smoke tests PASS.

- [ ] **Step 5: Add `download_partition_cb` to `partition_callbacks.py`**

Add two imports at the top of the file (with the existing imports):

```python
import os
import tempfile

from dash import dcc  # add dcc alongside the existing dash imports
```

Then append this callback at the end of the file:

```python
# ---------------------------------------------------------------------------
# Download Partition (HDF5) — sends qmap to the browser
# ---------------------------------------------------------------------------


@callback(
    Output("download-partition-data", "data"),
    Input("download-partition-btn", "n_clicks"),
    prevent_initial_call=True,
)
def download_partition_cb(n_clicks):
    """Write the partition to a temp file and send it to the browser."""
    if not model.is_ready() or model.new_partition is None:
        return no_update
    with tempfile.NamedTemporaryFile(suffix=".hdf", delete=False) as f:
        fname = f.name
    try:
        model.save_partition(fname)
        return dcc.send_file(fname, filename="qmap.hdf")
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass
```

- [ ] **Step 6: Run the full test suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
```

Expected: all 154 tests pass (no regressions; no new tests were added because the download callback requires a live Dash server to integration-test).

- [ ] **Step 7: Ruff check**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check \
    src/pysimplemask/web/partition_layout.py \
    src/pysimplemask/web/partition_callbacks.py \
    tests/web/test_partition_layout.py
```

Expected: no output.

- [ ] **Step 8: Commit**

```bash
git add \
    src/pysimplemask/web/partition_layout.py \
    src/pysimplemask/web/partition_callbacks.py \
    tests/web/test_partition_layout.py
git commit -m "feat(web): add Download Partition (HDF5) button for browser download"
```
