# Upload Result File Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add "Upload Result File" to the File section so users can push a local HDF5 file from their browser to the web server and immediately load it into the viewer.

**Architecture:** `dcc.Upload` component in `layout.py` receives the file as a base64 string; the callback decodes it, writes it to a temp file with `tempfile.NamedTemporaryFile`, calls `model.read_data()`, and returns the same 9 outputs as `load_file` (populating metadata fields and triggering the display update via the `model-loaded` store).

**Tech Stack:** Python `base64`, `tempfile`, `os`; Plotly Dash 4.x `dcc.Upload`

## Global Constraints

- Environment: `/local/MQICHU/envs/l2606_simplemask_refact/bin/`
- Upload component ID: `upload-result-file`
- Accept filter: `".hdf,.h5,.hdf5"`
- `beamline` is always `"APS_8IDI"` for uploaded files (auto-detection via `/xpcs` group handles result files)
- Temp file is **kept** after loading — `model.dset.fname` points to it; `file-path.value` is updated with the temp path
- Output ownership: `load_file` is primary owner of all 8 metadata/store outputs; upload callback uses `allow_duplicate=True` on all except `file-path.value` (which no callback currently owns — upload is primary)
- `model-loaded.data` is incremented as `(current_loaded or 0) + 1` to trigger `update_display`
- Status message on success: `f"Uploaded: {filename}"`
- Ruff must pass clean

---

## File Map

| Status | File | Change |
|--------|------|--------|
| Modify | `src/pysimplemask/web/layout.py` | Add `dcc.Upload(id="upload-result-file")` below the Load button |
| Modify | `src/pysimplemask/web/callbacks.py` | Add `upload_result_file_cb` callback; add `import base64, os, tempfile` |

---

## Task 1: Upload button + callback

**Files:**
- Modify: `src/pysimplemask/web/layout.py`
- Modify: `src/pysimplemask/web/callbacks.py`

**Interfaces:**
- `dcc.Upload(id="upload-result-file")` in layout — triggers callback when user selects a file
- `upload_result_file_cb(contents, filename, current_loaded)` — 9 outputs mirroring `load_file`

**Output table:**

| # | Output ID | Property | `allow_duplicate` |
|---|-----------|----------|-------------------|
| 0 | `model-loaded` | `data` | `True` |
| 1 | `file-path` | `value` | (none — primary owner) |
| 2 | `meta-beam_center_x` | `value` | `True` |
| 3 | `meta-beam_center_y` | `value` | `True` |
| 4 | `meta-energy` | `value` | `True` |
| 5 | `meta-detector_distance` | `value` | `True` |
| 6 | `meta-pixel_size` | `value` | `True` |
| 7 | `status-msg` | `children` | `True` |
| 8 | `display-channel` | `options` | `True` |

- [ ] **Step 1: Add `dcc.Upload` to `layout.py`**

In `_sidebar()`, locate the Load button:

```python
            html.Button(
                "Load",
                id="load-btn",
                style={"width": "100%", "marginBottom": "8px"},
            ),
            html.Div(
                id="status-msg",
                ...
            ),
```

Insert the Upload component between them:

```python
            html.Button(
                "Load",
                id="load-btn",
                style={"width": "100%", "marginBottom": "8px"},
            ),
            dcc.Upload(
                id="upload-result-file",
                children=html.Button(
                    "Upload Result File",
                    style={"width": "100%", "cursor": "pointer"},
                ),
                accept=".hdf,.h5,.hdf5",
                style={"width": "100%", "marginBottom": "8px"},
            ),
            html.Div(
                id="status-msg",
                ...
            ),
```

- [ ] **Step 2: Run existing web tests to confirm no regression from layout change**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/web/ -q
```

Expected: all web tests pass (the existing layout smoke tests don't check for `upload-result-file` — this is acceptable since there is no `test_layout.py` requiring it).

- [ ] **Step 3: Add `upload_result_file_cb` to `callbacks.py`**

Add three imports at the top of `src/pysimplemask/web/callbacks.py` (with the existing imports):

```python
import base64
import os
import tempfile
```

Then append this callback at the end of the file:

```python
# ---------------------------------------------------------------------------
# Upload Result File — receive HDF5 from browser, write to temp, load
# ---------------------------------------------------------------------------


@callback(
    Output("model-loaded",           "data",     allow_duplicate=True),
    Output("file-path",              "value"),
    Output("meta-beam_center_x",     "value",    allow_duplicate=True),
    Output("meta-beam_center_y",     "value",    allow_duplicate=True),
    Output("meta-energy",            "value",    allow_duplicate=True),
    Output("meta-detector_distance", "value",    allow_duplicate=True),
    Output("meta-pixel_size",        "value",    allow_duplicate=True),
    Output("status-msg",             "children", allow_duplicate=True),
    Output("display-channel",        "options",  allow_duplicate=True),
    Input("upload-result-file",  "contents"),
    State("upload-result-file",  "filename"),
    State("model-loaded",        "data"),
    prevent_initial_call=True,
)
def upload_result_file_cb(contents, filename, current_loaded):
    """Decode an uploaded HDF5 file, write to temp, and load into the model."""
    if contents is None:
        return (no_update,) * 9

    suffix = os.path.splitext(filename or ".hdf")[1] or ".hdf"
    _, b64 = contents.split(",", 1)
    file_bytes = base64.b64decode(b64)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(file_bytes)
        tmp_path = f.name

    _ERR = (no_update,) * 7  # placeholders for outputs 0–6

    try:
        ok = model.read_data(fname=tmp_path, beamline="APS_8IDI")
    except Exception as exc:
        return (*_ERR, f"Upload error: {exc}", no_update)
    if not ok:
        return (*_ERR, "Failed to load uploaded file.", no_update)

    meta = model.dset.metadata
    all_channel_labels = list(DISPLAY_FIELD) + list(model.qmap.keys())
    options = [{"label": v, "value": i} for i, v in enumerate(all_channel_labels)]

    return (
        (current_loaded or 0) + 1,           # 0: model-loaded → triggers update_display
        tmp_path,                              # 1: file-path.value
        round(float(meta["beam_center_x"]), 4),  # 2
        round(float(meta["beam_center_y"]), 4),  # 3
        round(float(meta["energy"]), 6),          # 4
        round(float(meta["detector_distance"]), 6),  # 5
        float(meta["pixel_size"]),                # 6
        f"Uploaded: {filename}",                  # 7: status-msg
        options,                                   # 8: display-channel.options
    )
```

- [ ] **Step 4: Run the full test suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
```

Expected: all 154 tests pass.

- [ ] **Step 5: Ruff check**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check \
    src/pysimplemask/web/layout.py \
    src/pysimplemask/web/callbacks.py
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add src/pysimplemask/web/layout.py src/pysimplemask/web/callbacks.py
git commit -m "feat(web): add Upload Result File button to File section"
```
