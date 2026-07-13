"""Dash callbacks for the pySimpleMask web viewer."""

from __future__ import annotations

import base64
import os
import tempfile

from dash import Input, Output, State, callback, ctx, no_update

from pysimplemask.core.reader.base_reader import DISPLAY_FIELD
from pysimplemask.web.image_utils import make_figure
from pysimplemask.web.server import model


# ---------------------------------------------------------------------------
# Load file
# ---------------------------------------------------------------------------


@callback(
    Output("model-loaded", "data"),
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
    """Load a scattering file and populate the metadata fields.

    Rather than outputting the figure directly (which Dash 4.x drops when
    returning many outputs at once), we increment ``model-loaded`` which
    triggers ``update_display`` to render the image via its own callback.
    """
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
    all_channel_labels = list(DISPLAY_FIELD) + list(model.qmap.keys())
    options = [{"label": v, "value": i} for i, v in enumerate(all_channel_labels)]

    return (
        n_clicks,  # increments model-loaded → triggers update_display
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
    State("meta-beam_center_x", "value"),
    State("meta-beam_center_y", "value"),
    State("meta-energy", "value"),
    State("meta-detector_distance", "value"),
    State("meta-pixel_size", "value"),
    prevent_initial_call=True,
)
def update_params(n_clicks, bcx, bcy, energy, distance, pixel_size):
    """Recompute qmap from edited metadata fields and refresh the image."""
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
    Output("detector-image", "figure", allow_duplicate=True),
    Output("meta-beam_center_x", "value", allow_duplicate=True),
    Output("meta-beam_center_y", "value", allow_duplicate=True),
    Output("status-msg", "children", allow_duplicate=True),
    Input("find-center-btn", "n_clicks"),
    State("display-channel", "value"),
    State("colormap", "value"),
    State("log-scale", "value"),
    prevent_initial_call=True,
)
def find_center_cb(n_clicks, channel_idx, colormap, log_scale_list):
    """Run the cross-correlation beam-center finder and replot with new crosshair."""
    if not model.is_ready():
        return no_update, no_update, no_update, "Load a file first."
    center_vh = model.find_center(beamstop_diameter=0)
    if center_vh is None:
        return no_update, no_update, no_update, "Find center failed."
    model.dset.set_center_vh(center_vh)
    model.update_parameters()
    meta = model.dset.metadata
    idx = int(channel_idx) if channel_idx is not None else 0
    fig = make_figure(
        model.dset.data_display[idx],
        colormap=colormap or "jet",
        log_scale=bool(log_scale_list),
        center_vh=model.get_center("vh"),
    )
    return fig, round(float(meta["beam_center_x"]), 4), round(float(meta["beam_center_y"]), 4), "Center found."


# ---------------------------------------------------------------------------
# Goto Max
# ---------------------------------------------------------------------------


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Output("meta-beam_center_x", "value", allow_duplicate=True),
    Output("meta-beam_center_y", "value", allow_duplicate=True),
    Output("status-msg", "children", allow_duplicate=True),
    Input("goto-max-btn", "n_clicks"),
    State("display-channel", "value"),
    State("colormap", "value"),
    State("log-scale", "value"),
    prevent_initial_call=True,
)
def goto_max_cb(n_clicks, channel_idx, colormap, log_scale_list):
    """Move beam center to max intensity pixel and replot with new crosshair."""
    if not model.is_ready():
        return no_update, no_update, no_update, "Load a file first."
    model.goto_max()
    model.update_parameters()
    meta = model.dset.metadata
    idx = int(channel_idx) if channel_idx is not None else 0
    fig = make_figure(
        model.dset.data_display[idx],
        colormap=colormap or "jet",
        log_scale=bool(log_scale_list),
        center_vh=model.get_center("vh"),
    )
    return fig, round(float(meta["beam_center_x"]), 4), round(float(meta["beam_center_y"]), 4), "Moved to max."


# ---------------------------------------------------------------------------
# Swap X/Y
# ---------------------------------------------------------------------------


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Output("meta-beam_center_x", "value", allow_duplicate=True),
    Output("meta-beam_center_y", "value", allow_duplicate=True),
    Output("status-msg", "children", allow_duplicate=True),
    Input("swap-xy-btn", "n_clicks"),
    State("display-channel", "value"),
    State("colormap", "value"),
    State("log-scale", "value"),
    prevent_initial_call=True,
)
def swap_xy_cb(n_clicks, channel_idx, colormap, log_scale_list):
    """Swap beam_center_x/y and replot with updated crosshair."""
    if not model.is_ready():
        return no_update, no_update, no_update, "Load a file first."
    model.dset.swapxy()
    model.update_parameters()
    meta = model.dset.metadata
    idx = int(channel_idx) if channel_idx is not None else 0
    fig = make_figure(
        model.dset.data_display[idx],
        colormap=colormap or "jet",
        log_scale=bool(log_scale_list),
        center_vh=model.get_center("vh"),
    )
    return fig, round(float(meta["beam_center_x"]), 4), round(float(meta["beam_center_y"]), 4), "X/Y swapped."


# ---------------------------------------------------------------------------
# Display controls (channel / colormap / log-scale)
# ---------------------------------------------------------------------------


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Input("display-channel", "value"),
    Input("colormap", "value"),
    Input("log-scale", "value"),
    Input("model-loaded", "data"),
    prevent_initial_call=True,
)
def update_display(channel_idx, colormap, log_scale_list, _model_loaded):
    """Re-render the detector image when display controls change."""
    if not model.is_ready():
        return no_update
    colormap = colormap or "jet"
    log_scale = bool(log_scale_list)
    idx = int(channel_idx) if channel_idx is not None else 0

    # binary_string=True (go.Image) bakes the colormap into the PNG server-side,
    # so all display changes — including colormap — require a full figure rebuild.
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
