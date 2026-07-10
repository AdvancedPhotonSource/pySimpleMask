"""Mask section callbacks for the pySimpleMask web viewer."""

from __future__ import annotations

import numpy as np
from dash import Input, Output, Patch, State, callback, ctx, no_update
from scipy import ndimage

from pysimplemask.web.image_utils import make_figure
from pysimplemask.web.server import model

# DISPLAY_FIELD indices
_PREVIEW = 5      # after mask_evaluate
_SCAT_MASK = 1    # after mask_apply
_SCATTERING = 0   # after reset


def _fig(channel_idx: int, colormap: str | None, log_scale_list: list) -> object:
    """Render the detector image for the given channel index."""
    arr = model.dset.data_display[channel_idx]
    return make_figure(
        arr,
        colormap=colormap or "jet",
        log_scale=bool(log_scale_list),
        center_vh=model.get_center("vh"),
    )


# ---------------------------------------------------------------------------
# Reset (primary owner of mask-status and display-channel.value outputs)
# ---------------------------------------------------------------------------


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Output("display-channel", "value"),
    Output("mask-status", "children"),
    Input("mask-reset-btn", "n_clicks"),
    State("colormap", "value"),
    State("log-scale", "value"),
    prevent_initial_call=True,
)
def mask_reset(n_clicks, colormap, log_scale_list):
    if not model.is_ready():
        return no_update, no_update, "Load a file first."
    model.mask_action("reset")
    return _fig(_SCATTERING, colormap, log_scale_list), _SCATTERING, "Mask reset."


# ---------------------------------------------------------------------------
# Blemish
# ---------------------------------------------------------------------------


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Output("display-channel", "value", allow_duplicate=True),
    Output("mask-status", "children", allow_duplicate=True),
    Input("blemish-eval-btn", "n_clicks"),
    State("blemish-path", "value"),
    State("blemish-key", "value"),
    State("colormap", "value"),
    State("log-scale", "value"),
    prevent_initial_call=True,
)
def blemish_evaluate(n_clicks, fname, key, colormap, log_scale_list):
    if not model.is_ready():
        return no_update, no_update, "Load a file first."
    if not fname:
        return no_update, no_update, "Enter a blemish file path."
    msg = model.mask_evaluate("mask_blemish", fname=fname, key=key or "/exchange/data")
    return _fig(_PREVIEW, colormap, log_scale_list), _PREVIEW, f"Blemish preview: {msg}"


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Output("display-channel", "value", allow_duplicate=True),
    Output("mask-status", "children", allow_duplicate=True),
    Input("blemish-apply-btn", "n_clicks"),
    State("colormap", "value"),
    State("log-scale", "value"),
    prevent_initial_call=True,
)
def blemish_apply(n_clicks, colormap, log_scale_list):
    if not model.is_ready():
        return no_update, no_update, "Load a file first."
    model.mask_apply("mask_blemish")
    return _fig(_SCAT_MASK, colormap, log_scale_list), _SCAT_MASK, "Blemish applied."


# ---------------------------------------------------------------------------
# External mask file
# ---------------------------------------------------------------------------


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Output("display-channel", "value", allow_duplicate=True),
    Output("mask-status", "children", allow_duplicate=True),
    Input("maskfile-eval-btn", "n_clicks"),
    State("maskfile-path", "value"),
    State("maskfile-key", "value"),
    State("colormap", "value"),
    State("log-scale", "value"),
    prevent_initial_call=True,
)
def maskfile_evaluate(n_clicks, fname, key, colormap, log_scale_list):
    if not model.is_ready():
        return no_update, no_update, "Load a file first."
    if not fname:
        return no_update, no_update, "Enter a mask file path."
    msg = model.mask_evaluate("mask_file", fname=fname, key=key or "/exchange/data")
    return _fig(_PREVIEW, colormap, log_scale_list), _PREVIEW, f"File preview: {msg}"


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Output("display-channel", "value", allow_duplicate=True),
    Output("mask-status", "children", allow_duplicate=True),
    Input("maskfile-apply-btn", "n_clicks"),
    State("colormap", "value"),
    State("log-scale", "value"),
    prevent_initial_call=True,
)
def maskfile_apply(n_clicks, colormap, log_scale_list):
    if not model.is_ready():
        return no_update, no_update, "Load a file first."
    model.mask_apply("mask_file")
    return _fig(_SCAT_MASK, colormap, log_scale_list), _SCAT_MASK, "File mask applied."


# ---------------------------------------------------------------------------
# Threshold evaluate / apply
# ---------------------------------------------------------------------------


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Output("display-channel", "value", allow_duplicate=True),
    Output("mask-status", "children", allow_duplicate=True),
    Input("thresh-eval-btn", "n_clicks"),
    State("thresh-low-enable", "value"),
    State("thresh-low", "value"),
    State("thresh-high-enable", "value"),
    State("thresh-high", "value"),
    State("colormap", "value"),
    State("log-scale", "value"),
    prevent_initial_call=True,
)
def threshold_evaluate(n_clicks, low_en, low, high_en, high,
                       colormap, log_scale_list):
    if not model.is_ready():
        return no_update, no_update, "Load a file first."
    msg = model.mask_evaluate(
        "mask_threshold",
        low=float(low) if low is not None else 0.0,
        high=float(high) if high is not None else 1e8,
        low_enable=bool(low_en),
        high_enable=bool(high_en),
    )
    return _fig(_PREVIEW, colormap, log_scale_list), _PREVIEW, f"Threshold preview: {msg}"


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Output("display-channel", "value", allow_duplicate=True),
    Output("mask-status", "children", allow_duplicate=True),
    Input("thresh-apply-btn", "n_clicks"),
    State("colormap", "value"),
    State("log-scale", "value"),
    prevent_initial_call=True,
)
def threshold_apply(n_clicks, colormap, log_scale_list):
    if not model.is_ready():
        return no_update, no_update, "Load a file first."
    model.mask_apply("mask_threshold")
    return _fig(_SCAT_MASK, colormap, log_scale_list), _SCAT_MASK, "Threshold applied."


# ---------------------------------------------------------------------------
# Morphology (Erode / Dilate / Open / Close) — applied to model.mask directly
# ---------------------------------------------------------------------------


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Output("display-channel", "value", allow_duplicate=True),
    Output("mask-status", "children", allow_duplicate=True),
    Input("morph-erode",  "n_clicks"),
    Input("morph-dilate", "n_clicks"),
    Input("morph-open",   "n_clicks"),
    Input("morph-close",  "n_clicks"),
    State("colormap", "value"),
    State("log-scale", "value"),
    prevent_initial_call=True,
)
def apply_morphology(erode, dilate, open_clicks, close_clicks,
                     colormap, log_scale_list):
    if not model.is_ready():
        return no_update, no_update, "Load a file first."
    action = ctx.triggered_id
    ops = {
        "morph-erode":  ndimage.binary_erosion,
        "morph-dilate": ndimage.binary_dilation,
        "morph-open":   ndimage.binary_opening,
        "morph-close":  ndimage.binary_closing,
    }
    if action not in ops:
        return no_update, no_update, no_update
    new_mask = ops[action](model.mask).astype(bool)
    model.mask = new_mask
    model.dset.update_mask(model.mask)
    return (_fig(_SCAT_MASK, colormap, log_scale_list), _SCAT_MASK,
            f"Morphology: {action.split('-')[1]} applied.")


# ---------------------------------------------------------------------------
# Draw tab — activate mode, capture shapes, evaluate, apply
# ---------------------------------------------------------------------------


@callback(
    Output("drawn-shapes", "data"),
    Input("detector-image", "relayoutData"),
    State("drawn-shapes", "data"),
    prevent_initial_call=True,
)
def capture_drawn_shapes(relayout_data, current):
    """Store Plotly's full shapes list whenever shapes change."""
    if relayout_data is None:
        return no_update
    shapes = relayout_data.get("shapes")
    if shapes is None:
        return no_update
    return list(shapes)


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Input("draw-activate-btn", "n_clicks"),
    State("draw-shape", "value"),
    prevent_initial_call=True,
)
def toggle_draw_mode(n_clicks, shape_type):
    """Switch dragmode to draw; toggle off on even clicks."""
    patch = Patch()
    if n_clicks and n_clicks % 2 == 1:
        patch["layout"]["dragmode"] = shape_type or "drawrect"
        patch["layout"]["newshape"] = {"line": {"color": "cyan", "width": 2}}
    else:
        patch["layout"]["dragmode"] = "pan"
    return patch


@callback(
    Output("drawn-shapes", "data", allow_duplicate=True),
    Output("detector-image", "figure", allow_duplicate=True),
    Input("draw-clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_draw_shapes(n_clicks):
    """Empty the shapes store and remove shapes from the figure."""
    patch = Patch()
    patch["layout"]["shapes"] = []
    return [], patch


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Output("display-channel", "value", allow_duplicate=True),
    Output("mask-status", "children", allow_duplicate=True),
    Input("draw-eval-btn", "n_clicks"),
    State("drawn-shapes", "data"),
    State("draw-mode", "value"),
    State("colormap", "value"),
    State("log-scale", "value"),
    prevent_initial_call=True,
)
def draw_evaluate(n_clicks, shapes, draw_mode, colormap, log_scale_list):
    if not model.is_ready():
        return no_update, no_update, "Load a file first."
    if not shapes:
        return no_update, no_update, "Draw shapes on the image first."
    from pysimplemask.web.shape_utils import plotly_shapes_to_mask
    keep_mask = plotly_shapes_to_mask(
        shapes, model.shape, mode=draw_mode or "exclusive"
    )
    # MaskArray.evaluate(arr) expects True where pixels should be masked OUT
    msg = model.mask_evaluate("mask_draw", arr=np.logical_not(keep_mask))
    return _fig(_PREVIEW, colormap, log_scale_list), _PREVIEW, f"Draw preview: {msg}"


@callback(
    Output("detector-image", "figure", allow_duplicate=True),
    Output("display-channel", "value", allow_duplicate=True),
    Output("mask-status", "children", allow_duplicate=True),
    Input("draw-apply-btn", "n_clicks"),
    State("colormap", "value"),
    State("log-scale", "value"),
    prevent_initial_call=True,
)
def draw_apply(n_clicks, colormap, log_scale_list):
    if not model.is_ready():
        return no_update, no_update, "Load a file first."
    model.mask_apply("mask_draw")
    return _fig(_SCAT_MASK, colormap, log_scale_list), _SCAT_MASK, "Draw mask applied."
