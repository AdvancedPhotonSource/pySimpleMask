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
