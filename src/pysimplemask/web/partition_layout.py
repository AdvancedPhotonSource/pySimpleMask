# Copyright © UChicago Argonne LLC
# See LICENSE file for details
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
        dcc.Download(id="download-partition-data"),
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
                html.Button("Download Partition (HDF5)", id="download-partition-btn",
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
