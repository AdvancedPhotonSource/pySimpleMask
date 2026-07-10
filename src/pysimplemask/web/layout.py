"""Dash component tree for the pySimpleMask web viewer."""

from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html

from pysimplemask.core.reader.base_reader import DISPLAY_FIELD
from pysimplemask.web.mask_layout import build_mask_section

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
            # Incremented by load_file on each successful load; triggers update_display.
            dcc.Store(id="model-loaded", data=0),
            dcc.Store(id="drawn-shapes", data=[]),
            dcc.Store(id="param-constraints", data=[]),
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
            *build_mask_section().children,
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
                figure=go.Figure(),
                style={"flex": "1", "minHeight": "500px"},
                responsive=True,
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
