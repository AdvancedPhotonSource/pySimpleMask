"""Mask section layout — 6-tab Dash component tree for the sidebar."""

from __future__ import annotations

from dash import dcc, html

_BTN = {"padding": "2px 8px", "cursor": "pointer"}
_LBL = {"fontSize": "12px", "color": "#555"}
_INP = {"width": "100%", "boxSizing": "border-box", "fontSize": "12px"}
_PAD = {"padding": "6px"}


def build_mask_section() -> html.Div:
    """Return the full mask section (Hr + heading + tabs) for the sidebar."""
    return html.Div(children=[
        html.Hr(),
        html.H4("Mask", style={"marginBottom": "4px"}),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "8px",
                   "marginBottom": "8px"},
            children=[
                html.Button("Reset", id="mask-reset-btn", style=_BTN),
                html.Div(id="mask-status",
                         style={"fontSize": "11px", "color": "#666", "flex": "1"}),
            ],
        ),
        dcc.Tabs(
            id="mask-tabs",
            value="threshold",
            style={"fontSize": "12px"},
            children=[
                dcc.Tab(label="Blemish/Files", value="blemish",
                        children=_tab_blemish()),
                dcc.Tab(label="Threshold", value="threshold",
                        children=_tab_threshold()),
                dcc.Tab(label="Draw", value="draw",
                        children=_tab_draw()),
                dcc.Tab(label="Manual", value="manual",
                        children=_tab_manual()),
                dcc.Tab(label="Outlier", value="outlier",
                        children=_tab_outlier()),
                dcc.Tab(label="Parameterize", value="parameterize",
                        children=_tab_parameterize()),
            ],
        ),
    ])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _eval_apply(eval_id: str, apply_id: str) -> html.Div:
    return html.Div(
        style={"display": "flex", "gap": "4px", "marginTop": "6px"},
        children=[
            html.Button("Evaluate", id=eval_id, style={**_BTN, "flex": "1"}),
            html.Button("Apply",    id=apply_id, style={**_BTN, "flex": "1"}),
        ],
    )


def _file_row(path_id: str, key_id: str, label: str,
              default_key: str = "/exchange/data") -> html.Div:
    return html.Div(children=[
        html.Label(label, style=_LBL),
        dcc.Input(id=path_id, type="text", placeholder="/path/to/file",
                  style=_INP),
        dcc.Input(id=key_id, type="text", value=default_key,
                  placeholder="HDF5 key",
                  style={**_INP, "marginTop": "2px"}),
    ])


# ---------------------------------------------------------------------------
# Tab content functions
# ---------------------------------------------------------------------------


def _tab_blemish() -> list:
    return [html.Div(style=_PAD, children=[
        _file_row("blemish-path", "blemish-key", "Blemish file:"),
        _eval_apply("blemish-eval-btn", "blemish-apply-btn"),
        html.Hr(style={"margin": "8px 0"}),
        _file_row("maskfile-path", "maskfile-key", "External mask:"),
        _eval_apply("maskfile-eval-btn", "maskfile-apply-btn"),
    ])]


def _tab_threshold() -> list:
    return [html.Div(style=_PAD, children=[
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "4px",
                   "marginBottom": "4px"},
            children=[
                dcc.Checklist(id="thresh-low-enable",
                              options=[{"label": "Low", "value": "on"}],
                              value=["on"], inline=True),
                dcc.Input(id="thresh-low", type="number", value=0,
                          style={"flex": "1"}),
            ],
        ),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "4px",
                   "marginBottom": "4px"},
            children=[
                dcc.Checklist(id="thresh-high-enable",
                              options=[{"label": "High", "value": "on"}],
                              value=["on"], inline=True),
                dcc.Input(id="thresh-high", type="number", value=1e8,
                          style={"flex": "1"}),
            ],
        ),
        html.Div(
            style={"display": "flex", "gap": "2px", "marginBottom": "4px"},
            children=[
                html.Button("Erode",  id="morph-erode",  style={**_BTN, "flex": "1"}),
                html.Button("Dilate", id="morph-dilate", style={**_BTN, "flex": "1"}),
                html.Button("Open",   id="morph-open",   style={**_BTN, "flex": "1"}),
                html.Button("Close",  id="morph-close",  style={**_BTN, "flex": "1"}),
            ],
        ),
        _eval_apply("thresh-eval-btn", "thresh-apply-btn"),
    ])]


def _tab_draw() -> list:
    return [html.Div(style=_PAD, children=[
        html.Label("Shape:", style=_LBL),
        dcc.RadioItems(
            id="draw-shape",
            options=[
                {"label": "Circle",    "value": "drawcircle"},
                {"label": "Rectangle", "value": "drawrect"},
                {"label": "Polygon",   "value": "drawclosedpath"},
            ],
            value="drawrect",
            inline=True,
            style={"fontSize": "12px", "marginBottom": "4px"},
        ),
        html.Label("Mode:", style=_LBL),
        dcc.RadioItems(
            id="draw-mode",
            options=[
                {"label": "Exclusive (mask inside)", "value": "exclusive"},
                {"label": "Inclusive (keep inside)", "value": "inclusive"},
            ],
            value="exclusive",
            style={"fontSize": "12px", "marginBottom": "4px"},
        ),
        html.Div(
            style={"display": "flex", "gap": "4px", "marginBottom": "4px"},
            children=[
                html.Button("Activate Draw", id="draw-activate-btn",
                            style={**_BTN, "flex": "2"}),
                html.Button("Clear", id="draw-clear-btn",
                            style={**_BTN, "flex": "1"}),
            ],
        ),
        _eval_apply("draw-eval-btn", "draw-apply-btn"),
    ])]


def _tab_manual() -> list:
    return [html.Div(style=_PAD, children=[
        html.Label("Pixel list (row col per line):", style=_LBL),
        dcc.Textarea(
            id="manual-pixels",
            placeholder="row col\nrow col\n...",
            style={"width": "100%", "height": "80px", "boxSizing": "border-box",
                   "fontSize": "11px"},
        ),
        dcc.Upload(
            id="manual-upload",
            children=html.Button(
                "Upload file (.txt/.csv/.json)",
                style={**_BTN, "width": "100%", "marginTop": "4px"},
            ),
            accept=".txt,.csv,.json",
        ),
        _eval_apply("manual-eval-btn", "manual-apply-btn"),
    ])]


def _tab_outlier() -> list:
    return [html.Div(style=_PAD, children=[
        html.Label("Target:", style=_LBL),
        dcc.Dropdown(
            id="outlier-target",
            options=[
                {"label": "Circular Rings",  "value": "rings"},
                {"label": "Adjacent Boxes",  "value": "boxes"},
            ],
            value="rings",
            clearable=False,
            style={"fontSize": "12px", "marginBottom": "4px"},
        ),
        html.Label("Method:", style=_LBL),
        dcc.Dropdown(
            id="outlier-method",
            options=[
                {"label": "MAD",        "value": "mad"},
                {"label": "Percentile", "value": "percentile"},
            ],
            value="percentile",
            clearable=False,
            style={"fontSize": "12px", "marginBottom": "4px"},
        ),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "4px",
                   "marginBottom": "4px"},
            children=[
                html.Label("Cutoff:", style={**_LBL, "flexShrink": 0}),
                dcc.Input(id="outlier-cutoff", type="number", value=3.0,
                          style={"flex": "1"}),
            ],
        ),
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "4px",
                   "marginBottom": "4px"},
            children=[
                html.Label("Num rings:", id="outlier-param-label",
                           style={**_LBL, "flexShrink": 0}),
                dcc.Input(id="outlier-param", type="number", value=180,
                          style={"flex": "1"}),
            ],
        ),
        _eval_apply("outlier-eval-btn", "outlier-apply-btn"),
    ])]


def _tab_parameterize() -> list:
    return [html.Div(style=_PAD, children=[
        html.Div(
            style={"display": "flex", "gap": "4px", "marginBottom": "4px"},
            children=[
                html.Button("Add row",      id="param-add-btn",
                            style={**_BTN, "flex": "1"}),
                html.Button("Remove last",  id="param-remove-btn",
                            style={**_BTN, "flex": "1"}),
            ],
        ),
        html.Div(id="param-rows"),
        _eval_apply("param-eval-btn", "param-apply-btn"),
    ])]
