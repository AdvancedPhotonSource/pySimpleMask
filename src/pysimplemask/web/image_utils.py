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

    # Downsample large arrays: serialising 2k×2k floats as JSON is ~50 MB and
    # stalls the browser.  Cap each axis at 1024 px for display.
    _MAX_PX = 1024
    h, w = display.shape
    if h > _MAX_PX or w > _MAX_PX:
        step = max(h // _MAX_PX, w // _MAX_PX, 1)
        display = display[::step, ::step]
        if center_vh is not None:
            center_vh = (center_vh[0] / step, center_vh[1] / step)

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
