"""Array-to-Plotly-figure utilities for the pySimpleMask web viewer."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
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
        colormap: Matplotlib-compatible colormap name (e.g. ``"jet"``, ``"viridis"``).
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

    # Downsample large arrays: cap each axis at 1024 px for display.
    _MAX_PX = 1024
    h, w = display.shape
    if h > _MAX_PX or w > _MAX_PX:
        step = max(h // _MAX_PX, w // _MAX_PX, 1)
        display = display[::step, ::step]
        if center_vh is not None:
            center_vh = (center_vh[0] / step, center_vh[1] / step)

    # Apply colormap via matplotlib → H×W×3 uint8 RGB.
    # px.imshow with binary_string=True on a 2D array renders grayscale only;
    # passing a pre-colored RGB array gives correct hues and encodes as a PNG
    # (~1.5 MB) instead of a per-pixel JSON heatmap (~14 MB for 1k×1k arrays).
    vmin, vmax = display.min(), display.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(colormap)
    rgba = cmap(norm(display))          # H×W×4, float 0-1
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)   # H×W×3 uint8

    fig = px.imshow(rgb, origin="upper", binary_string=True)
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        autosize=True,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    if center_vh is not None:
        row, col = center_vh
        h_disp, w_disp = rgb.shape[:2]
        # compute_display_center applies a swing-angle correction that can push
        # the beam center far outside the detector area.  If the crosshair is
        # out of bounds, Plotly expands the axis to include it and the image
        # shrinks to invisible.  Skip the crosshair when it falls outside the
        # display canvas.
        if 0 <= row < h_disp and 0 <= col < w_disp:
            arm = max(h_disp, w_disp) * 0.02  # 2% of display size
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
