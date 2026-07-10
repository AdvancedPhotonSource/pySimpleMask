"""Unit tests for image_utils.make_figure."""

import numpy as np
import pytest

dash = pytest.importorskip("dash")
import plotly.graph_objects as go  # noqa: E402

from pysimplemask.web.image_utils import make_figure  # noqa: E402


def test_make_figure_returns_go_figure():
    arr = np.ones((16, 16), dtype=np.float32) * 5.0
    fig = make_figure(arr)
    assert isinstance(fig, go.Figure)


def test_make_figure_image_trace_type():
    # binary_string=True produces a go.Image trace (canvas-based), not go.Heatmap (SVG).
    arr = np.ones((16, 16), dtype=np.float32)
    fig = make_figure(arr, center_vh=None)
    assert len(fig.data) == 1
    assert fig.data[0].type == "image"
    assert fig.data[0].source is not None  # base64 PNG binary string


def test_make_figure_has_two_traces_with_crosshair():
    arr = np.ones((64, 64), dtype=np.float32)
    fig = make_figure(arr, center_vh=(32, 32))
    assert len(fig.data) == 2
    assert fig.data[0].type == "image"
    assert fig.data[1].type == "scatter"


def test_make_figure_log_scale_produces_image():
    # Log scale is applied before PNG encoding; the trace is still go.Image.
    arr = np.array([[1.0, 10.0, 100.0], [0.1, 0.01, 1000.0]], dtype=np.float64)
    fig = make_figure(arr, log_scale=True)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "image"
    assert fig.data[0].source is not None


def test_make_figure_log_scale_handles_zeros():
    # All-zero or zero-containing arrays must not raise with log_scale=True.
    arr = np.array([[0.0, 5.0], [2.0, 0.0]], dtype=np.float32)
    fig = make_figure(arr, log_scale=True)
    assert isinstance(fig, go.Figure)
    assert fig.data[0].type == "image"


def test_make_figure_colormap_accepted():
    arr = np.ones((8, 8), dtype=np.float32)
    fig = make_figure(arr, colormap="viridis")
    assert isinstance(fig, go.Figure)
