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


def test_make_figure_has_one_trace_without_crosshair():
    arr = np.ones((16, 16), dtype=np.float32)
    fig = make_figure(arr, center_vh=None)
    assert len(fig.data) == 1


def test_make_figure_has_two_traces_with_crosshair():
    arr = np.ones((64, 64), dtype=np.float32)
    fig = make_figure(arr, center_vh=(32, 32))
    assert len(fig.data) == 2
    assert fig.data[1].type == "scatter"


def test_make_figure_log_scale_applies_log10():
    arr = np.array([[1.0, 10.0, 100.0], [0.1, 0.01, 1000.0]], dtype=np.float64)
    fig = make_figure(arr, log_scale=True)
    z = fig.data[0].z
    assert z is not None
    np.testing.assert_allclose(z[0][1], np.log10(10.0), rtol=1e-5)
    np.testing.assert_allclose(z[0][2], np.log10(100.0), rtol=1e-5)


def test_make_figure_log_scale_handles_zeros():
    arr = np.array([[0.0, 5.0], [2.0, 0.0]], dtype=np.float32)
    fig = make_figure(arr, log_scale=True)
    assert isinstance(fig, go.Figure)
    z = fig.data[0].z
    # zeros are replaced by floor (min positive value = 2.0)
    floor = 2.0
    np.testing.assert_allclose(z[0][0], np.log10(floor), rtol=1e-4)


def test_make_figure_colormap_accepted():
    arr = np.ones((8, 8), dtype=np.float32)
    fig = make_figure(arr, colormap="viridis")
    assert isinstance(fig, go.Figure)
