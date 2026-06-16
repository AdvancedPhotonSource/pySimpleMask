"""Tests for the centro-symmetry beam-center finder."""

import numpy as np
import pytest

from pysimplemask.core.find_center import (
    center_crop,
    estimate_center2,
    find_center,
)


def _ring_image(shape=(128, 130), center=(64.0, 65.0), r0=30.0, w=6.0):
    """A smooth ring pattern, centro-symmetric about ``center``."""
    yy, xx = np.indices(shape)
    r = np.hypot(yy - center[0], xx - center[1])
    img = np.exp(-(((r - r0) / w) ** 2)) + 0.2 * np.exp(-((r / (2 * w)) ** 2))
    return img * 1000.0 + 1.0


# --- accuracy ---------------------------------------------------------------


def test_recovers_known_center_integer():
    cy, cx = 64, 65
    out = find_center(_ring_image((128, 130), (cy, cx)), center_guess=(cy + 4, cx - 5))
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, [cy, cx], atol=0.7)


def test_recovers_known_center_half_integer():
    cy, cx = 64.5, 65.5
    out = find_center(_ring_image((140, 140), (cy, cx)), center_guess=(cy + 3, cx + 3))
    assert np.allclose(out, [cy, cx], atol=0.8)


def test_recovers_without_guess():
    cy, cx = 70, 72
    out = find_center(_ring_image((150, 150), (cy, cx)))
    assert np.allclose(out, [cy, cx], atol=1.0)


def test_scale_linear():
    cy, cx = 64, 64
    out = find_center(
        _ring_image((128, 128), (cy, cx)), center_guess=(cy + 3, cx + 3), scale="linear"
    )
    assert np.allclose(out, [cy, cx], atol=0.8)


def test_perfect_guess_stays():
    cy, cx = 64, 64
    out = find_center(_ring_image((128, 128), (cy, cx)), center_guess=(cy, cx))
    assert np.allclose(out, [cy, cx], atol=0.5)


# --- masked / robustness ----------------------------------------------------


def test_recovers_with_gap_mask():
    cy, cx = 64, 64
    img = _ring_image((128, 128), (cy, cx))
    mask = np.ones(img.shape, dtype=bool)
    mask[61:67, :] = False  # detector gap across the center
    out = find_center(img, mask=mask, center_guess=(cy + 3, cx - 3))
    assert np.allclose(out, [cy, cx], atol=1.5)


def test_recovers_with_beamstop():
    cy, cx = 64, 64
    img = _ring_image((128, 128), (cy, cx))
    yy, xx = np.indices(img.shape)
    mask = np.hypot(yy - cy, xx - cx) >= 10  # block central disk
    img = img.copy()
    img[~mask] = 0
    out = find_center(img, mask=mask, center_guess=(cy + 4, cx + 4))
    assert np.allclose(out, [cy, cx], atol=1.5)


# --- edge cases -------------------------------------------------------------


def test_invalid_scale_raises():
    with pytest.raises(ValueError):
        find_center(_ring_image(), scale="sqrt")


def test_non_2d_raises():
    with pytest.raises(ValueError):
        find_center(np.zeros((2, 3, 4)))


def test_flat_image_returns_fallback():
    out = find_center(np.full((64, 64), 5.0), center_guess=(10, 20))
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, [10, 20])


def test_no_positive_pixels_returns_fallback():
    out = find_center(np.zeros((64, 64)), scale="log")
    assert np.allclose(out, [32, 32])  # image center (no guess)


def test_out_of_bounds_guess_falls_back():
    cy, cx = 30, 30
    out = find_center(_ring_image((64, 64), (cy, cx)), center_guess=(999, 999))
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, [cy, cx], atol=1.5)


def test_near_edge_guess_no_crash():
    out = find_center(_ring_image((128, 128), (64, 64)), center_guess=(1, 1))
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, [1, 1], atol=1.0)  # crop collapses -> guess returned


def test_returns_ndarray():
    out = find_center(_ring_image(), center_guess=(66, 66))
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)


# --- helpers ----------------------------------------------------------------


def test_estimate_center2_centroid():
    img = np.zeros((10, 10))
    img[3, 7] = 10.0
    assert np.allclose(estimate_center2(img), [3, 7])


def test_estimate_center2_empty_fallback():
    assert np.allclose(estimate_center2(np.zeros((10, 12))), [5, 6])


def test_center_crop_symmetric_and_capped():
    img = np.arange(100).reshape(10, 10).astype(float)
    center, crop, cmask, half = center_crop(img, center=(5, 5), max_radius=2)
    assert half == 2
    assert crop.shape == (5, 5)
    assert cmask is None
