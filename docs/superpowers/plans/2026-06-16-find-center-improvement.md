# find_center Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden and sharpen `core/find_center.py` (robustness guards, center-of-mass initial guess, iterate-to-convergence, bounded crop, consistent `np.ndarray` returns) and add a synthetic-fixture test suite — keeping the centro-symmetry cross-correlation algorithm.

**Architecture:** A preprocessing + initial-guess + convergence-loop pipeline over the existing helpers. `find_center` validates input, builds a masked log/linear normalized image, picks an initial center (validated `center_guess` else center of mass), then repeatedly crops a symmetric window, cross-correlates it with its 180° flip, and nudges the center by half the shift until it converges. All edge cases return a sensible center instead of crashing.

**Tech Stack:** Python 3.12, numpy, scikit-image (`phase_cross_correlation`), pytest. Env at `/local/MQICHU/envs/l2606_simplemask_refact/bin` (alias `PY` below). Repo root: `/home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact`.

**Spec:** `docs/superpowers/specs/2026-06-16-find-center-improvement-design.md`.

---

## File structure

- **Rewrite** `src/pysimplemask/core/find_center.py` — the finder. Public `find_center(img, mask=None, scale="log", center_guess=None, tol=0.1, max_iter=10, max_radius=None) -> np.ndarray`; helpers `estimate_center2`, `center_crop`, `estimate_center_cross_correlation`, `_initial_center`, `_fallback_center`; module constant `_MIN_HALF_SIZE = 8`. The biased `estimate_center` (brightest-10%) is removed.
- **Create** `tests/core/test_find_center.py` — synthetic centro-symmetric fixtures + accuracy, robustness, masked, edge-case, and API tests.
- No caller changes: `SimpleMaskModel.find_center` calls `find_center(scat, mask=..., center_guess=..., scale="log")`, all still valid.

---

## Task 1: Write the test suite (red)

**Files:**
- Create: `tests/core/test_find_center.py`

- [ ] **Step 1: Write the complete test module**

File `tests/core/test_find_center.py`:

```python
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
```

- [ ] **Step 2: Run the suite against the current code to confirm it fails**

Run: `cd /home/beams4/MQICHU/Tools_cloud/xpcs_toolchains/pySimpleMask_refact && /local/MQICHU/envs/l2606_simplemask_refact/bin/python -m pytest tests/core/test_find_center.py -q`
Expected: FAIL — current `find_center` returns a `list` (not `np.ndarray`), crashes on the masked gap/beamstop paths (masked `phase_cross_correlation` returns only the shift, which the old `shift, _, _ =` unpack cannot handle), and has no guards for invalid scale / non-2-D / flat / no-positive / out-of-bounds. `center_crop` also has a different return arity. Confirm multiple failures/errors. Do not fix yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/core/test_find_center.py
git commit -m "test(core): add find_center synthetic-fixture suite (red)"
```

---

## Task 2: Rewrite find_center.py (green)

**Files:**
- Modify (full rewrite): `src/pysimplemask/core/find_center.py`

- [ ] **Step 1: Replace the file contents**

File `src/pysimplemask/core/find_center.py` (entire new contents):

```python
"""Beam-center finder for centro-symmetric scattering patterns.

A transmission scattering pattern is point-symmetric about the direct-beam
center (``I(q) = I(-q)``). This module locates that center by cross-correlating
the image with its 180-degree flip and iterating to convergence.
"""

import logging

import numpy as np
from skimage.registration import phase_cross_correlation

logger = logging.getLogger(__name__)

# Symmetric crops smaller than this half-size (pixels) are too small to refine on.
_MIN_HALF_SIZE = 8


def estimate_center2(img, mask=None):
    """Return the intensity-weighted centroid (center of mass) as ``[y, x]``.

    Falls back to the geometric image center when there is no signal.
    """
    if mask is not None:
        img = img * mask
    total = float(np.sum(img))
    if total <= 0:
        return np.array([img.shape[0] / 2.0, img.shape[1] / 2.0])
    yy, xx = np.indices(img.shape)
    return np.array([np.sum(img * yy) / total, np.sum(img * xx) / total])


def center_crop(img, mask=None, center=None, max_radius=None):
    """Crop the largest square centered exactly on ``center``.

    The half-size is the distance to the nearest edge, optionally capped by
    ``max_radius``. The per-crop background (minimum over valid pixels) is
    subtracted. Returns ``(center_int, cropped_img, cropped_mask, half_size)``.
    """
    if center is None:
        center = estimate_center2(img, mask)
    center = np.round(np.asarray(center)).astype(int)
    center[0] = int(np.clip(center[0], 0, img.shape[0] - 1))
    center[1] = int(np.clip(center[1], 0, img.shape[1] - 1))

    half_size = min(
        center[0],
        img.shape[0] - 1 - center[0],
        center[1],
        img.shape[1] - 1 - center[1],
    )
    if max_radius is not None:
        half_size = min(half_size, int(max_radius))

    sl_v = slice(center[0] - half_size, center[0] + half_size + 1)
    sl_h = slice(center[1] - half_size, center[1] + half_size + 1)
    cropped_img = img[sl_v, sl_h]
    cropped_mask = mask[sl_v, sl_h] if mask is not None else None

    if cropped_mask is not None and np.any(cropped_mask):
        min_value = np.min(cropped_img[cropped_mask > 0])
    elif cropped_img.size > 0:
        min_value = np.min(cropped_img)
    else:
        min_value = 0.0
    cropped_img = cropped_img - min_value
    return center, cropped_img, cropped_mask, int(half_size)


def estimate_center_cross_correlation(img, mask, center, max_radius=None):
    """Refine ``center`` by one cross-correlation of the crop with its 180 flip."""
    center_int, cropped_img, cropped_mask, half_size = center_crop(
        img, mask, center, max_radius=max_radius
    )
    if half_size < _MIN_HALF_SIZE:
        logger.warning(
            "center crop too small (half_size=%d); skipping refinement", half_size
        )
        return np.asarray(center, dtype=float)

    moving_image = np.flip(cropped_img)
    if cropped_mask is not None and np.mean(cropped_mask > 0) < 0.98:
        reference_mask = cropped_mask
        moving_mask = np.flip(cropped_mask)
    else:
        reference_mask = None
        moving_mask = None

    result = phase_cross_correlation(
        cropped_img,
        moving_image,
        reference_mask=reference_mask,
        moving_mask=moving_mask,
        upsample_factor=4,
        overlap_ratio=0.75,
    )
    # Unmasked mode returns (shift, error, phasediff); masked mode returns shift.
    shift = result[0] if isinstance(result, tuple) else result
    return center_int.astype(float) + np.asarray(shift, dtype=float) / 2.0


def _fallback_center(img, center_guess):
    """Best center to return when refinement cannot run."""
    if center_guess is not None:
        return np.asarray(center_guess, dtype=float)
    return np.array([img.shape[0] / 2.0, img.shape[1] / 2.0])


def _initial_center(work, mask, center_guess):
    """In-bounds ``center_guess`` if given, else the center of mass."""
    if center_guess is not None:
        c = np.asarray(center_guess, dtype=float)
        if 0 <= c[0] < work.shape[0] and 0 <= c[1] < work.shape[1]:
            return c
        logger.warning("center_guess %s out of bounds; using center of mass", c)
    return estimate_center2(work, mask)


def find_center(
    img,
    mask=None,
    scale="log",
    center_guess=None,
    tol=0.1,
    max_iter=10,
    max_radius=None,
):
    """Locate the beam center of a centro-symmetric scattering image.

    Args:
        img: 2-D scattering image.
        mask: Optional boolean mask of valid pixels.
        scale: ``"log"`` (default) or ``"linear"`` preprocessing.
        center_guess: Optional initial ``(y, x)``; falls back to center of mass.
        tol: Convergence threshold on the per-iteration shift, in pixels.
        max_iter: Maximum refinement iterations.
        max_radius: Optional cap on the symmetric-crop half-size (perf knob).

    Returns:
        np.ndarray: refined center ``[y, x]`` as floats.
    """
    if scale not in ("log", "linear"):
        raise ValueError(f"scale must be 'log' or 'linear', got {scale!r}")
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError(f"img must be 2-D, got {img.ndim}-D")

    if mask is None:
        mask = np.ones(img.shape, dtype=bool)

    work = img.astype(np.float64).copy()
    work[mask == 0] = 0

    if scale == "log":
        positive = work[work > 0]
        if positive.size == 0:
            logger.warning("no positive pixels; returning fallback center")
            return _fallback_center(img, center_guess)
        work[work <= 0] = positive.min()
        work = np.log10(work)

    span = float(work.max() - work.min())
    if span <= 0:
        logger.warning("flat image; returning fallback center")
        return _fallback_center(img, center_guess)
    work = (work - work.min()) / span

    center = _initial_center(work, mask, center_guess)
    for _ in range(max_iter):
        new_center = estimate_center_cross_correlation(work, mask, center, max_radius)
        shift_mag = float(np.hypot(*(np.asarray(new_center) - np.asarray(center))))
        center = new_center
        if shift_mag < tol:
            break
    return np.asarray(center, dtype=float)
```

- [ ] **Step 2: Run the find_center tests**

Run: `/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m pytest tests/core/test_find_center.py -q`
Expected: all tests PASS. If a masked test (`test_recovers_with_gap_mask` / `test_recovers_with_beamstop`) is off by >1.5 px, widen that test's `atol` to 2.0 (masked `phase_cross_correlation` yields integer-pixel shifts, so half-integer center resolution) — do not change the implementation.

- [ ] **Step 3: Commit**

```bash
git add src/pysimplemask/core/find_center.py
git commit -m "feat(core): harden + sharpen find_center (CoM guess, convergence, guards)"
```

---

## Task 3: Verify the whole package and lint

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m pytest tests -q`
Expected: all green (previous 69 + the new find_center tests).

- [ ] **Step 2: Lint**

Run: `/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m ruff check src tests`
Expected: `All checks passed!` (fix any unused import the rewrite leaves, e.g. confirm `estimate_center`/`skimage.io` references are gone).

- [ ] **Step 3: Confirm the caller still works headless**

Run:
```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -c "
import numpy as np
from pysimplemask.core.find_center import find_center
yy, xx = np.indices((100, 100))
r = np.hypot(yy - 50, xx - 50)
img = np.exp(-(((r - 25) / 5) ** 2)) * 1000 + 1
print('center:', find_center(img, center_guess=(53, 47)))
print('Qt-free:', __import__('sys').modules.get('PySide6') is None)
"
```
Expected: a center near `[50, 50]`; `Qt-free: True`.

- [ ] **Step 4: Commit (if any lint fixes were needed)**

```bash
git add -A
git commit -m "chore(core): lint fixes after find_center rewrite"
```

---

## Self-review notes (spec coverage)

- Spec §"Public API" → Task 2 (signature, ndarray return, dropped `iter_center`, new params).
- §"Algorithm changes" (CoM guess, convergence loop) → Task 2 `_initial_center` + loop; tests `test_recovers_without_guess`, `test_perfect_guess_stays`.
- §"Robustness guards" → Task 2 guards; tests `test_invalid_scale_raises`, `test_non_2d_raises`, `test_flat_image_returns_fallback`, `test_no_positive_pixels_returns_fallback`, `test_out_of_bounds_guess_falls_back`, `test_near_edge_guess_no_crash`.
- §"Performance" (`max_radius`, early-exit) → Task 2 `center_crop` cap + convergence `break`; `test_center_crop_symmetric_and_capped`.
- §"Testing" → Task 1 (accuracy, gap, beamstop, edge, API).
- §"Risks" (masked return shape) → Task 2 `result[0] if isinstance(result, tuple) else result`; covered by the masked tests.
```
