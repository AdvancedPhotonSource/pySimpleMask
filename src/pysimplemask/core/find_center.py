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
