# find_center improvement — design

**Date:** 2026-06-16
**Branch:** refact
**Scope:** `src/pysimplemask/core/find_center.py` (the beam-center finder) and a new test
module `tests/core/test_find_center.py`. No change to callers.

## Goal

Harden and sharpen the beam-center finder across four dimensions the user selected —
robustness, accuracy, performance, and tests — while keeping the sound core algorithm
(exploiting the centro-symmetry of scattering patterns via cross-correlation of the image
with its 180° flip). Approach **B (harden + sharpen)**, not an algorithmic overhaul.

## Background — current mechanism

A transmission scattering pattern is point-symmetric about the direct-beam center
(`I(q) = I(-q)`). `find_center` finds that point by: pick an initial guess → crop the
largest square centered on it → cross-correlate the crop with its 180° flip → move the
center by half the measured shift → repeat. Current helpers: `estimate_center`
(brightest-10% centroid), `estimate_center2` (intensity-weighted center of mass),
`center_crop` (symmetric crop + background subtract), `estimate_center_cross_correlation`
(masked phase cross-correlation, half-shift correction), `find_center` (preprocess →
guess → fixed 2 iterations).

## Decisions (settled with the user)

1. Goals: robustness + accuracy + performance + tests (general hardening; no specific incident).
2. Approach **B**.
3. The **default center result may change** (more accurate) — accepted.
4. `max_radius` default **`None`** (unbounded; accuracy-first) — a documented knob for large detectors.

## Public API

```python
def find_center(img, mask=None, scale="log", center_guess=None,
                tol=0.1, max_iter=10, max_radius=None):
    ...
    return np.ndarray([y, x])  # float
```

- Caller `SimpleMaskModel.find_center` passes `mask`, `center_guess`, `scale` — unaffected.
- Removes the unused `iter_center=2` parameter in favor of `tol` / `max_iter`
  (iterate-to-convergence). No caller passes `iter_center`.
- New optional params: `tol` (pixels, convergence threshold, default 0.1), `max_iter`
  (default 10), `max_radius` (crop half-size cap, default `None`).
- **Return type standardized to `np.ndarray([y, x])` float** (today
  `estimate_center_cross_correlation`/`find_center` return a `list`). The caller indexes
  `[0]`/`[1]` then `float(...)`, so an ndarray is compatible.

## Algorithm changes (accuracy)

- **Initial guess → center-of-mass.** Use `estimate_center2` (intensity-weighted centroid)
  as the default initial guess instead of `estimate_center` (brightest-10% centroid), which
  a beamstop or hot pixels bias. The biased `estimate_center` is **removed** (internal,
  unused after this change).
- **Iterate to convergence.** Replace the fixed 2 passes with a loop:
  `for _ in range(max_iter): new = refine(center); shift = ||new - center||; center = new;
  if shift < tol: break`. Converges in 1–2 for a good guess; allows more when needed,
  capped at `max_iter`.
- Keep the core unchanged: symmetric crop → `np.flip` (180°) → masked
  `phase_cross_correlation(..., upsample_factor=4, overlap_ratio=0.75)` → `center + shift/2`.

## Robustness guards

- `scale not in {"log", "linear"}` → `ValueError` (was a bare `assert`, stripped under `-O`).
- `img` not 2-D → `ValueError`.
- **No positive pixels** in the `log` branch → fall back to the guess (or image center) with
  a warning, instead of `np.min(empty)` raising.
- **Flat image** (`max == min` after preprocessing) → return the guess/CoM instead of a
  divide-by-zero in normalization.
- `center_guess` out of bounds → fall back to center-of-mass.
- **Crop collapses near an edge**: if the symmetric half-size drops below a minimum
  (`_MIN_HALF_SIZE`, e.g. 8 px), log a warning and return the current center rather than
  "refine" on a meaningless few-pixel window.
- **All-masked crop** → guard the `min` used for background subtraction.
- Consistent `np.ndarray` returns from all helpers.

## Performance

- **Early-exit** on convergence — typically fewer than the previous fixed 2 iterations.
- **`max_radius` cap** on the symmetric crop bounds the cross-correlation FFT for large
  (4k+) detectors; the center is determined by structure near the beam, so a bounded window
  preserves accuracy while cutting cost. Default `None` (no behavior change); set to bound
  worst-case cost.

## Internal structure (after)

- `estimate_center2(img, mask) -> np.ndarray` — CoM guess (kept; minor guards).
- `center_crop(img, mask, center, max_radius=None) -> (center_int, cropped_img, cropped_mask)`
  — symmetric crop honoring `max_radius`, guarded background subtraction, reports the
  achieved half-size so the caller can detect collapse.
- `estimate_center_cross_correlation(img, mask, center, max_radius=None) -> np.ndarray` —
  unchanged masked-CC logic; returns ndarray; passes `max_radius` through.
- `find_center(...)` — validate → preprocess (mask, log/normalize with guards) → initial
  guess (validated `center_guess` else CoM) → convergence loop → return ndarray.
- `estimate_center` (brightest-10%) — removed.

## Testing — `tests/core/test_find_center.py` (synthetic, no external data)

Build centro-symmetric radial patterns at a known center on a grid (e.g. a smooth ring /
`exp(-((r-r0)/w)^2)` or `1/(1+r^2)`), then:

- **Accuracy**: `find_center` recovers the known center within ~0.5 px for integer and
  half-integer centers, with and without `center_guess`, for `scale` in {`log`, `linear`}.
- **Robustness**: recovers the center with (a) a detector-gap mask, and (b) a **beamstop**
  (a zeroed disk at the center) — exercising the CoM guess + masked CC.
- **Edge cases**: invalid `scale` → `ValueError`; non-2-D `img` → `ValueError`; flat image
  returns without crashing; out-of-bounds `center_guess` falls back; near-edge guess that
  collapses the crop returns the guess with no crash.
- **Convergence / perf**: converges within `max_iter`; a near-perfect guess early-exits in
  ≤2 iterations.
- **API**: return value is an `np.ndarray` of two floats.

Tolerances account for `upsample_factor=4` (≈⅛-px per step) plus discretization of the
synthetic pattern.

## Risks & mitigations

- **Default result shift** (CoM + convergence vs brightest-10% + fixed-2): intended;
  pinned by the recovery tests on known-center fixtures.
- **`max_radius` too small** could miss large-q symmetry → default `None`; documented as the
  large-detector knob.
- **phase_cross_correlation masked mode** needs consistent masks; the refine step passes a
  reference mask (and a flipped moving mask when coverage warrants), guarded for the
  no-mask case. Covered by the gap/beamstop tests.
