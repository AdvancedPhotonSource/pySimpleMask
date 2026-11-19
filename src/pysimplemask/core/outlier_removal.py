# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import numpy as np


def compute_outlier_percentile(values, cutoff, percentiles, eps=1e-16):
    """
    For a given 1D array 'values', clip extremes based on the given 'percentiles',
    compute mean/std of the clipped subset, then define:
       reference = mean_clipped
       threshold = mean_clipped + cutoff * std_clipped
    Returns:
       reference_val, threshold_val, outlier_mask
    where outlier_mask is a boolean array of the same shape as 'values'.
    """
    p_lo, p_hi = np.percentile(values, percentiles)

    # Guard against degenerate p_lo == p_hi
    if p_lo >= p_hi:
        p_lo, p_hi = values.min() - eps, values.max() + eps

    clipped_vals = values[(values >= p_lo) & (values <= p_hi)]
    if clipped_vals.size == 0:
        # If everything is outside the clip range => no outliers
        # We'll treat it like a degenerate case: reference=0 => skip
        return 0.0, 0.0, np.zeros_like(values, dtype=bool)

    mean_c = np.mean(clipped_vals)
    std_c = np.std(clipped_vals)

    if std_c < eps:
        # Region is essentially constant => no outliers
        reference_val = mean_c
        threshold_val = mean_c
        outlier_mask = np.zeros_like(values, dtype=bool)
    else:
        reference_val = mean_c
        threshold_val = mean_c + cutoff * std_c
        outlier_mask = np.abs(values - mean_c) >= (cutoff * std_c)

    return reference_val, threshold_val, outlier_mask


def compute_outlier_mad(values, cutoff, eps=1e-16):
    """
    For a given 1D array 'values', compute:
       median_val = median(values)
       mad_val = median(|x - median_val|)
       reference = median_val
       threshold = median_val + cutoff * mad_val
    Returns:
       reference_val, threshold_val, outlier_mask
    where outlier_mask is a boolean array of the same shape as 'values'.
    """
    median_val = np.median(values)
    abs_dev = np.abs(values - median_val)
    mad_val = np.median(abs_dev)

    if mad_val < eps:
        # Region is essentially constant => no outliers
        reference_val = median_val
        threshold_val = median_val
        outlier_mask = np.zeros_like(values, dtype=bool)
    else:
        reference_val = median_val
        threshold_val = median_val + cutoff * mad_val
        outlier_mask = abs_dev >= (cutoff * mad_val)

    return reference_val, threshold_val, outlier_mask


def outlier_removal_with_saxs(
    qlist,
    partition,
    saxs_lin,
    method="percentile",
    cutoff=3.0,
    percentile=(5, 95),
    eps=1e-16,
):
    """
    Unified outlier removal for SAXS data using either a percentile-based method
    or a Median Absolute Deviation (MAD) method, separated into helper functions.

    For each region (label = 1..N in 'partition'):
      - Gathers all pixels in that region.
      - Depending on 'method':
          'percentile': uses compute_outlier_percentile(...)
          'mad': uses compute_outlier_mad(...)
      - The reference, threshold, max_value, and raw_avg are recorded.

    Finally, it filters out columns where reference <= 0, and returns:
      - saxs1d: shape (5, k)
        Rows: [q, reference, threshold, max_val, raw_avg]
      - bad_pixel_all: shape (2, M) listing outlier pixel indices.

    Parameters
    ----------
    qlist : 1D array
        The q-values for each labeled region (labels 1..num_q).
    partition : array-like (same shape as saxs_lin)
        Integer mask of region labels. 0 = invalid region, 1..num_q = valid.
    saxs_lin : array-like (same shape as partition)
        SAXS intensity data.
    method : {'percentile', 'mad'}, optional
        Determines the outlier removal strategy. Default is 'percentile'.
    cutoff : float, optional
        Multiplier for outlier detection. Default is 3.0.
    percentile : tuple of two floats, optional
        The (low, high) percentile used for clipping if method='percentile'.
        Default is (5, 95).
    eps : float, optional
        Small value to avoid division by zero or near-zero. Default is 1e-16.

    Returns
    -------
    saxs1d : np.ndarray of shape (5, k)
        Rows: [q_value, reference, threshold, max_val, raw_avg],
        filtered so that reference > 0.
    bad_pixel_all : np.ndarray of shape (2, M)
        2D indices (row, col) of all outlier pixels in 'saxs_lin'.
    """
    # 1) Precompute raw average for each region label=1..num_q using bincount
    partition_r = partition.ravel()
    saxs_lin_r = saxs_lin.ravel()

    scat_sum = np.bincount(partition_r, weights=saxs_lin_r)
    scat_cnt = np.bincount(partition_r)
    scat_avg_full = scat_sum / np.clip(scat_cnt, 1, None)  # shape: (max_label+1,)

    # label=0 is invalid region => we skip, region labels => [1..num_q]
    # We'll have an array of raw averages for region i = i-1 index
    scat_avg_raw = scat_avg_full[1:]  # shape: (num_q,)

    # We'll store [q, reference, threshold, max_val, raw_avg] for each region
    saxs1d = np.zeros((5, qlist.size), dtype=np.float64)

    # Collect outlier indices in a list
    bad_pixel_list = []

    for n in range(qlist.size):
        region_label = n + 1
        roi_mask = partition == region_label
        if not np.any(roi_mask):
            # No pixels for this label => skip
            continue

        idx_2d = np.nonzero(roi_mask)  # (2, #pixels_in_region)
        values = saxs_lin[idx_2d]

        # 2) Dispatch to the selected method
        if method.lower() == "percentile":
            ref_val, thr_val, outlier_mask = compute_outlier_percentile(
                values, cutoff=cutoff, percentiles=percentile, eps=eps
            )
        elif method.lower() == "mad":
            ref_val, thr_val, outlier_mask = compute_outlier_mad(
                values, cutoff=cutoff, eps=eps
            )
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'percentile' or 'mad'.")

        # 3) Record region stats
        saxs1d[0, n] = qlist[n]  # q
        saxs1d[1, n] = ref_val  # reference (mean_clipped or median)
        saxs1d[2, n] = thr_val  # threshold
        saxs1d[3, n] = values.max()  # max value
        saxs1d[4, n] = scat_avg_raw[n]  # raw average (from bincount)

        # 4) Accumulate outlier indices
        if np.any(outlier_mask):
            bad_pix_coords = np.array(idx_2d)[:, outlier_mask]
            bad_pixel_list.append(bad_pix_coords)

    # 5) Filter out columns where the reference <= 0
    valid_cols = saxs1d[1] > 0
    saxs1d = saxs1d[:, valid_cols]

    # 6) Combine all outlier indices
    if bad_pixel_list:
        bad_pixel_all = np.hstack(bad_pixel_list)
    else:
        bad_pixel_all = np.zeros((2, 0), dtype=int)

    return saxs1d, bad_pixel_all


def _tile_bounds(start, stop, box_size, include_partial):
    """``[(a, b), ...]`` of adjacent ``box_size`` tiles laid over ``[start, stop)``.

    Only complete tiles fit unless ``include_partial`` is set, in which case
    the last tile is clipped to ``stop`` so the whole span is covered.
    """
    span = stop - start
    n = span // box_size
    if include_partial and span % box_size:
        n += 1
    return [
        (start + i * box_size, min(start + (i + 1) * box_size, stop)) for i in range(n)
    ]


def outlier_removal_adjacent_boxes(
    saxs_lin,
    mask,
    box_size=32,
    method="percentile",
    cutoff=3.0,
    percentile=(5, 95),
    eps=1e-16,
    region=None,
):
    """Outlier removal by dividing the detector into adjacent square boxes.

    The image is split into non-overlapping ``box_size × box_size`` tiles
    (only complete tiles; edge remainder is ignored). Masked or non-positive
    pixels are excluded. Within each box the chosen metric (percentile or MAD)
    flags outlier pixels. Boxes are then sorted by their mean valid-pixel
    intensity so the result plots as a 1-D curve analogous to the q-ring curve.

    Parameters
    ----------
    saxs_lin : np.ndarray, shape (H, W)
        Raw scattering image (linear, not log).
    mask : np.ndarray, shape (H, W), dtype bool
        True = valid pixel.
    box_size : int
        Side length of each square box in pixels.
    method : {'percentile', 'mad'}
        Outlier detection strategy.
    cutoff : float
        Multiplier for the outlier threshold.
    percentile : tuple of two floats
        (low, high) percentile for clipping when method='percentile'.
    eps : float
        Guard against near-zero division.
    region : np.ndarray, shape (H, W), dtype bool, optional
        Restrict the removal to the pixels where ``region`` is True (e.g. the
        rectangle ROI drawn in the GUI's Outlier tab). The tile grid is then
        anchored at the top-left corner of the region's bounding box and the
        edge tiles are clipped to that box, so the whole region is covered
        even when its extent is not a multiple of ``box_size`` (unlike the
        whole-image case, where the edge remainder is ignored). ``None``
        processes the whole image.

    Returns
    -------
    saxs1d : np.ndarray, shape (5, k)
        Rows: [box_index_sorted, reference, threshold, max_val, raw_avg].
        Only boxes with at least one valid pixel and reference > 0 are kept.
        ``box_index_sorted`` is a monotonically increasing float rank (0, 1, 2, …).
    bad_pixel_all : np.ndarray, shape (2, M)
        Row/col indices of all detected outlier pixels.
    """
    if box_size < 1:
        raise ValueError(f"box_size must be >= 1, got {box_size}")
    H, W = saxs_lin.shape
    if region is None:
        rb0, rb1, cb0, cb1 = 0, H, 0, W
        partial_tiles = False
    else:
        region = np.asarray(region, dtype=bool)
        if region.shape != saxs_lin.shape:
            raise ValueError(
                f"region shape {region.shape} does not match image shape {saxs_lin.shape}"
            )
        rows = np.flatnonzero(region.any(axis=1))
        cols = np.flatnonzero(region.any(axis=0))
        if rows.size == 0:
            return np.zeros((5, 0)), np.zeros((2, 0), dtype=int)
        rb0, rb1 = int(rows[0]), int(rows[-1]) + 1
        cb0, cb1 = int(cols[0]), int(cols[-1]) + 1
        mask = mask & region
        partial_tiles = True

    row_tiles = _tile_bounds(rb0, rb1, box_size, partial_tiles)
    col_tiles = _tile_bounds(cb0, cb1, box_size, partial_tiles)
    if not row_tiles or not col_tiles:
        return np.zeros((5, 0)), np.zeros((2, 0), dtype=int)

    method_lc = method.lower()
    if method_lc not in ("percentile", "mad"):
        raise ValueError(f"Unknown method '{method}'. Use 'percentile' or 'mad'.")

    records = []   # (raw_avg, ref, thr, max_val, bad_coords)

    for r0, r1 in row_tiles:
        for c0, c1 in col_tiles:
            box_mask = mask[r0:r1, c0:c1]
            box_data = saxs_lin[r0:r1, c0:c1]

            valid = box_mask & (box_data > 0)
            if not np.any(valid):
                continue

            rows_v, cols_v = np.nonzero(valid)
            values = box_data[rows_v, cols_v]
            raw_avg = float(values.mean())

            if method_lc == "percentile":
                ref, thr, om = compute_outlier_percentile(
                    values, cutoff=cutoff, percentiles=percentile, eps=eps
                )
            else:  # "mad"
                ref, thr, om = compute_outlier_mad(values, cutoff=cutoff, eps=eps)

            if ref <= 0:
                continue

            # Convert local box coords back to global image coords
            global_rows = rows_v + r0
            global_cols = cols_v + c0
            if np.any(om):
                bad_coords = np.array(
                    [global_rows[om], global_cols[om]], dtype=int
                )
            else:
                bad_coords = np.zeros((2, 0), dtype=int)

            records.append((raw_avg, ref, thr, float(values.max()), bad_coords))

    if not records:
        return np.zeros((5, 0)), np.zeros((2, 0), dtype=int)

    # Sort boxes by raw mean intensity (descending: brightest first).
    records.sort(key=lambda r: r[0], reverse=True)

    k = len(records)
    saxs1d = np.zeros((5, k), dtype=np.float64)
    bad_list = []
    for i, (raw_avg, ref, thr, max_val, bad_coords) in enumerate(records):
        saxs1d[0, i] = float(i)      # sorted box rank (x-axis)
        saxs1d[1, i] = ref           # reference
        saxs1d[2, i] = thr           # threshold
        saxs1d[3, i] = max_val       # max value
        saxs1d[4, i] = raw_avg       # raw average
        if bad_coords.shape[1] > 0:
            bad_list.append(bad_coords)

    bad_pixel_all = np.hstack(bad_list) if bad_list else np.zeros((2, 0), dtype=int)
    return saxs1d, bad_pixel_all
