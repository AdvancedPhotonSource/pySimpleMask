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
    eps=1e-16
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
        roi_mask = (partition == region_label)
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
        saxs1d[0, n] = qlist[n]      # q
        saxs1d[1, n] = ref_val       # reference (mean_clipped or median)
        saxs1d[2, n] = thr_val       # threshold
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
