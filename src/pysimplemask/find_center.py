import numpy as np
import skimage.io as skio
from skimage.registration import phase_cross_correlation


def estimate_center(img, threshold=90):
    """Estimates the center of an image based on a percentile threshold.

    Parameters
    ----------
    img : ndarray
        The input image.
    threshold : float, optional
        The percentile threshold to use. Default is 90.

    Returns
    -------
    ndarray
        The estimated center coordinates (y, x).
    """
    cutoff = np.percentile(img.ravel(), threshold)
    y_indices, x_indices = np.where(img >= cutoff)
    if y_indices.size == 0 or x_indices.size == 0:
        return np.array([img.shape[0] // 2, img.shape[1] // 2])
    return np.array([np.mean(y_indices), np.mean(x_indices)])


def estimate_center2(img, mask):
    """Estimates the center using a weighted average based on pixel intensity.

    Parameters
    ----------
    img : ndarray
        The input image.
    mask : ndarray
        A boolean mask indicating valid pixels.

    Returns
    -------
    ndarray
        The estimated center coordinates (y, x).
    """
    masked_img = img * mask
    total_intensity = np.sum(masked_img)

    if total_intensity == 0:
        return np.array([img.shape[0] // 2, img.shape[1] // 2])

    y, x = np.indices(img.shape)
    y_cen = np.sum(masked_img * y) / total_intensity
    x_cen = np.sum(masked_img * x) / total_intensity
    return np.array([y_cen, x_cen])


def center_crop(img, mask=None, center=None):
    """Crops the image around the estimated center.

    Parameters
    ----------
    img : ndarray
        The input image.
    mask : ndarray, optional
        A boolean mask indicating valid pixels. Default is None.
    center : ndarray, optional
        The center coordinates (y, x) around which to crop. Default is None.

    Returns
    -------
    tuple
        A tuple containing the center coordinates, the cropped image, and the cropped mask (if provided).
    """
    if center is not None:
        if (
            center[0] < 0
            or center[1] < 0
            or center[0] >= img.shape[0]
            or center[1] >= img.shape[1]
        ):
            center = None

    if center is None:
        center = estimate_center2(img, mask)

    center = np.round(center).astype(int)
    half_size = min(
        center[0], img.shape[0] - center[0], center[1], img.shape[1] - center[1]
    )

    cropped_img = img[
        center[0] - half_size : center[0] + half_size + 1,
        center[1] - half_size : center[1] + half_size + 1,
    ]
    cropped_mask = None
    if mask is not None:
        cropped_mask = mask[
            center[0] - half_size : center[0] + half_size + 1,
            center[1] - half_size : center[1] + half_size + 1,
        ]

    min_value = np.min(
        cropped_img[cropped_mask == 1]
        if cropped_mask is not None and np.any(cropped_mask == 1)
        else cropped_img
    )
    cropped_img = cropped_img - min_value
    return center, cropped_img, cropped_mask


def estimate_center_cross_correlation(img, mask, center):
    """Refines the center estimation using phase cross-correlation.

    Parameters
    ----------
    img : ndarray
        The input image.
    mask : ndarray
        A boolean mask indicating valid pixels.
    center : ndarray
        The initial center coordinates (y, x).

    Returns
    -------
    ndarray
        The refined center coordinates (y, x).
    """
    center_int, cropped_img, cropped_mask = center_crop(img, mask, center)
    moving_image = np.flip(cropped_img)

    if cropped_mask is not None and np.sum(cropped_mask) / cropped_mask.size < 0.98:
        moving_mask = np.flip(cropped_mask)
    else:
        moving_mask = None

    shift, _, _ = phase_cross_correlation(
        cropped_img,
        moving_image,
        reference_mask=cropped_mask,
        moving_mask=moving_mask,
        upsample_factor=4,
        overlap_ratio=0.75,
    )
    new_center = center_int.astype(float) + shift / 2.0
    new_center = new_center.tolist()
    return new_center


def find_center(img, mask=None, scale="log", iter_center=2, center_guess=None):
    """Finds the center of an image using iterative refinement.

    Parameters
    ----------
    img : ndarray
        The input image.
    mask : ndarray, optional
        A boolean mask indicating valid pixels. Default is None.
    scale : str, optional
        The scaling to apply to the image ('log' or 'linear'). Default is 'log'.
    iter_center : int, optional
        The number of iterations to perform. Default is 2.
    center_guess : ndarray, optional
        An initial guess for the center coordinates (y, x). Default is None.

    Returns
    -------
    ndarray
        The refined center coordinates (y, x).
    """

    if mask is None:
        mask = np.ones_like(img, dtype=bool)

    masked_img = img.copy()
    masked_img[mask == 0] = 0

    assert scale in ("log", "linear")
    if scale == "log":
        min_value = np.min(masked_img[masked_img > 0])
        masked_img[masked_img <= 0] = min_value
        masked_img = np.log10(masked_img).astype(np.float32)
    else:
        masked_img = masked_img.astype(np.float32)

    masked_img = (masked_img - np.min(masked_img)) / (
        np.max(masked_img) - np.min(masked_img)
    )

    center = center_guess if center_guess is not None else estimate_center(masked_img)

    for _ in range(iter_center):
        center = estimate_center_cross_correlation(masked_img, mask, center)
    return center


if __name__ == "__main__":
    img = skio.imread("../tests/data/saxs_test.tif")
    print(find_center(img))
