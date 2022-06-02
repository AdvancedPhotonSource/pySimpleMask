import numpy as np
import skimage.io as skio
from skimage.morphology import disk
from skimage.filters import median
from skimage.registration import phase_cross_correlation
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import binary_dilation
# import matplotlib.pyplot as plt


def percentile_filter(img0, mask, percentile_range=(2.0, 98.0)):
    img = np.copy(img0)

    valid_roi = mask > 0
    img_valid = img[valid_roi]

    # get ride of the bad pixels
    vmin = np.percentile(img_valid, percentile_range[0])
    vmax = np.percentile(img_valid, percentile_range[1])
    invalid_roi = np.logical_or(img <= vmin, img >= vmax)
    img_median = median(img, disk(15))
    img[invalid_roi] = img_median[invalid_roi]
    return img


def median_filter(img0, disk_size=7, threshold=3.0):
    img = np.copy(img0)
    img_median = median(img, disk(disk_size))
    img_median[img_median == 0] = 1
    roi = img * 1.0 / img_median > threshold
    img[roi] = img_median[roi]
    # only apply the upper litmit; photon counting detector can yield lots of
    # zeros;
    # roi = img * 1.0 / img_median < 1.0 / threshold
    # img[roi] = img_median[roi]
    return img


def estimate_center(img, threshold=90):
    # estimate center using percentile
    cutoff = np.percentile(img.ravel(), threshold)
    cen_idx = np.nonzero(img >= cutoff)
    cen = np.mean(np.array(cen_idx), axis=1)
    return cen


def estimate_center2(img0, mask):
    img = np.copy(img0)
    v, h = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]),
                       indexing='ij')
    img[mask == 0] = 0
    min_val = np.min(img[mask == 1])
    img = img - min_val

    v_cen = np.sum((img * v)[mask == 1]) / np.sum(img)
    h_cen = np.sum((img * h)[mask == 1]) / np.sum(img)
    return np.array((v_cen, h_cen))


def center_crop(img0, mask=None, cen=None):
    img = np.copy(img0)
    if cen is None:
        cen = estimate_center2(img, mask)

    cen = (cen + 0.5).astype(np.int64)
    h = min(cen[0], img.shape[0] + 1 - cen[0])
    w = min(cen[1], img.shape[1] + 1 - cen[1])
    size = min(h, w)
    h, w = size, size

    img = img[cen[0] - h: cen[0] + h + 1,
              cen[1] - w: cen[1] + w + 1]

    mask = mask[cen[0] - h: cen[0] + h + 1,
                cen[1] - w: cen[1] + w + 1]
    min_value = np.min(img[mask == 1])
    img = img - min_value
    return cen, img, mask 


def fix_gaps_interpolation(img, mask2d, iterations=2):
    data = img * mask2d
    mask = np.where(mask2d == 1)
    interp = NearestNDInterpolator(np.transpose(mask), data[mask])
    img = interp(*np.indices(data.shape))
    mask2d = binary_dilation(mask2d, iterations=iterations)
    img[mask2d == 0] = 0
    return img, mask2d


def estimate_center_cross_correlation(img0, mask, center):
    cen_int, img, mask = center_crop(img0, mask, center)
    moving_image = np.flipud(np.fliplr(img))

    if np.sum(mask) / mask.size < 0.98:
        moving_mask = np.flipud(np.fliplr(mask))
    else:
        moving_mask = None

    shift = phase_cross_correlation(img, moving_image,
        reference_mask=mask, moving_mask=moving_mask, upsample_factor=32,
        overlap_ratio=0.75)
    new_center = cen_int.astype(np.float64) + shift / 2.0

    return new_center


def find_center(img, mask=None, iter_bad_pixel=0, iter_median_filter=2,
                scale='log', iter_center=3, center_guess=None):

    if mask is None:
        mask = np.ones_like(img, dtype=np.bool)
    img[mask == 0] = 0

    # remove bad pixels using percentile filter
    for _ in range(iter_bad_pixel):
        img = percentile_filter(img, mask)

    # remove bad pixels using median filter
    for _ in range(iter_median_filter):
        img = median_filter(img) * mask

    if scale == 'log':
        min_value = np.min(img[img > 0])
        img[img <= 0] = min_value
        img = np.log10(img).astype(np.float32)
    else:
        img = img.astype(np.float32)
    
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img[mask == 0] = 0
    img, mask = fix_gaps_interpolation(img, mask, iterations=2)

    if center_guess is None:
        center = estimate_center(img)
    else:
        center = np.array(center_guess)
    
    for n in range(iter_center):
        center = estimate_center_cross_correlation(img, mask, center)

    return center


if __name__ == '__main__':
    img = skio.imread('../tests/data/saxs_test.tif')
    print(find_center(img))
