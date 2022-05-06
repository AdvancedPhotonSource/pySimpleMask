import matplotlib.pyplot as plt
import numpy as np
import skimage.io as skio
from skimage.morphology import disk
from skimage.filters import median
from skimage.registration import phase_cross_correlation

img = skio.imread('../tests/data/saxs_test.tif')



def percentile_filter(img0, percentile_range=(2.0, 98.0)):
    img = np.copy(img0)
    # get ride of the bad pixels
    vmin = np.percentile(img.ravel(), percentile_range[0])
    vmax = np.percentile(img.ravel(), percentile_range[1])
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


def center_crop(img0, cen=None):
    img = np.copy(img0)
    if cen is None:
        cen = estimate_center(img)
    cen = (cen + 0.5).astype(np.int64)
    h = min(cen[0], img.shape[0] + 1 - cen[0])
    w = min(cen[1], img.shape[1] + 1 - cen[1])

    img = img[cen[0] - h: cen[0] + h + 1,
              cen[1] - w: cen[1] + w + 1]
        
    return cen, img 


def estimate_center_cross_correlation(img, center=None):
    cen_int, img = center_crop(img, center)
    img2 = np.flipud(np.fliplr(img))
    shift, _, _ = phase_cross_correlation(img, img2, upsample_factor=32)
    new_center = cen_int.astype(np.float64) + shift / 2.0
    return new_center


def find_center(img, iter_bad_pixel=1, iter_median_filter=3,
                scale='log', iter_center=3):
    # remove bad pixels using percentile filter
    for _ in range(iter_bad_pixel):
        img = percentile_filter(img)
    
    # remove bad pixels using median filter
    for _ in range(iter_median_filter):
        img = median_filter(img)

    if scale == 'log':
        min_value = np.min(img[img > 0])
        img[img <= 0] = min_value
        img = np.log10(img).astype(np.float32)
    else:
        img = img.astype(np.float32)

    center = None
    for n in range(iter_center):
        center = estimate_center_cross_correlation(img, center=center)

    return center


if __name__ == '__main__':
    print(find_center(img))