import numpy as np
from astropy.io import fits
from skimage.io import imread
from .timepix_reader import get_saxs_mp as timepix_get_saxs
import logging


logger = logging.getLogger(__name__)



class FileReader(object):
    def __init__(self, fname) -> None:
        self.fname = fname
        self.ftype = 'Base Class'
        self.saxs = None
        self.meta = None
        self.saxs_lin_min = None 
        self.saxs_log_min = None
        self.saxs_lin = None 
        self.saxs_log = None
        pass

    def clean_data(self, mask=None):
        mask = self.saxs > 0
        saxs_nonzero_1d = self.saxs[mask]
        self.saxs_lin_min = np.percentile(saxs_nonzero_1d, 1)
        self.saxs_log_min = np.log10(self.saxs_lin_min)
        self.saxs_lin = np.copy(self.saxs)
        # self.saxs_lin[~mask] += self.saxs_lin_min
        self.saxs_lin += self.saxs_lin_min
        self.saxs_log = np.log10(self.saxs_lin)
        # self.saxs_log[~mask] = np.nan
    
    def get_center(self):
        center = (self.meta['bcx'], self.meta['bcy'])
        return center
    
    def get_scattering_with_mask(self, mask, log_style=True):
        roi = mask > 0
        if log_style:
            data = np.copy(self.saxs_log)
        else:
            data = np.copy(self.saxs_lin)
        data_min = np.min(data[roi])
        data[~roi] = data_min
        return data

    def get_scattering(self, *kargs, **kwargs):
        raise NotImplementedError

    def get_data(self, *kargs, **kwargs):
        return None

    def load_meta(self):
        # implement the method that reads the real metadata; other wise it
        # will just return some fake metadata as the place holder
        # return get_fake_metadata(self.shape)
        raise NotImplementedError


class TiffReader(FileReader):
    def __init__(self, fname) -> None:
        self.fname = fname
        self.shape = None

    def get_scattering(self, **kwargs):
        data = imread(self.fname)
        self.shape = data.shape
        return data


class TimePixRawReader(FileReader):

    def __init__(self, fname) -> None:
        self.fname = fname
        self.shape = None

    def get_scattering(self, **kwargs):
        data = timepix_get_saxs(self.fname, 8)
        self.shape = data.shape
        return data



class FitsReader(FileReader):
    def __init__(self, fname) -> None:
        self.fname = fname
        self.shape = None

    def get_scattering(self, index=2, **kwargs):
        with fits.open(self.fname) as f:
            data = np.array(f[index].data)
        self.shape = data.shape
        return data
