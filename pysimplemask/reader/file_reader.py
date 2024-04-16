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
        pass

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
