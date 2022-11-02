
import os
import glob
import h5py
import magic
import numpy as np
import hdf5plugin
from astropy.io import fits
from skimage.io import imread
from .reader.timepix_reader import get_saxs_mp as timepix_get_saxs
from .reader.aps_reader import HdfDataset, RigakuDataset, ImmDataset
# from .reader.hdf2sax import hdf2saxs


def verify_metadata_hdf(fname):
    file_format = magic.from_file(fname)
    if 'Hierarchical Data Format' not in file_format:
        return False

    try:
        with h5py.File(fname, 'r') as hf:
            if '/measurement/instrument/acquisition' in hf:
                return True
            else:
                return False
    except Exception:
        return False


def get_fake_metadata(shape):
    # fake metadata
    metadata = {
        'datetime': "2022-05-08 14:00:51,799",
        'energy': 11.0,         # keV
        'det_dist': 7800,       # mm
        'pix_dim': 55e-3,       # mm
        'bcx': shape[1] // 2.0,
        'bcy': shape[0] // 2.0
    }
    return metadata


def get_metadata(fname, shape):
    files = glob.glob(os.path.dirname(os.path.realpath(fname)) + '/*.*')

    meta_fname = None
    for f in files:
        if verify_metadata_hdf(f):
            meta_fname = f
            break

    if meta_fname is None:
        return get_fake_metadata(shape)

    # read real metadata
    keys = {
        'ccdx': '/measurement/instrument/acquisition/stage_x',
        'ccdx0': '/measurement/instrument/acquisition/stage_zero_x',
        'ccdz': '/measurement/instrument/acquisition/stage_z',
        'ccdz0': '/measurement/instrument/acquisition/stage_zero_z',
        'datetime': '/measurement/instrument/source_begin/datetime',
        'energy': '/measurement/instrument/source_begin/energy',
        'det_dist': '/measurement/instrument/detector/distance',
        'pix_dim': '/measurement/instrument/detector/x_pixel_size',
        'bcx0': '/measurement/instrument/acquisition/beam_center_x',
        'bcy0': '/measurement/instrument/acquisition/beam_center_y',
    }

    meta = {}
    with h5py.File(meta_fname, 'r') as f:
        for key, val in keys.items():
            meta[key] = np.squeeze(f[val][()])
        meta['data_name'] = os.path.basename(meta_fname).encode("ascii")

    ccdx, ccdx0 = meta['ccdx'], meta['ccdx0']
    ccdz, ccdz0 = meta['ccdz'], meta['ccdz0']

    meta['bcx'] = meta['bcx0'] + (ccdx - ccdx0) / meta['pix_dim']
    meta['bcy'] = meta['bcy0'] + (ccdz - ccdz0) / meta['pix_dim']
    meta.pop('bcx0', None)
    meta.pop('bcy0', None)
    return meta


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
        return get_fake_metadata(self.shape)


class APS8IDIReader(FileReader):
    def __init__(self, fname) -> None:
        super(APS8IDIReader, self).__init__(fname)
        self.handler = None
        self.ftype = 'Base Class'

        if fname.endswith('.bin'):
            print("-----------.bin found.-----------")
            self.handler = RigakuDataset(fname, batch_size=1000)

        elif fname.endswith('.imm'):
            print("-----------.imm found.-----------")
            self.handler = ImmDataset(fname, batch_size=100)

        elif fname.endswith('.h5') or fname.endswith('.hdf'):
            print("-----------h5/hdf found.-----------")
            self.handler = HdfDataset(fname, batch_size=100)
        self.shape = self.handler.det_size

    def get_scattering(self, **kwargs):
        return self.handler.get_scattering(**kwargs)
    
    def get_data(self, roi_list):
        return self.handler.get_data(roi_list)

    def load_meta(self):
        return get_metadata(self.fname, self.shape)


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


def read_raw_file(fname):
    ext_name = os.path.splitext(fname)[-1]
    if ext_name in ('.hdf', '.h5', '.hdf5', '.imm', '.bin'):
        # exclude the hdf meta file; 
        if verify_metadata_hdf(fname):
            print('please select the raw file not the meta file.')
            return None 
        else:
            return APS8IDIReader(fname)
    elif ext_name in ('.tif', '.tiff'):
        return TiffReader(fname)
    elif ext_name in ('.fits'):
        return FitsReader(fname)
    elif ext_name == '.raw':
        return TimePixRawReader(fname)


if __name__ == '__main__':
    pass