
from .reader.imm_reader_with_plot import IMMReader8ID
from .reader.rigaku_reader import RigakuReader
from .reader.hdf2sax import hdf2saxs
from .reader.timepix_reader import get_saxs_mp as timepix_get_saxs
import os
import numpy as np
import h5py
import hdf5plugin
from skimage.io import imread
from astropy.io import fits


def verify_metadata_hdf(fname):
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

class FileReader(object):

    def __init__(self, fname) -> None:
        self.fname = fname
        pass

    def get_scattering(self, *kargs, **kwargs):
        raise NotImplementedError
    
    def load_meta(self):
        # implement the method that reads the real metadata; other wise it
        # will just return some fake metadata as the place holder
        return get_fake_metadata(self.shape) 


class APS8IDIReader(FileReader):
    def __init__(self, fname) -> None:
        super(APS8IDIReader, self).__init__(fname)

    def get_scattering(self, num_frames=-1, begin_idx=0, **kwargs):
        # seeks directory of existing hdf program
        dirname = os.path.dirname(os.path.realpath(self.fname))
        files = os.listdir(os.path.dirname(os.path.realpath(self.fname)))

        saxs = None
        for fname in files:
            if fname.endswith('.bin'):
                print("-----------.bin found.-----------")
                bin_file = os.path.join(dirname, fname)
                reader = RigakuReader(bin_file)
                saxs = reader.load()

            # seeks .imm file
            elif fname.endswith('.imm'):
                print("-----------.imm found.-----------")
                imm_file = os.path.join(dirname, fname)
                reader = IMMReader8ID(imm_file, no_of_frames=num_frames)
                saxs = reader.calc_avg_pixel()

            # seeks .h5 file
            elif fname.endswith('.h5') or fname.endswith('.hdf'):
                hdf_file = os.path.join(dirname, fname)
                with h5py.File(hdf_file, 'r') as hf:
                    if '/entry/data/data' in hf:
                        # correct h5 file, contains raw data
                        print("-----------.hdf/.h5 found.-----------")
                        saxs = hdf2saxs(hdf_file, begin_idx=begin_idx,
                                        num_frames=num_frames)
        return saxs

    def load_meta(self):
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
        with h5py.File(self.fname, 'r') as f:
            for key, val in keys.items():
                meta[key] = np.squeeze(f[val][()])
            meta['data_name'] = os.path.basename(self.fname).encode("ascii")

        ccdx, ccdx0 = meta['ccdx'], meta['ccdx0']
        ccdz, ccdz0 = meta['ccdz'], meta['ccdz0']

        meta['bcx'] = meta['bcx0'] + (ccdx - ccdx0) / meta['pix_dim']
        meta['bcy'] = meta['bcy0'] + (ccdz - ccdz0) / meta['pix_dim']
        meta.pop('bcx0', None)
        meta.pop('bcy0', None)

        return meta
    


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
    

class NormalHDFReader(FileReader):
    def __init__(self, fname) -> None:
        self.fname = fname
        self.shape = None

    def get_scattering(self, key='/entry/data/data', **kwargs):
        data = hdf2saxs(self.fname, key=key, **kwargs)
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
    if ext_name in ('.hdf', '.h5', '.hdf5'):
        if verify_metadata_hdf(fname):
            return APS8IDIReader(fname)
        else:
            return NormalHDFReader(fname)
    elif ext_name in ('.tif', '.tiff'):
        return TiffReader(fname)
    elif ext_name in ('.fits'):
        return FitsReader(fname)
    elif ext_name == '.raw':
        return TimePixRawReader(fname)


if __name__ == '__main__':
    pass