import os
import glob
import h5py
import numpy as np
import hdf5plugin
from astropy.io import fits
from skimage.io import imread
from .reader.timepix_reader import get_saxs_mp as timepix_get_saxs
from .reader.aps_reader import (HdfDataset, RigakuDataset, ImmDataset,
                                Rigaku3MDataset)
import logging


logger = logging.getLogger(__name__)


def get_file_type(fname):
    extname = os.path.splitext(fname)[-1]
    if extname not in ('.hdf', '.h5', '.hdf5', '.imm', '.bin', 
                       '.000', '.001', '.002', '.003', '.004', '.005'):
        return 'unknown_type'

    if extname in ('.000', '.001', '.002', '.003', '.004', '.005',
                   '.imm', '.bin'):
        return "aps_legacy"

    try:
        with h5py.File(fname, 'r') as hf:
            # elif '/entry_0000/instrument/id02-eiger500k-saxs' in hf:
            #     return 'esrf_hdf' 
            if '/entry/data/data' in hf:
                return 'aps_hdf'
    except Exception:
        pass
    return 'unknow_hdf'


def get_fake_metadata(shape):
    # fake metadata
    logger.warning('failed to get the raw metadata, using default values instead')
    metadata = {
        # 'datetime': "2022-05-08 14:00:51,799",
        'energy': 12.4,         # keV
        'det_dist': 11500,       # mm
        'pix_dim': 75e-3,       # mm
        'bcx': 773.0,
        'bcy': 809.0
    }
    return metadata


def get_metadata(fname, shape):
    """
    get metadata from the raw file
    Parameters
    ----------
    fname: str
        raw file name
    shape: tuple
        shape of the data
    Returns
    -------
    metadata: dict
        metadata of the raw file
    """
    try:
        metadata = get_hdf_metadata(fname, shape)
    except Exception as e:
        logger.error(f'failed to get metadata from {fname}')
        logger.error(e)
        metadata = get_fake_metadata(shape)
    return metadata


def get_hdf_metadata(fname, shape):
    prefix = os.path.join(os.path.dirname(fname), '*_metadata.hdf')
    meta_fnames = glob.glob(prefix)
    assert len(meta_fnames) > 0, f'no *_metadata.hdf found in the folder of {fname}'
    if len(meta_fnames) > 1:
        logger.warning(f'multiple *_metadata.hdf found in the folder of {fname}. using the first one')
    meta_fname = meta_fnames[0]
    logger.info(f'using metadata file: {meta_fname}')
    # read real metadata
    keys = {
        'energy': '/entry/instrument/incident_beam/incident_energy',
        'ccdx': '/entry/instrument/detector_1/position_x',
        'ccdy': '/entry/instrument/detector_1/position_y',
        'ccdx0': '/entry/instrument/detector_1/beam_center_position_x',
        'ccdy0': '/entry/instrument/detector_1/beam_center_position_y',
        'x_pixel_size': '/entry/instrument/detector_1/x_pixel_size',
        'y_pixel_size': '/entry/instrument/detector_1/y_pixel_size',
        'bcx0': '/entry/instrument/detector_1/beam_center_x',
        'bcy0': '/entry/instrument/detector_1/beam_center_y',
        'det_dist': '/entry/instrument/detector_1/distance',
    }

    meta = {}
    with h5py.File(meta_fname, 'r') as f:
        for key, val in keys.items():
            meta[key] = f[val][()]
        meta['data_name'] = os.path.basename(meta_fname)

    ccdx, ccdx0 = meta['ccdx'], meta['ccdx0']
    ccdy, ccdy0 = meta['ccdy'], meta['ccdy0']

    meta['bcx'] = meta['bcx0'] + (ccdx - ccdx0) / meta['x_pixel_size']
    meta['bcy'] = meta['bcy0'] + (ccdy - ccdy0) / meta['y_pixel_size']
    meta.pop('bcx0', None)
    meta.pop('bcy0', None)
    meta['pix_dim'] = meta['x_pixel_size']
    meta.pop('x_pixel_size', None)
    meta.pop('y_pixel_size', None)
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
        rigaku_endings = tuple(f".bin.00{i}" for i in range(6))

        if fname.endswith('.bin'):
            logger.info('Rigaku 500k dataset')
            self.handler = RigakuDataset(fname, batch_size=1000)
        elif fname.endswith(rigaku_endings):
            logger.info('Rigaku 3M (6 x 500K) dataset')
            self.handler = Rigaku3MDataset(fname, batch_size=1000)
        elif fname.endswith('.imm'):
            logger.info('IMM dataset')
            self.handler = ImmDataset(fname, batch_size=100)
        elif fname.endswith('.h5') or fname.endswith('.hdf'):
            logger.info('APS HDF dataset')
            self.handler = HdfDataset(fname, batch_size=100)
        else:
            logger.error('Unsupported APS dataset')
            return None

        self.shape = self.handler.det_size

    def get_scattering(self, **kwargs):
        return self.handler.get_scattering(**kwargs)
    
    def get_data(self, roi_list):
        return self.handler.get_data(roi_list)

    def load_meta(self):
        return get_metadata(self.fname, self.shape)


# class EsrfReader(FileReader):
#     def __init__(self, fname) -> None:
#         super(EsrfReader, self).__init__(fname)
#         self.handler = None
#         self.ftype = 'ESRF HDF file'
#         self.handler = EsrfHdfDataset(fname)
#         self.shape = self.handler.det_size
# 
#     def get_scattering(self, **kwargs):
#         return self.handler.get_scattering(**kwargs)
#     
#     def get_data(self, roi_list):
#         return self.handler.get_data(roi_list)
# 
#     def load_meta(self):
#         keys = {
#         # 'ccdx': '/measurement/instrument/acquisition/stage_x',
#         # 'ccdx0': '/measurement/instrument/acquisition/stage_zero_x',
#         # 'ccdz': '/measurement/instrument/acquisition/stage_z',
#         # 'ccdz0': '/measurement/instrument/acquisition/stage_zero_z',
#         'datetime': '/entry_0000/start_time',
#         'energy': '/entry_0000/instrument/id02-eiger500k-saxs/header/WaveLength',
#         'det_dist': '/entry_0000/instrument/id02-eiger500k-saxs/header/SampleDistance',
#         'pix_dim': '/entry_0000/instrument/id02-eiger500k-saxs/header/PSize_1',
#         'bcx': '/entry_0000/instrument/id02-eiger500k-saxs/header/Center_1',
#         'bcy': '/entry_0000/instrument/id02-eiger500k-saxs/header/Center_2',
#         }
# 
#         meta = {}
#         with h5py.File(self.fname, 'r') as f:
#             for key, val in keys.items():
#                 val = f[val][()]
#                 if isinstance(val, bytes) and key != 'datetime':
#                     val = float(val.decode())
#                 if key == 'energy':
#                     val = 12.398e-10 / val
#                 if key in ['det_dist', 'pix_dim']:
#                     val = val * 1000
#                 meta[key] = val
#             meta['data_name'] = os.path.basename(self.fname).encode("ascii")
# 
#         return meta


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
    logger.info(f'using file extension {ext_name}')
    if ext_name in ('.hdf', '.h5', '.hdf5', '.imm', '.bin',
                    '.000', '.001', '.002', '.003', '.004', '.005'):
        # exclude the hdf meta file; 
        ftype = get_file_type(fname)
        if ftype == 'aps_metadata':
            logger.info('please select the raw file not the meta file.')
            return None 
        elif ftype in ['aps_hdf', 'aps_legacy']:
            return APS8IDIReader(fname)
        elif ftype == 'esrf_hdf':
            return EsrfReader(fname)
    elif ext_name in ('.tif', '.tiff'):
        return TiffReader(fname)
    elif ext_name in ('.fits'):
        return FitsReader(fname)
    elif ext_name == '.raw':
        return TimePixRawReader(fname)
    else:
        logger.info(f'file format with {ext_name} is not supported')
        return None


if __name__ == '__main__':
    pass
