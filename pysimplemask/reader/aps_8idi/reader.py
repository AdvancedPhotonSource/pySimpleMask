import os
import glob
import h5py
import magic
import numpy as np
import hdf5plugin
from . import (HdfDataset, RigakuDataset, ImmDataset, Rigaku3MDataset)
import logging
from ..file_reader import FileReader


logger = logging.getLogger(__name__)


def get_file_type(fname):
    extname = os.path.splitext(fname)[-1]
    if extname not in ('.hdf', '.h5', '.hdf5', '.imm', '.bin', '.000'):
        return 'unknown_type'

    file_format = magic.from_file(fname)
    if 'Hierarchical Data Format' not in file_format:
        # legacy file
        return 'aps_legacy'
    try:
        with h5py.File(fname, 'r') as hf:
            if '/measurement/instrument/acquisition' in hf:
                return 'aps_metadata'
            elif '/entry_0000/instrument/id02-eiger500k-saxs' in hf:
                return 'esrf_hdf' 
            elif '/entry/data/data' in hf:
                return 'aps_hdf'
    except Exception:
        pass
    return 'unknow_hdf'


def get_fake_metadata(shape):
    # fake metadata
    logger.warn('failed to get the raw metadata, using default values instead')
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
        if get_file_type(f) == 'aps_metadata':
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


class APS8IDIReader(FileReader):
    def __init__(self, fname) -> None:
        super(APS8IDIReader, self).__init__(fname)
        self.handler = None
        self.ftype = 'Base Class'

        if fname.endswith('.bin'):
            logger.info('Rigaku 500k dataset')
            self.handler = RigakuDataset(fname, batch_size=1000)
        elif fname.endswith('.bin.000'):
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



def get_reader(fname):
    ext_name = os.path.splitext(fname)[-1]
    logger.info(f'using file extension {ext_name}')
    if ext_name in ('.hdf', '.h5', '.hdf5', '.imm', '.bin', '.000'):
        # exclude the hdf meta file; 
        ftype = get_file_type(fname)
        if ftype == 'aps_metadata':
            logger.info('please select the raw file not the meta file.')
            return None 
        elif ftype in ['aps_hdf', 'aps_legacy']:
            return APS8IDIReader(fname)
    else:
        raise ValueError(f'{ext_name} extension not supported')
   

if __name__ == '__main__':
    pass