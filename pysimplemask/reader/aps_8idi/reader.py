import numpy as np
import h5py
import hdf5plugin
import logging
import datetime
from ..file_reader import FileReader


logger = logging.getLogger(__name__)


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
    metadata['datetime'] = str(datetime.datetime.now())
    return metadata


def get_metadata(fname):
    # read real metadata
    keys = {
        'ccdx': '/entry/instrument/bluesky/metadata/ccdx',
        'ccdx0': '/entry/instrument/bluesky/metadata/ccdx0',
        'ccdy': '/entry/instrument/bluesky/metadata/ccdy',
        'ccdy0': '/entry/instrument/bluesky/metadata/ccdy0',
        'energy': '/entry/instrument/bluesky/metadata/X_energy',
        'det_dist': '/entry/instrument/bluesky/metadata/det_dist',
        'pix_dim': '/entry/instrument/bluesky/metadata/pix_dim_x',
        'bcx': '/entry/instrument/detector_1/beam_center_x',
        'bcy': '/entry/instrument/detector_1/beam_center_y',
        'xdim': '/entry/instrument/bluesky/metadata/xdim',
        'ydim': '/entry/instrument/bluesky/metadata/ydim'
    }

    meta = {}
    with h5py.File(fname, 'r') as f:
        for key, val in keys.items():
            meta[key] = np.squeeze(f[val][()])
        meta['data_name'] = fname

    ccdx, ccdx0 = meta['ccdx'], meta['ccdx0'] 
    ccdy, ccdy0 = meta['ccdy'], meta['ccdy0'] 

    meta['bcx'] = meta['bcx'] + (ccdx - ccdx0) / meta['pix_dim']
    meta['bcy'] = meta['bcy'] + (ccdy - ccdy0) / meta['pix_dim']
    meta['shape'] = (meta['ydim'], meta['xdim'])

    # scattering geometry
    meta['sg_type'] = 'transmission'
    meta['datetime'] = str(datetime.datetime.now())
    
    for key in ['det_dist', 'pix_dim']:
        meta[key] *= 1000

    return meta


class APS8IDIReader(FileReader):
    def __init__(self, fname) -> None:
        super(APS8IDIReader, self).__init__(fname)
        self.ftype = 'APS-8IDI-nexus'
        self.shape = None

    def get_scattering(self, begin_idx=0, num_frames=-1):
        with h5py.File(self.fname, 'r') as f:
            dset = f['/entry/data/data']
            if num_frames <= 0:
                num_frames= dset.shape[0]
            sl = slice(begin_idx, min(begin_idx + num_frames, dset.shape[0]))
            data = dset[sl].sum(axis=0).astype(np.float32)
        self.shape = data.shape
        return data
    
    def load_meta(self):
        if self.shape is None:
            self.get_scattering(0, 1)
        fname = self.fname.replace('.h5', '.hdf')
        meta = get_metadata(fname)
        meta['shape'] = self.shape
        return meta


if __name__ == '__main__':
    pass