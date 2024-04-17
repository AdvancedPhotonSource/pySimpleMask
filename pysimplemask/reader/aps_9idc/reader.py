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
    meta['datetime'] = str(datetime.datetime.now())
    return metadata


def get_metadata(fname):
    # read real metadata
    keys = {
        'ccdx': '/entry/metadata/det_y',
        'ccdz': '/entry/metadata/det_z',
        'energy': '/entry/metadata/energy',
        'det_dist': '/entry/metadata/det_distance',
        'pix_dim': '/entry/metadata/det_pixel_size',
        'bc': '/entry/metadata/beam_center',
        'alpha_i': '/entry/metadata/alpha_incident_deg',
        'shape': '/entry/metadata/det_shape', 
    }

    meta = {}
    with h5py.File(fname, 'r') as f:
        for key, val in keys.items():
            meta[key] = np.squeeze(f[val][()])
        meta['data_name'] = fname

    ccdx, ccdx0 = meta['ccdx'], 0 
    ccdz, ccdz0 = meta['ccdz'], 0

    meta['bcx'] = meta['bc'][0] + (ccdx - ccdx0) / meta['pix_dim']
    meta['bcy'] = meta['bc'][1] + (ccdz - ccdz0) / meta['pix_dim']
    meta.pop('bc', None)

    # scattering geometry
    meta['sg_type'] = 'reflection'
    meta['datetime'] = str(datetime.datetime.now())
    
    for key in ['det_dist', 'pix_dim']:
        meta[key] *= 1000

    return meta


class APS9IDCReader(FileReader):
    def __init__(self, fname) -> None:
        super(APS9IDCReader, self).__init__(fname)
        self.handler = None
        self.ftype = 'Base Class'

    def get_scattering(self, begin_idx=0, num_frames=-1):
        with h5py.File(self.fname, 'r') as f:
            dset = f['/entry/data/data']
            if num_frames <= 0:
                num_frames= dset.shape[0]
            sl = slice(begin_idx, min(begin_idx + num_frames, dset.shape[0]))
            data = dset[sl].sum(axis=0).astype(np.float32)
        return data
    
    def get_data(self, roi_list):
        return self.handler.get_data(roi_list)

    def load_meta(self):
        return get_metadata(self.fname)


if __name__ == '__main__':
    pass