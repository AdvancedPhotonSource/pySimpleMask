import numpy as np
import h5py
import hdf5plugin
import logging
import datetime
from ..file_reader import FileReader


logger = logging.getLogger(__name__)


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
        'bcx': '/entry/instrument/bluesky/metadata/bcx',
        'bcy': '/entry/instrument/bluesky/metadata/bcy',
        # 'xdim': '/entry/instrument/bluesky/metadata/xdim',
        # 'ydim': '/entry/instrument/bluesky/metadata/ydim'
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
    # delete this line once Pete fixed the metadata
    meta['pix_dim'] = 0.075e-3

    # scattering geometry
    meta['sg_type'] = 'transmission'
    meta['datetime'] = str(datetime.datetime.now())
    
    for key in ['det_dist', 'pix_dim']:
        meta[key] *= 1000
    for key in ['ccdx', 'ccdx0', 'ccdy', 'ccdy0']:
        meta.pop(key, None)

    return meta


class APS8IDIReader(FileReader):
    def __init__(self, fname, begin_idx=0, num_frames=-1) -> None:
        super(APS8IDIReader, self).__init__(fname)
        self.ftype = 'APS-8IDI-nexus'
        self.saxs = self.get_scattering(begin_idx, num_frames)
        self.meta = self.load_meta()
        self.meta['shape'] = self.saxs.shape
        self.shape = self.saxs.shape
        self.prepare_data()
    
    def get_scattering(self, begin_idx=0, num_frames=-1, block_size=32):
        with h5py.File(self.fname, 'r') as f:
            dset = f['/entry/data/data']
            if num_frames <= 0:
                num_frames= dset.shape[0]
            sl = slice(begin_idx, min(begin_idx + num_frames, dset.shape[0]))
            if sl.stop - sl.start <= block_size:
                data = dset[sl].sum(axis=0).astype(np.float32)
            else:
                data = 0
                for n in range(sl.start, sl.stop, block_size):
                    slt = slice(n, min(n + block_size, sl.stop))
                    data += dset[slt].astype(np.float32).sum(axis=0)
        return data
    
    def load_meta(self):
        fname = self.fname.replace('.h5', '.hdf')
        meta = get_metadata(fname)
        return meta


if __name__ == '__main__':
    pass