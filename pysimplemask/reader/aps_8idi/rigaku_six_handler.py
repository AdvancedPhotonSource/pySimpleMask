import numpy as np
from scipy.sparse import coo_matrix
import logging
from .xpcs_dataset import XpcsDataset
from .rigaku_handler import RigakuDataset
import glob


logger = logging.getLogger(__name__)


def convert_sparse(a):
    output = np.zeros(shape=(3, a.size), dtype=np.uint32)
    # index
    output[0] = ((a >> 16) & (2 ** 21 - 1)).astype(np.uint32)
    # frame
    output[1] = (a >> 40).astype(np.uint32)
    # count
    output[2] = (a & (2 ** 12 - 1)).astype(np.uint8)
    return output


class RigakuSixDataset(XpcsDataset):
    """
    Parameters
    ----------
    filename: string
        path to .imm file
    """
    def __init__(self, *args, dtype=np.uint8, **kwargs):
        super(RigakuSixDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "Rigaku 64bit Binary"
        self.is_sparse = True
        self.dtype = np.uint8
        self.dataset_list = self.read_data()
        self.det_size = (1536 + 40, 2048 + 20)
    
    def read_data(self):
        fname_list = glob.glob(self.fname[:-5] + '*.bin')
        dataset_list = []
        assert len(fname_list) == 6, 'rigaku_six must have six bin files'
        for f in fname_list:
            dataset_list.append(RigakuDataset(f))
        return dataset_list
    
    def stitch_modules(self, data_list):
        canvas = np.zeros(self.det_size, dtype=np.uint32)
        for n in range(3):
            offset_v = n * 20 if n > 0 else 0
            sl_v = slice(n * 512 + offset_v, (n + 1) * 512 + offset_v)
            for m in range(2):
                offset_h = m * 20 if m > 0 else 0
                sl_h = slice(m * 1024 + offset_h, (m + 1) * 1024 + offset_h)
                canvas[sl_v, sl_h] = data_list[n * 2 + m]
        return canvas
    
    def get_scattering(self, *args, **kwargs):
        data_list = [
            dset.get_scattering(*args, **kwargs) for dset in self.dataset_list
        ]
        scat = self.stitch_modules(data_list)
        return scat

    def __getbatch__(self, idx):
        pass


def test():
    fname = '../../tests/data/E0135_rigaku_six/E0135_La0p65_L2_013C_att04_Rq0_00001_part2.bin'
    ds = RigakuSixDataset(fname)
    #     print(n, ds[n].shape)


if __name__ == '__main__':
    test()
