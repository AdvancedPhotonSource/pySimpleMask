import numpy as np
from scipy.sparse import coo_matrix
import logging
from .xpcs_dataset import XpcsDataset
from .rigaku_handler import RigakuDataset
import glob
import os


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


class Rigaku3MDataset(XpcsDataset):
    """
    Parameters
    ----------
    filename: string
        path to Rigaku3M binary.000 file
    """
    def __init__(self, *args, dtype=np.uint8, gap=(70, 52), layout=(3, 2),
                 **kwargs):
        super(Rigaku3MDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "Rigaku 64bit Binary"
        self.is_sparse = True
        self.dtype = np.uint8
        self.container = self.get_modules(dtype, **kwargs)
        self.gap = gap
        self.layout = layout
        self.update_info()
    
    def get_modules(self, dtype, **kwargs):
        # this is the order for the 6 modules
        index_list = (5, 0, 4, 1, 3, 2)
        flist = [self.fname[:-3] + f'00{n}' for n in index_list]
        for f in flist:
            assert os.path.isfile(f)
        return [RigakuDataset(f, dtype=dtype, **kwargs) for f in flist]
    
    def update_info(self):
        frame_num = [x.frame_num for x in self.container]
        assert len(list(set(frame_num))) == 1, 'check frame numbers'
        self.update_batch_info(frame_num[0])
        shape_one = self.container[0].det_size 
        shape = [shape_one[n] * self.layout[n] + self.gap[n] * (self.layout[n] - 1) for n in range(2)]
        self.det_size = shape
    
    def patch_data(self, scat_list):
        gap = self.gap
        layout = self.layout

        shape_one = scat_list[0].shape
        shape = self.det_size

        canvas = np.zeros(shape, dtype=scat_list[0].dtype)
        #  canvas[:, :] = np.nan
        for row in range(layout[0]):
            st_v = row * (shape_one[0] + gap[0])
            sl_v = slice(st_v, st_v + shape_one[0])
            for col in range(layout[1]):
                st_h = col * (shape_one[1] + gap[1])
                sl_h = slice(st_h, st_h + shape_one[1])
                canvas[sl_v, sl_h] = scat_list[row * layout[1] + col]
        return canvas
    
    def get_scattering(self, num_frames=-1, begin_idx=0):
        scat_list = []
        for module in self.container:
            scat_list.append(module.get_scattering(num_frames=num_frames,
                                                   begin_idx=begin_idx))
        saxs = self.patch_data(scat_list)
        return saxs

    def __getbatch__(self, idx):
        # frame begin and end
        beg, end, size = self.get_raw_index(idx)
        x = np.zeros((size, self.pixel_num), dtype=np.uint8)

        if self.stride == 1:
            # the data is continuous in RAM; convert by batch
            sla, slb = self.mem_addr[beg], self.mem_addr[end]
            a, b = sla.start, slb.start
            x[self.ifc[1][a:b] - beg, self.ifc[0][a:b]] = self.ifc[2][a:b]
        else:
            # the data is discrete in RAM; convert frame by frame
            for n, idx in enumerate(np.arange(beg, end, self.stride)):
                sl = self.mem_addr[idx]
                x[n, self.ifc[0][sl]] = self.ifc[2][sl]
        return x


def test():
    fname = "../../../tests/data/verify_circular_correlation/F091_D100_Capillary_Post_att00_Lq0_Rq0_00001/F091_D100_Capillary_Post_att00_Lq0_Rq0_00001.bin"
    ds = RigakuDataset(fname)


if __name__ == '__main__':
    test()
