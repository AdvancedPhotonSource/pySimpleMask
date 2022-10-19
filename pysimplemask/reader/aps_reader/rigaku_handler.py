import numpy as np
import logging
from .xpcs_dataset import XpcsDataset


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


class RigakuDataset(XpcsDataset):
    """
    Parameters
    ----------
    filename: string
        path to .imm file
    """
    def __init__(self, *args, dtype=np.uint8, **kwargs):
        super(RigakuDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "Rigaku 64bit Binary"
        self.is_sparse = True
        self.dtype = np.uint8
        self.ifc, self.mem_addr = self.read_data()

    def read_data(self):
        with open(self.fname, 'r') as f:
            a = np.fromfile(f, dtype=np.uint64)
            d = convert_sparse(a)
            index, frame, count = d[0], d[1], d[2]
            frame_num = frame[-1] + 1

        # update frame_num and batch_num
        self.update_batch_info(frame_num)

        all_index = np.arange(0, self.frame_num_raw + 2)
        all_index[-1] = self.frame_num_raw
        # build an index map for all indexes
        beg = np.searchsorted(frame, all_index[:-1])
        end = np.searchsorted(frame, all_index[1:])
        mem_addr = [slice(a, b) for a, b in zip(beg, end)]
        return (index, frame, count), mem_addr

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
    fname = ("/home/beams/MQICHU/ramdisk/O018_Silica_D100_att0_Rq0_00001/"
             "O018_Silica_D100_att0_Rq0_00001.bin")
    ds = RigakuDataset(fname)
    # for n in range(len(ds)):
    #     print(n, ds[n].shape)


if __name__ == '__main__':
    test()
