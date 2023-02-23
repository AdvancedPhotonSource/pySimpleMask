import matplotlib.pyplot as plt
import numpy as np
from aps_reader import ImmDataset, RigakuDataset, HdfDataset

fname = '/mnt/c/Users/mqichu/Documents/local_dev/pysimplemask/tests/data/H432_OH_100_025C_att05_001/H432_OH_100_025C_att05_001_00001-01000.imm'
dset = ImmDataset(fname, batch_size=128)
print(dset.frame_num_raw)

# fname = '/mnt/c/Users/mqichu/Documents/local_dev/pysimplemask/tests/data/asymmetric_analysis/D0201_A5_normalcart_normalsf_redo_37C_037C_att04_001/D0201_A5_normalcart_normalsf_redo_37C_037C_att04_001_001.h5'
# dset = HdfDataset(fname, batch_size=128)

# fname = '/mnt/c/Users/mqichu/Documents/local_dev/pysimplemask/tests/data/E0135_La0p65_L2_013C_att04_Rq0_00001/E0135_La0p65_L2_013C_att04_Rq0_00001.bin'
# dset = RigakuDataset(fname, batch_size=128)

roi1 = ([1, 2, 3, 4], [5, 6, 7, 8])
roi2 = ([11, 12, 13, 14], [15, 16, 17, 18])
roi_list = [roi1, roi2]


# saxs = dset.get_scattering(num_frames=-1, begin_idx=0)

data = dset.get_data(roi_list)
for x in data:
    print(x.shape, type(x))
# print('max value', np.max(saxs))
# min_val = np.min(saxs[saxs > 0])
# saxs[saxs <= 0] = min_val
# plt.imshow(np.log10(saxs))
# plt.show()
