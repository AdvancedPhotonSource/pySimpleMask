from pysimplemask.reader.aps_reader import Rigaku3MDataset
import matplotlib.pyplot as plt

fname = '/home/8ididata/Rigaku3M_Test_Data/SparsificationData2/Div1_000001.bin.000'
dset = Rigaku3MDataset(fname, batch_size=1000)
print(dset.det_size)
plt.imshow(dset.get_scattering())
plt.show()

