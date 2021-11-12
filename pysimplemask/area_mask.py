import os
import h5py
import hdf5plugin
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt


class MaskBase():
    def __init__(self, shape=(512, 1024)) -> None:
        self.shape = shape
        self.zero_loc = None
        self.mtype = 'base'
        self.enabled = True

    def set_enabled(self, flag=True):
        self.enabled = flag

    def get_mask(self):
        mask = np.ones(self.shape, dtype=np.bool)
        if self.zero_loc is not None:
            mask[tuple(self.zero_loc)] = 0
        return mask

    def combine_mask(self, mask):
        if self.zero_loc is not None and self.enabled:
            if mask is None:
                mask = self.get_mask()
            else:
                mask[tuple(self.zero_loc)] = 0
        return mask

    def show_mask(self):
        plt.imshow(self.get_mask(), vmin=0, vmax=1, cmap=plt.cm.jet)
        plt.show()


class MaskList(MaskBase):
    def __init__(self, shape=(512, 1024)) -> None:
        super().__init__(shape=shape)
        self.mtype = 'list'
        self.xylist = None

    def append_zero_pt(self, row, col):
        print(self.zero_loc.shape)
        self.zero_loc = np.append(self.zero_loc,
                                  np.array([row, col]).reshape(2, 1), axis=1)

    def evaluate(self, zero_loc=None):
        self.zero_loc = zero_loc


class MaskFile(MaskBase):
    def __init__(self, shape=(512, 1024), fname=None, **kwargs) -> None:
        super().__init__(shape=shape)
        self.mtype = 'file'

    def evaluate(self, fname=None, key=None):
        if not os.path.isfile(fname):
            return

        _, ext = os.path.splitext(fname)
        mask = None
        if ext in ['.hdf', '.h5', '.hdf5']:
            try:
                with h5py.File(fname, 'r') as f:
                    mask = f[key][()]
            except Exception:
                print('cannot read the hdf file, check path')
        elif ext in ['.tiff', '.tif']:
            mask = skio.imread(fname).astype(np.int)
        else:
            print('only support tif and hdf file.')

        if mask is None:
            self.zero_loc = None
            return

        if mask.shape != self.shape:
            mask = np.swapaxes(mask, 0, 1)

        assert mask.shape == self.shape
        mask = (mask <= 0)
        self.zero_loc = np.array(np.nonzero(mask))


class MaskThreshold(MaskBase):
    def __init__(self, shape=(512, 1024)) -> None:
        super().__init__(shape=shape)

    def evaluate(self,  saxs_log=None, low=0, high=1e8, scale='linear'):
        if scale == 'linear':
            low = np.log10(max(1e-12, low))
            high = np.log10(max(1e-12, high))
        mask = (saxs_log > low) * (saxs_log < high)
        mask = np.logical_not(mask)
        self.zero_loc = np.array(np.nonzero(mask))


class MaskArray(MaskBase):
    def __init__(self, shape=(512, 1024)) -> None:
        super().__init__(shape=shape)

    def evaluate(self, arr=None):
        if arr is not None:
            self.zero_loc = np.array(np.nonzero(arr))


class MaskAssemble():
    def __init__(self, shape=(128, 128), saxs_log=None) -> None:
        self.workers = {
            'mask_blemish': MaskFile(shape),
            'mask_file': MaskFile(shape),
            'mask_threshold': MaskThreshold(shape),
            'mask_list': MaskList(shape),
            'mask_draw': MaskArray(shape),
            'mask_outlier': MaskArray(shape),
        }
        self.saxs_log = saxs_log
    
    def enable(self, target, flag=True):
        self.workers[target].set_enabled(flag)

    def evaluate(self, target, **kwargs):
        if target != 'mask_threshold':
            self.workers[target].evaluate(**kwargs)
        else:
            self.workers[target].evaluate(self.saxs_log, **kwargs)

    def get_one_mask(self, target):
        return self.workers[target].get_mask()

    def get_mask(self):
        mask = None
        for key in self.workers.keys():
            mask = self.workers[key].combine_mask(mask)
        return mask

    def show_mask(self):
        plt.imshow(self.get_mask())
        plt.show()


def test_01(shape=(64, 64)):
    a = MaskList(shape)
    a.evaluate(np.array([[1, 2, 3], [1, 2, 3]], dtype=np.int64))
    a.append_zero_pt(10, 20)
    a.show_mask()


def test_02(shape=(512, 1024)):
    a = MaskFile(shape)
    a.evaluate(fname='test_qmap.h5', key='/data/mask')
    a.show_mask()


def test_03(shape=(512, 1024)):
    a = MaskArray(shape)
    mask = np.random.randint(0, 2, size=shape)
    a.evaluate(mask)
    a.show_mask()


def test_04(shape=(512, 1024)):
    ma = MaskAssemble(shape)
    ma.evaluate('blemish_file', fname='test_qmap.h5', key='/data/mask')
    ma.evaluate('mask_list', zero_loc=np.array(
        [[1, 2, 3], [1, 2, 3]], dtype=np.int64))
    ma.show_mask()


# test_03()
