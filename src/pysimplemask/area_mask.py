import os
import h5py
import hdf5plugin
import numpy as np
import skimage.io as skio
import matplotlib.pyplot as plt
import logging


logger = logging.getLogger(__name__)


def create_qring(qmin, qmax, pmin, pmax, qnum=1, flag_const_width=True):
    qrings = []
    if qmin > qmax:
        qmin, qmax = qmax, qmin
    qcen = (qmin + qmax) / 2.0
    qhalf = (qmax - qmin) / 2.0

    for n in range(1, qnum + 1):
        if flag_const_width:
            low = qcen * n - qhalf
            high = qcen * n + qhalf
        else:
            low = (qcen - qhalf) * n
            high = (qcen + qhalf) * n
        qrings.append((low, high, pmin, pmax))
    return qrings


class MaskBase():
    def __init__(self, shape=(512, 1024)) -> None:
        self.shape = shape
        self.zero_loc = None
        self.mtype = 'base'
        self.qrings = []

    def describe(self):
        if self.zero_loc is None:
            return 'Mask is not initialized'
        bad_num = len(self.zero_loc[0])
        total_num = self.shape[0] * self.shape[1]
        ratio = bad_num / total_num * 100.0
        msg = f'{self.mtype}: bad_pixel: {bad_num}/{ratio:0.3f}%'
        return msg

    def get_mask(self):
        mask = np.ones(self.shape, dtype=bool)
        if self.zero_loc is not None:
            mask[tuple(self.zero_loc)] = 0
        return mask

    def combine_mask(self, mask):
        if self.zero_loc is not None:
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
                logger.error('cannot read the hdf file, check path')
        elif ext in ['.tiff', '.tif']:
            mask = skio.imread(fname).astype(np.int64)
        else:
            logger.error(f'MaskFile only support tif and hdf file. found {fname}')

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


class MaskQring(MaskBase):
    """
    use a ring on the qmap to define the mask
    """

    def __init__(self, shape=(512, 1024)) -> None:
        super().__init__(shape=shape)
        self.qrings = []

    def evaluate(self, qmap, pmap, qrings=None):
        mask = np.zeros_like(qmap, dtype=bool)
        if qrings is None:
            return

        for n in range(len(qrings)):
            pmap_loc = np.copy(pmap)

            qmin, qmax, pmin, pmax = qrings[n]
            if qmin > qmax:
                qmin, qmax = qmax, qmin

            qroi = np.logical_and((qmap >= qmin), (qmap < qmax))
            if pmin > pmax:
                pmax += 360.0
                pmap_loc[pmap_loc < pmin] += 360.0

            proi = np.logical_and((pmap_loc >= pmin), (pmap_loc < pmax))

            mask[qroi * proi] = 1

        mask = np.logical_not(mask)
        self.zero_loc = np.array(np.nonzero(mask))
        self.qrings = qrings

    def get_qrings(self):
        return self.qrings.copy()


class MaskArray(MaskBase):
    def __init__(self, shape=(512, 1024)) -> None:
        super().__init__(shape=shape)

    def evaluate(self, arr=None):
        if arr is not None:
            self.zero_loc = np.array(np.nonzero(arr))


class MaskAssemble():
    def __init__(self, shape=(128, 128), saxs_log=None, qmap=None, pmap=None,
            ) -> None:
        self.workers = {
            'mask_blemish': MaskFile(shape),
            'mask_file': MaskFile(shape),
            'mask_threshold': MaskThreshold(shape),
            'mask_list': MaskList(shape),
            'mask_draw': MaskArray(shape),
            'mask_outlier': MaskList(shape),
            'mask_qring': MaskQring(shape)
        }
        self.shape = shape
        self.saxs_log = saxs_log
        self.qmap = qmap
        self.pmap = pmap
        self.mask_record = [np.ones_like(qmap, dtype=bool)]
        self.mask_ptr = 0
        # 0: no mask; 1: apply the default mask
        self.mask_ptr_min = 0
    
    def update_qmap(self, qmap_all):
        self.qmap = qmap_all['q']
        self.pmap = qmap_all['phi']
    
    def apply_default_mask(self,
                           default_blemish_path="~/Documents/Miaoqi/areaDetectorBlemish"):
        basename = os.path.expanduser(default_blemish_path)

        if tuple(self.shape) == (1813, 1558):
            fname = os.path.join(basename, 'lambda2M_latest_blemish')
        elif tuple(self.shape) == (2162, 2068):
            fname = os.path.join(basename, 'eiger4M_latest_blemish')
        elif tuple(self.shape) == (1676, 2100):
            fname = os.path.join(basename, 'rigaku3M_latest_blemish')
        else:
            logger.warning('detector shape/type not supported')
            self.mask_ptr_min = 0
            return self.get_mask()

        if os.path.isfile(fname):  # returns True for symbolic links too
            realpath = os.path.realpath(fname)
            logger.info(f'apply default blemish {realpath}')
            ext = os.path.splitext(realpath)[-1]
            if ext in ('.tif', '.tiff'):
                self.evaluate('mask_blemish', fname=realpath)
            elif ext in ('.h5', '.hdf5', '.hdf'):
                self.evaluate('mask_blemish', fname=realpath, key='/data/mask')
            else:
                logger.warning(f'default blemish {fname} not supported')
                self.mask_ptr_min = 0
                return self.get_mask()
            self.apply('mask_blemish')
            self.mask_ptr_min = 1
        else:
            logger.warning(f'default blemish {fname} not found')
        return self.get_mask()

    def apply(self, target):
        if target is None:
            return self.get_mask()

        mask = self.get_one_mask(target)
        mask = np.logical_and(self.get_mask(), mask)
        if not np.allclose(self.mask_record[-1], mask):
            while len(self.mask_record) > self.mask_ptr + 1:
                self.mask_record.pop()
            # len(self.mask_record) == self.mask_ptr + 1
            self.mask_record.append(mask)
            self.mask_ptr += 1
        return mask

    def evaluate(self, target, **kwargs):
        if target == 'mask_threshold':
            self.workers[target].evaluate(self.saxs_log, **kwargs)
        elif target == 'mask_qring':
            self.workers[target].evaluate(self.qmap, self.pmap, **kwargs)
        else:
            self.workers[target].evaluate(**kwargs)

        return self.workers[target].describe()
    
    def redo_undo(self, action='redo'):
        if action == 'undo':
            if self.mask_ptr > self.mask_ptr_min:
                self.mask_ptr -= 1
        elif action == 'redo': 
            if self.mask_ptr < len(self.mask_record) - 1:
                self.mask_ptr += 1
        elif action == 'reset':
            # if 1 + mask_ptr_min = 2: keep the default mask
            # if 1 + mask_ptr_min = 1: no default mask
            while len(self.mask_record) > 1 + self.mask_ptr_min:
                self.mask_record.pop()
            self.mask_ptr = self.mask_ptr_min
        
    def get_one_mask(self, target):
        return self.workers[target].get_mask()

    def get_mask(self):
        # get the current combined mask
        mask = self.mask_record[self.mask_ptr]
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
if __name__ == '__main__':
    test_01()
