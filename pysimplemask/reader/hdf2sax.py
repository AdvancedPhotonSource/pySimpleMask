import h5py
import hdf5plugin
import numpy as np


def hdf2saxs(fname, beg_idx=0, num_frames=-1, key='/entry/data/data',
             max_ram=1.0):
    '''
    read a xpcs hdf file, collapse the time series to get a two dimensional
    small angle scattering pattern.
    Args:
        fname: string. filename of the hdf/h5 file;
        beg_idx: integer: beginning index. It is used to skip the frames in
            in the beginning in case the detector was not ready. default is 0.
        num_frames: integer: number of frames to average. This is useful if
            the file has too many frames which takes too much time to load.
            default is -1, which means it will use all frames.
        key: the field name for the data stored in the HDF file
        max_ram: the maximal ram usage for reading the data, uint is GB
    Return:
        a 2d numpy array for the saxs pattern.
    Example:
        fname = 'xyz.h5'
        saxs = hdf2saxs(fname, num_frames=100)
        print(y.shape)
    '''
    with h5py.File(fname, 'r') as f:
        if key not in f:
            raise IOError(f'key [{key}] does not exist.')
        x = f[key]

        frame_mem_size = x.shape[1] * x.shape[2] * x[0, 0, 0].itemsize

        if num_frames < 0:
            num_frames = x.shape[0] - beg_idx
        # apply memory restriction
        num_frames = min(num_frames, int(max_ram * 1024 ** 3 / frame_mem_size))

        end_idx = min(x.shape[0], beg_idx + num_frames)
        y = x[beg_idx: end_idx].astype(np.float32)
        y = np.mean(y, axis=0)

    return y


def test01():
    fname = ('/Users/mqichu/Downloads/simulation_0010k_sparse_0.005/simulation_0010k_sparse_0.005.h5')
# C:\Users\jeffr\Desktop\suli_fall_2021\D093_Lut_20C_att02_Lq0_003

    y = hdf2saxs(fname, num_frames=100)

    print(y)
    assert y.ndim == 2
    print(np.max(y))


if __name__ == "__main__":
    test01()
