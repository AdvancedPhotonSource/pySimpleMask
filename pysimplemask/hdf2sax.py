import h5py


def hdf2saxs(fname, beg_idx=0, num_frames=-1):
    '''
    read a xpcs hdf file, collapse the time series to get a two dimensional
    small angle scattering pattern.
    Args:
        fname: string. filename of the hdf/h5 file; the file must have
            '/entry/data/data'
        beg_idx: integer: beginning index. It is used to skip the frames in
            in the beginning in case the detector was not ready. default is 0.
        num_frames: integer: number of frames to average. This is useful if
            the file has too many frames which takes too much time to load.
            default is -1, which means it will use all frames.
    Return:
        a 2d numpy array for the saxs pattern.
    Example:
        fname = 'xyz.h5'
        saxs = hdf2saxs(fname, num_frames=100)
        print(y.shape)
    '''
    with h5py.File(fname, 'r') as f:
        x = f['/entry/data/data']
        if num_frames < 0:
            end_idx = x.shape[0]
        else:
            end_idx = min(x.shape[0], beg_idx + num_frames)
        y = x[beg_idx: end_idx].mean(axis=0).astype(x.dtype)
    return y


def test01():
    fname = ('/Users/jeffr/Desktop/suli_fall_2021/D093_Lut_20C_att02_Lq0_003/D093_Lut_20C_att02_Lq0_003_001.h5')
# C:\Users\jeffr\Desktop\suli_fall_2021\D093_Lut_20C_att02_Lq0_003

    y = hdf2saxs(fname, num_frames=100)
    
    print(y)
    assert y.ndim == 2
    # print(y)
    
    


if __name__ == "__main__":
    test01()