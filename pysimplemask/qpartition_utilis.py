import numpy as np
import logging
import matplotlib.pyplot as plt
import h5py

logger = logging.getLogger(__name__)


def adjust_dynamic_range(arr):
    # adjust the dynamic range of the array to the maximum possible
    if arr.dtype in [np.float64, np.float32, np.float16]:
        return arr
    # convert to int if possible
    # some downstream applications using pytorch doesn't support uint16 and
    # uint32, uint64
    max_val = np.max(arr)
    if max_val <= 255:
        dtype = np.uint8
    elif max_val <= 2 ** 15 - 1:
        dtype = np.int16
    elif max_val <= 2 ** 31 - 1:
        dtype = np.int32
    else:
        dtype = np.int64
    return arr.astype(dtype)


def create_single_partition(map_name='q', xmap=None, mask=None, vbeg=None, 
                            vend=None, style='linear', n_bins=36,
                            map_unit='a.u'):
    if vbeg is None or vend is None:
        vbeg = np.nanmin(xmap[mask > 0])
        vend = np.nanmax(xmap[mask > 0])

    if np.sum((xmap >= vbeg) * (xmap <= vend)) == 0:
        vbeg, vend = vend, vbeg
        if np.sum((xmap >= vbeg) * (xmap <= vend)) == 0:
            logger.error(f'cannot find any pixel that in [{vbeg}, {vend}]')
            return None, None

    assert style in ('linear', 'logarithmic')
    if style == 'linear':
        vspan = np.linspace(vbeg, vend, n_bins + 1)
        vlist = (vspan[1:] + vspan[:-1]) / 2.0
    elif style == 'logarithmic':
        assert vbeg > 0, 'vbeg must > 0 when using logarithmic style'
        vspan = np.logspace(np.log10(vbeg), np.log10(vend), n_bins + 1)
        vlist = np.sqrt(vspan[1:] * vspan[:-1])
    partition = np.zeros_like(xmap, dtype=np.int64)

    mask_prev = xmap < vspan[0] 
    for m in range(n_bins):
        if m < n_bins - 1:
            mask_curr = xmap < vspan[m + 1]
        else:
            # include the last point;
            mask_curr = xmap <= vspan[m + 1]
        mask_roi = (~mask_prev) * mask_curr * mask
        partition[mask_roi == 1] = m + 1
        mask_prev = mask_curr
    counts = np.bincount(partition.ravel(), minlength=n_bins+1)[1:]
    result = {
        'partition': adjust_dynamic_range(partition),
        'vlist': vlist,
        'counts': counts,
        'map_name': [map_name],
        'map_unit': [map_unit],
        'map_bins': [n_bins]
    }
    return result


def combine_two_partitions(pt0_dict, pt1_dict):
    pt0 = pt0_dict['partition']
    pt1 = pt1_dict['partition']
    mask = pt0 * pt1 > 0

    nbins0 = pt0_dict['vlist'].size
    nbins1 = pt1_dict['vlist'].size 

    # pt_a, pt_b and pt_c start from 1
    pt_c = (pt0.astype(np.int64) - 1) * nbins1 + (pt1.astype(np.int64) - 1) + 1
    pt_c = np.clip(pt_c, a_min=0, a_max=None)
    pt_c *= mask
    minlength = nbins0 * nbins1 + 1
    counts = np.bincount(pt_c.ravel(),
                         minlength=minlength)[1:].reshape(nbins0, nbins1)

    vlist = np.zeros((nbins0, nbins1, 2))
    vlist[:, :, 0] = pt0_dict['vlist'][:, np.newaxis]
    vlist[:, :, 1] = pt1_dict['vlist']
    # vlist = vlist.reshape(-1, 2)

    result = {
        'partition': adjust_dynamic_range(pt_c),
        'vlist': vlist,
        'counts': counts,
        'map_name': pt0_dict['map_name'] + pt1_dict['map_name'],
        'map_bins': pt0_dict['map_bins'] + pt1_dict['map_bins'],
        'map_unit': pt0_dict['map_unit'] + pt1_dict['map_unit'],
    }
    return result 


def create_static_dynamic_partitions(sn=100, dn=10, **kwargs):
    # static
    static_p = create_single_partition(n_bins=sn, **kwargs)
    # dynamic
    dynamic_p = create_single_partition(n_bins=dn, **kwargs)
    return static_p, dynamic_p


def create_partitions(kwargs0, kwargs1=None):
    static_p, dynamic_p = create_static_dynamic_partitions(**kwargs0)
    if kwargs1 is not None:
        static_p1, dynamic_p1 = create_static_dynamic_partitions(**kwargs1)
        static_p = combine_two_partitions(static_p, static_p1)
        dynamic_p = combine_two_partitions(dynamic_p, dynamic_p1)
    return static_p, dynamic_p 


def test():
    fname = '/Users/mqichu/Documents/pysimplemask/tests/data/qmap_output_simplemask.hdf'
    with h5py.File(fname) as f:
        qmap = f['/data/Maps/q'][()]
        mask = f['/data/mask'][()]
        pmap = f['/data/Maps/phi'][()]
    pd0 = create_single_partition(qmap, mask, n_bins=17)
    # pd0 = create_single_partition(qmap, mask, n_bins=7, style='logarithmic')
    pd1 = create_single_partition(pmap, mask, n_bins=37)

    pd2 = combine_two_partitions(pd0, pd1)
    plt.imshow(pd2['partition'])
    plt.colorbar()
    plt.show()


def test2():
    fname = '/Users/mqichu/Documents/pysimplemask/tests/data/qmap_output_simplemask.hdf'
    with h5py.File(fname) as f:
        qmap = f['/data/Maps/q'][()]
        mask = f['/data/mask'][()]
        pmap = f['/data/Maps/phi'][()]
    kwargs0 = {'map_name': 'q', 'xmap': qmap, 'mask': mask, 'sn': 60, 'dn': 10, 'style': 'logarithmic'}
    kwargs1 = {'map_name': 'p', 'xmap': pmap, 'mask': mask, 'sn': 55, 'dn': 11, 'style': 'linear'}
    
    static_pd2, dynamic_pd2 = create_partitions(kwargs0, kwargs1)
    plt.imshow(static_pd2['partition'])
    plt.colorbar()
    plt.show()
    # plt.imshow(dynamic_pd2['partition'])
    # plt.colorbar()
    # plt.show()


if __name__ == '__main__':
    # test()
    test2()