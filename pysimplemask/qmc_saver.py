import numpy as np
import h5py
import os
import skimage.io as skio


def save_qmc(*args, method='nexus'):
    if method == 'nexus':
        save_as_nexus(*args)
    elif method == 'numpy':
        save_as_numpy(*args)
    elif method == 'tensor':
        save_as_tensor(*args)
    elif method == 'txt_img':
        save_as_txt_img(*args)


def save_as_nexus(save_fname, partition_info,
                  prefix='/entry/instrument/masks'):
    
    if not save_fname.endswith(('.h5', '.hdf', '.nx')):
        save_fname = save_fname + '.hdf'

    with h5py.File(save_fname, 'w') as hf:
        if prefix in hf:
            del hf[prefix]

        group = hf.create_group(prefix)
        for key, val in partition_info.items():
            full_key = '/'.join([prefix, key])
            if key in ('mask', 'dynamic_roi_map', 'static_roi_map'):
                compression = 'gzip'
            else:
                compression = None
            group.create_dataset(full_key, data=val, compression=compression)


def save_as_numpy(save_fname, partition_info):
    if not save_fname.endswith('.npz'):
        save_fname = save_fname + '.npz'
    # np.savez(save_fname, **partition_info)
    np.savez_compressed(save_fname, **partition_info)


def save_as_tensor(save_fname, partition_info):
    if not save_fname.endswith('.pt'):
        save_fname = save_fname + '.pt'
    import torch
    torch.save(partition_info, save_fname)


def save_as_txt_img(save_fname, partition_info):
    folder = os.path.splitext(save_fname)[0]
    if not os.path.isdir(folder):
        os.mkdir(folder)
    
    for key, val in partition_info.items():
        fname = os.path.join(folder, key + '.txt')
        if key == 'mask':
            fname = fname[:-4] + '.tif'
            skio.imsave(fname, val)
        else:
            np.savetxt(fname, val)