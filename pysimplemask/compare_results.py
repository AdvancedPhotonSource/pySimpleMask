import numpy as np
import matplotlib.pyplot as plt
import h5py

f1 = '../tests/data/qmap_output_simplemask.hdf'
f2 = '../tests/data/jeffrey_GUI_test.h5'


def get_value(f, key):
    with h5py.File(f, 'r') as f:
        return np.squeeze(f[key][()])



def plot_1d(v1, v2):
    print('v1', np.min(v1), np.max(v1), np.mean(v1), v1.shape)
    print('v2', np.min(v2), np.max(v2), np.mean(v2), v2.shape)

    fig, axes = plt.subplots(3, 1)
    axes[0].plot(v1)
    axes[0].set_title(f1)

    axes[1].plot(v2)
    axes[1].set_title(f2)

    axes[2].plot(v1 / v2)
    plt.show()



def plot_2d(v1, v2, title=None):
    print('v1', np.min(v1), np.max(v1), np.mean(v1), v1.shape)
    print('v2', np.min(v2), np.max(v2), np.mean(v2), v2.shape)
    print('diff', np.sum(np.abs(v1 * 1.0 - v2 * 1.0)))
    diff = (v1 != v2)
    print(np.nonzero(diff))

    fig, axes = plt.subplots(3, 1, figsize=(9, 9))
    im0 = axes[0].imshow(v1)
    axes[0].set_title(f1)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(v2)
    axes[1].set_title(f2)
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(v1 * 1.0 - v2 * 1.0)
    fig.colorbar(im2, ax=axes[2])
    if title is not None:
        fig.suptitle(title)
    plt.show()


def compare(key='/data/Maps/q'):
    v1 = get_value(f1, key)
    v2 = get_value(f2, key)
    if v1.ndim == 1:
        plot_1d(v1, v2)
    elif v1.ndim == 2:
        plot_2d(v1, v2, key)


compare(key='/data/Maps/q')
compare(key='/data/dynamicMap')
compare(key='/data/staticMap')
compare(key='/data/dqval')
