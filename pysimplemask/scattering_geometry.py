import numpy as np


def to_single_precision(qmap):
    for k, v in qmap.items():
        if v.dtype == np.float64:
            qmap[k] = v.astype(np.float32)
        elif v.dtype in [np.int64, np.uint64]:
            qmap[k] = v.astype(np.uint32)
    return qmap


def get_scattering_geometry(sg_type='transmission', metadata=None):
    if sg_type == 'transmission':
        return compute_qmaps_transmission(metadata)
    elif sg_type == 'reflection':
        return compute_qmaps_reflection(metadata)
    else:
        raise ValueError(f'sg_type [{sg_type}] not supported.')


def compute_qmaps_transmission(meta):
    k0 = 2 * np.pi / (12.398 / meta['energy'])
    v = np.arange(meta['shape'][0], dtype=np.uint32) - meta['bcy']
    h = np.arange(meta['shape'][1], dtype=np.uint32) - meta['bcx']
    vg, hg = np.meshgrid(v, h, indexing='ij')

    r = np.sqrt(vg * vg + hg * hg) * meta['pix_dim']
    # phi = np.arctan2(vg, hg)
    # to be compatible with matlab xpcs-gui; phi = 0 starts at 6 clock
    # and it goes clockwise;
    phi = np.arctan2(hg, vg)
    phi[phi < 0] = phi[phi < 0] + np.pi * 2.0
    phi = np.max(phi) - phi     # make it clockwise

    alpha = np.arctan(r / meta['det_dist'])
    qr = np.sin(alpha) * k0
    qr = 2 * np.sin(alpha / 2) * k0
    qx = qr * np.cos(phi)
    qy = qr * np.sin(phi)

    # keep phi and q as np.float64 to keep the precision.
    qmap = {
        'q': qr,
        'phi': np.rad2deg(phi),
        'qx': qx,
        'qy': qy,
        'alpha': np.rad2deg(alpha),
        'x': hg,
        'y': vg
    }

    qmap_unit = {
        'q': '1/Å',
        'phi': 'deg',
        'qx': '1/Å',
        'qy': '1/Å',
        'alpha': 'deg',
        'x': 'pixel',
        'y': 'pixel' 
    }
    qmap = to_single_precision(qmap)
    return qmap, qmap_unit


def compute_qmaps_reflection(meta):
    k0 = 2 * np.pi / (12.398 / meta['energy'])
    v = np.arange(meta['shape'][0], dtype=np.uint32) - meta['bcy']
    h = np.arange(meta['shape'][1], dtype=np.uint32) - meta['bcx']

    v = v * (-1)
    vg, hg = np.meshgrid(v, h, indexing='ij')

    phi = np.arctan2(hg, vg)
    # phi[phi < 0] = phi[phi < 0] + np.pi * 2.0
    phi = np.pi / 2.0 - phi     # make it clockwise

    alpha_i = np.deg2rad(meta['alpha_i'])
    alpha_f = np.arctan(vg * meta['pix_dim'] / meta['det_dist'])
    tth = np.arctan(hg * meta['pix_dim'] / meta['det_dist'])

    qx = k0 * (np.cos(alpha_f) * np.cos(tth) - np.cos(alpha_i))
    qy = k0 * (np.cos(alpha_f) * np.sin(tth))
    qz = k0 * (np.sin(alpha_i) + np.sin(alpha_f))

    qr = np.hypot(qx, qy)
    q = np.hypot(qr, qz)
    chi = np.arccos(qz / q)

    # keep phi and q as np.float64 to keep the precision.
    qmap = {
        'q': q,
        'phi': np.rad2deg(phi),
        'qx': qx,
        'qy': qy,
        'qz': qz,
        'tth': tth,
        'alpha_f': np.rad2deg(alpha_f),
        'chi': np.rad2deg(chi),
        'x': hg,
        'y': vg
    }

    qmap_unit = {
        'q': '1/Å',
        'phi': 'deg',
        'qx': '1/Å',
        'qy': '1/Å',
        'qz': '1/Å',
        'tth': 'deg',
        'alpha_f': 'deg',
        'chi': 'deg',
        'x': 'pixel',
        'y': 'pixel'
    }
    qmap = to_single_precision(qmap)

    return qmap, qmap_unit