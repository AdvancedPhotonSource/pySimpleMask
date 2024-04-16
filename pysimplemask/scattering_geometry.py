import numpy as np



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

    phi = np.rad2deg(phi)

    # keep phi and q as np.float64 to keep the precision.
    qmap = {
        'phi': phi,
        'alpha': alpha.astype(np.float32),
        'q': qr,
        'qx': qx.astype(np.float32),
        'qy': qy.astype(np.float32)
    }

    return qmap



def compute_qmaps_reflection(meta):
    k0 = 2 * np.pi / (12.398 / meta['energy'])
    v = np.arange(meta['shape'][0], dtype=np.uint32) - meta['bcy']
    h = np.arange(meta['shape'][1], dtype=np.uint32) - meta['bcx']

    v *= meta['pixel_size'] * (-1)
    h *= meta['pixel_size']

    vg, hg = np.meshgrid(v, h, indexing='ij')

    phi = np.arctan2(hg, vg)
    phi[phi < 0] = phi[phi < 0] + np.pi * 2.0
    phi = np.max(phi) - phi     # make it clockwise

    alpha_i = np.deg2rad(meta['alpha_i_deg'])
    alpha_f = np.arctan(vg / meta['det_dist'])
    tth = np.arctan(hg/ meta['det_dist'])

    qx = k0 * (np.cos(alpha_f) * np.cos(tth) - np.cos(alpha_i))
    qy = k0 * (np.cos(alpha_f) * np.sin(tth))
    qz = k0 * (np.sin(alpha_i) + np.sin(alpha_f))

    qr = np.hypot(qx, qy)
    q = np.hypot(qr, qz)
    phi = np.rad2deg(phi)
    chi = np.rad2deg(np.arccos(qz / q))

    # keep phi and q as np.float64 to keep the precision.
    qmap = {
        'q': q,
        'phi': phi,
        'qx': qx.astype(np.float32),
        'qy': qy.astype(np.float32),
        'qz': qz.astype(np.float32),
        'tth': tth.astype(np.float32),
        'alpha_f': alpha_f.astype(np.float32),
        'chi': chi
    }

    return qmap