import numpy as np

E2KCONST = 12.39841984


def compute_transmission_qmap(energy, center, shape, pix_dim, det_dist):
    k0 = 2 * np.pi / (E2KCONST / energy)
    v = np.arange(shape[0], dtype=np.uint32) - center[0]
    h = np.arange(shape[1], dtype=np.uint32) - center[1]
    vg, hg = np.meshgrid(v, h, indexing="ij")
    r = np.hypot(vg, hg) * pix_dim
    phi = np.arctan2(vg, hg) * (-1)
    alpha = np.arctan(r / det_dist)

    qr = np.sin(alpha) * k0
    qx = qr * np.cos(phi)
    qy = qr * np.sin(phi)
    phi = np.rad2deg(phi)

    # keep phi and q as np.float64 to keep the precision.
    qmap = {
        "phi": phi,
        "alpha": alpha.astype(np.float32),
        "q": qr,
        "qx": qx.astype(np.float32),
        "qy": qy.astype(np.float32),
        "x": hg,
        "y": vg,
    }

    qmap_unit = {
        "phi": "deg",
        "alpha": "deg",
        "q": "Å⁻¹",
        "qx": "Å⁻¹",
        "qy": "Å⁻¹",
        "x": "pixel",
        "y": "pixel",
    }
    return qmap, qmap_unit


def compute_reflection_qmap(energy, center, pix_dim, det_dist):
    pass
