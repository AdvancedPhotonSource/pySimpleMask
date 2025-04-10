from functools import lru_cache
import numpy as np


@lru_cache(maxsize=128)
def create_transmission_qmap(
    energy_kev, center_vh, shape, pix_dim_meter, det_dist_meter
):
    k0 = 2 * np.pi / (12.398 / energy_kev)
    v = np.arange(shape[0], dtype=np.uint32) - center_vh[0]
    h = np.arange(shape[1], dtype=np.uint32) - center_vh[1]
    vg, hg = np.meshgrid(v, h, indexing="ij")

    r = np.hypot(vg * pix_dim_meter[0], hg * pix_dim_meter[1])  # unit meter
    phi = np.arctan2(vg, hg) * (-1)
    alpha = np.arctan(r / det_dist_meter)

    qr = np.sin(alpha) * k0
    # qr = 2 * np.sin(alpha / 2) * k0
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
