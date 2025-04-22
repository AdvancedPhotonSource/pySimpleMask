from functools import lru_cache

import numpy as np

E2KCONST = 12.39841984


def compute_qmap(stype, metadata):
    if stype == "Transmission":
        return compute_transmission_qmap(
            metadata["energy"],
            (metadata["bcy"], metadata["bcx"]),
            metadata["shape"],
            metadata["pix_dim"],
            metadata["det_dist"],
        )
    elif stype == "Reflection":
        print(metadata)
        return compute_reflection_qmap(
            metadata["energy"],
            (metadata["bcy"], metadata["bcx"]),
            metadata["shape"],
            metadata["pix_dim"],
            metadata["det_dist"],
            alpha_i_deg=metadata["alpha_i_deg"],
        )


@lru_cache(maxsize=128)
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
        "TTH": alpha.astype(np.float32),
        "q": qr,
        "qx": qx.astype(np.float32),
        "qy": qy.astype(np.float32),
        "x": hg,
        "y": vg,
    }

    qmap_unit = {
        "phi": "deg",
        "TTH": "deg",
        "q": "Å⁻¹",
        "qx": "Å⁻¹",
        "qy": "Å⁻¹",
        "x": "pixel",
        "y": "pixel",
    }
    return qmap, qmap_unit


@lru_cache(maxsize=128)
def compute_reflection_qmap(energy, center, shape, pix_dim, det_dist, alpha_i_deg=0.14):
    k0 = 2 * np.pi / (E2KCONST / energy)
    v = np.arange(shape[0], dtype=np.uint32) - center[0]
    h = np.arange(shape[1], dtype=np.uint32) - center[1]
    vg, hg = np.meshgrid(v, h, indexing="ij")

    r = np.hypot(vg, hg) * pix_dim
    phi = np.arctan2(vg, hg) * (-1)
    TTH = np.arctan(r / det_dist)

    alpha_i = np.deg2rad(alpha_i_deg)
    alpha_f = np.arctan(vg * (-1 * pix_dim) / det_dist) - 2 * alpha_i
    tth = np.arctan(hg * pix_dim / det_dist)

    qx = k0 * (np.cos(alpha_f) * np.cos(tth) - np.cos(alpha_i))
    qy = k0 * (np.cos(alpha_f) * np.sin(tth))
    qz = k0 * (np.sin(alpha_i) + np.sin(alpha_f))
    qr = np.hypot(qx, qy)
    q = np.hypot(qr, qz)

    qmap = {
        "phi": phi,
        "TTH": TTH,
        "tth": tth,
        "alpha_f": alpha_f,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qr": qr,
        "q": q,
        "x": hg,
        "y": vg,
    }

    for key in ["phi", "TTH", "tth", "alpha_f"]:
        qmap[key] = np.rad2deg(qmap[key])

    qmap_unit = {
        "phi": "deg",
        "TTH": "deg",
        "tth": "deg",
        "alpha_f": "deg",
        "qx": "Å⁻¹",
        "qy": "Å⁻¹",
        "qz": "Å⁻¹",
        "qr": "Å⁻¹",
        "q": "Å⁻¹",
        "x": "pixel",
        "y": "pixel",
    }
    return qmap, qmap_unit
