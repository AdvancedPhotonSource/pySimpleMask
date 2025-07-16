import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)
E2KCONST = 12.39841984


def compute_qmap(stype, metadata):
    if stype == "Transmission":
        return compute_transmission_qmap(
            metadata["energy"],
            (metadata["bcy"], metadata["bcx"]),
            metadata["shape"],
            metadata["pix_dim"],
            metadata["det_dist"],
            metadata["swing_angle"],
        )
    elif stype == "Reflection":
        return compute_reflection_qmap(
            metadata["energy"],
            (metadata["bcy"], metadata["bcx"]),
            metadata["shape"],
            metadata["pix_dim"],
            metadata["det_dist"],
            alpha_i_deg=metadata["alpha_i_deg"],
            orientation=metadata["orientation"],
        )


def compute_display_center(center, det_dist, pix_dim, swing_angle=0):
    center_v = center[0]
    # if swing_angle < 0, the center shift towards the nagative direction (DOOR)
    center_h = center[1] + det_dist * np.tan(np.deg2rad(swing_angle)) / pix_dim
    return (center_v, center_h)


@lru_cache(maxsize=128)
def compute_transmission_qmap(energy, center, shape, pix_dim, det_dist, swing_angle):
    k0 = 2 * np.pi / (E2KCONST / energy)

    # swing_angle is negative when swin towards the wall
    swing_angle_rad = np.deg2rad(swing_angle)
    # the horizontal center shift towards the nagative direction when angle < 0

    # before swing angle correction
    v = np.arange(shape[0], dtype=np.uint32) - center[0]
    h = np.arange(shape[1], dtype=np.uint32) - center[1]
    vg, hg = np.meshgrid(v, h, indexing="ij")
    vg = vg * pix_dim  # vertical grid
    hg = hg * pix_dim  # horizontal grid
    lg = np.ones_like(vg) * det_dist  # longitudinal grid

    # pixel-wise distance to the detector
    pixel_det = np.sqrt(vg**2 + hg**2 + lg**2)
    # apply the swing angle correction. outboard is positive, inboard is negative
    pixel_angle = np.arctan2(hg, pixel_det) * (-1) + swing_angle_rad

    # the left part is positive, the right part is negative
    hg_rot = np.sin(pixel_angle) * pixel_det * (-1)
    lg_rot = np.cos(pixel_angle) * pixel_det  # along the beam
    vg_rot = vg  # swing does not change the vertical direction
    coor_mat = np.stack([hg_rot, vg_rot, lg_rot], axis=-1)  # shape (v, h, 3), lab frame

    direct_beam = np.array([0, 0, det_dist])  # incoming direct beam vector
    norm = np.linalg.norm(coor_mat, axis=-1) * np.linalg.norm(direct_beam)
    # compute the angle between the direct beam and the scattered pixel vector
    alpha = np.arccos(np.dot(coor_mat, direct_beam) / norm)

    phi = np.arctan2(vg_rot, hg_rot) * (-1)

    qr = np.sin(alpha) * k0
    qx = qr * np.cos(phi)
    qy = qr * np.sin(phi)

    # keep phi and q as np.float64 to keep the precision.
    qmap = {
        "phi": np.rad2deg(phi),
        "TTH": np.rad2deg(alpha),
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
def compute_reflection_qmap(
    energy, center, shape, pix_dim, det_dist, alpha_i_deg=0.14, orientation="north"
):
    k0 = 2 * np.pi / (E2KCONST / energy)

    v = np.arange(shape[0], dtype=np.uint32) - center[0]
    h = np.arange(shape[1], dtype=np.uint32) - center[1]
    vg, hg = np.meshgrid(v, h, indexing="ij")
    vg *= -1

    if orientation == "north":
        pass
    elif orientation == "west":
        vg, hg = -hg, vg
    elif orientation == "south":
        vg, hg = -vg, -hg
    elif orientation == "east":
        vg, hg = hg, -vg
    else:
        logger.warning("Unknown orientation: {orientation}. using default north")

    r = np.hypot(vg, hg) * pix_dim
    phi = np.arctan2(vg, hg)
    TTH = np.arctan(r / det_dist)

    alpha_i = np.deg2rad(alpha_i_deg)
    alpha_f = np.arctan(vg * pix_dim / det_dist) - alpha_i
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
        "y": vg * (-1),
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
