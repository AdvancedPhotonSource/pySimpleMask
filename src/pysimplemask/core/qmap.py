# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)
E2KCONST = 12.39841984


def compute_qmap(stype, metadata):
    if stype == "Transmission":
        return compute_transmission_qmap(
            metadata["energy"],
            (metadata["beam_center_y"], metadata["beam_center_x"]),
            (metadata["detector_shape_y"], metadata["detector_shape_x"]),
            metadata["pixel_size"],
            metadata["detector_distance"],
            metadata["swing_angle_horizontal"],
            metadata.get("swing_angle_vertical", 0.0),
        )
    elif stype == "Reflection":
        return compute_reflection_qmap(
            metadata["energy"],
            (metadata["beam_center_y"], metadata["beam_center_x"]),
            (metadata["specular_y"], metadata["specular_x"]),
            (metadata["detector_shape_y"], metadata["detector_shape_x"]),
            metadata["pixel_size"],
            metadata["detector_distance"],
            alpha_i_deg=metadata["incident_angle"],
        )


def compute_display_center(
    center,
    detector_distance,
    pixel_size,
    swing_angle_horizontal=0,
    swing_angle_vertical=0,
):
    # center shift follows the same logic as horizontal:
    # swing > 0 (Outboard/Up) -> Beam hits "Lower/Inboard" on detector
    # Based on existing horizontal logic: swing < 0 -> center shift negative.
    # So swing > 0 -> center shift positive.
    # We apply same sign for vertical.
    center_v = (
        center[0]
        + detector_distance * np.tan(np.deg2rad(swing_angle_vertical)) / pixel_size
    )
    center_h = (
        center[1]
        + detector_distance * np.tan(np.deg2rad(swing_angle_horizontal)) / pixel_size
    )
    return (float(center_v), float(center_h))


@lru_cache(maxsize=128)
def compute_transmission_qmap(
    energy,
    center,
    shape,
    pixel_size,
    detector_distance,
    swing_angle_horizontal,
    swing_angle_vertical=0,
):
    k0 = 2 * np.pi / (E2KCONST / energy)

    # swing_angle_horizontal is negative when swin towards the wall
    swing_angle_horizontal_rad = np.deg2rad(swing_angle_horizontal)
    swing_angle_vertical_rad = np.deg2rad(swing_angle_vertical)

    # before swing angle correction
    v = np.arange(shape[0], dtype=np.int32)
    h = np.arange(shape[1], dtype=np.int32)
    vg_pxl, hg_pxl = np.meshgrid(v, h, indexing="ij")
    vg = (vg_pxl - center[0]) * pixel_size  # vertical grid
    hg = (hg_pxl - center[1]) * pixel_size  # horizontal grid
    lg = np.ones_like(vg) * detector_distance  # longitudinal grid

    # Rotation Matrices
    # Vertical Swing (around X): Z (beam) -> +Y (Up) for positive angle
    # (0, 0, 1) -> (0, sin, cos)

    cv = np.cos(swing_angle_vertical_rad)
    sv = np.sin(swing_angle_vertical_rad)

    Rx = np.array([[1, 0, 0], [0, cv, -sv], [0, sv, cv]])

    ch = np.cos(swing_angle_horizontal_rad)
    sh = np.sin(swing_angle_horizontal_rad)
    # Ry for Horizontal. Previously validated as [[c, 0, -s], [0, 1, 0], [s, 0, c]]
    # This maps (0,0,1) -> (-s, 0, c).
    # If swing > 0 is Outboard (+X?), then -s should be +X? No.
    # Assumed previous logic was correct.
    Ry = np.array([[ch, 0, -sh], [0, 1, 0], [sh, 0, ch]])

    # Combined Rotation: Horizontal First (parent), then Vertical (child)
    # But usually "first horizontal" means horizontal motor is at base.
    # So R_total = R_horizontal @ R_Vertical.
    R = Ry @ Rx

    # Apply rotation
    # v_lab = R @ v_det
    # v_det components are hg, vg, lg
    # hg is x, vg is y, lg is z
    det_vec = np.stack([hg, vg, lg], axis=-1)
    coor_mat = np.dot(det_vec, R.T)

    hg_rot = coor_mat[..., 0]
    vg_rot = coor_mat[..., 1]
    lg_rot = coor_mat[..., 2]

    # direct_beam = np.array([0, 0, detector_distance])  # incoming direct beam vector
    # Calculate alpha (scattering angle 2theta)
    # cos(alpha) = (v . beam) / (|v| |beam|)
    # |beam| = D. beam = (0,0,D).
    # v . beam = lg_rot * D
    # |v| = sqrt(hg_r^2 + vg_r^2 + lg_r^2)
    norm = np.linalg.norm(coor_mat, axis=-1)
    cos_alpha = (lg_rot * detector_distance) / (norm * detector_distance)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)

    # phi (azimuth).
    phi = np.arctan2(vg_rot, hg_rot) * (-1)

    qr = np.sin(alpha) * k0
    qx = qr * np.cos(phi)
    qy = qr * np.sin(phi)
    # qz = k0 * (np.cos(alpha) - 1.0)
    q = 2.0 * k0 * np.sin(alpha / 2.0)

    # keep phi and q as np.float64 to keep the precision.
    qmap = {
        "phi": np.rad2deg(phi),
        "TTH": np.rad2deg(alpha),
        "q": q,
        # "qr": qr.astype(np.float32),
        # "qz": qz.astype(np.float32),
        "qx": qx.astype(np.float32),
        "qy": qy.astype(np.float32),
        "x": hg_pxl,
        "y": vg_pxl,
    }

    qmap_unit = {
        "phi": "deg",
        "TTH": "deg",
        "q": "Å⁻¹",
        # "qr": "Å⁻¹",
        #  "qz": "Å⁻¹",
        "qx": "Å⁻¹",
        "qy": "Å⁻¹",
        "x": "pixel",
        "y": "pixel",
    }
    return qmap, qmap_unit


@lru_cache(maxsize=128)
def compute_reflection_qmap(
    energy,
    center,
    specular,
    shape,
    pixel_size,
    detector_distance,
    alpha_i_deg=0.14,
):
    """Per-pixel q-map for grazing-incidence (reflection) geometry.

    ``center`` (direct beam) and ``specular`` (reflected beam) are (row, col)
    in pixels. The line joining them is the sample-surface normal projected on
    the detector, so it fixes the detector rotation about the beam; alpha_f is
    measured along that line and tth along the horizon perpendicular to it.
    """
    k0 = 2 * np.pi / (E2KCONST / energy)

    # unit vectors in (row, col): normal points from the specular spot to the
    # beam center; parallel is the normal rotated by 90 deg (the horizon).
    normal = np.array(center, dtype=np.float64) - np.array(specular, dtype=np.float64)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        raise ValueError(
            f"specular {specular} coincides with the beam center {center}; "
            "the sample normal is undefined"
        )
    normal /= norm
    parallel = np.array([normal[1], -normal[0]])

    v = center[0] - np.arange(shape[0], dtype=np.float64)
    h = center[1] - np.arange(shape[1], dtype=np.float64)
    dv, dh = np.meshgrid(v, h, indexing="ij")
    # pixel offset from the beam center in the sample frame, in meters:
    # v_off along the surface normal (up), h_off along the horizon.
    v_off = (dv * normal[0] + dh * normal[1]) * pixel_size
    h_off = (dv * parallel[0] + dh * parallel[1]) * pixel_size

    phi = np.arctan2(v_off, h_off)
    TTH = np.arctan2(np.hypot(v_off, h_off), detector_distance)

    alpha_i = np.deg2rad(alpha_i_deg)
    # Exact exit angles. Lab frame: x along the incident beam, z along v_off,
    # y along h_off; the ray to a pixel is s = (D, h_off, v_off) / |s|. The
    # surface is tilted by alpha_i about y, with normal n = (-sin ai, 0, cos ai)
    # and in-plane beam axis e1 = (cos ai, 0, sin ai):
    #   sin(alpha_f) = s.n,   tan(tth) = s.y / s.e1
    # The small-angle form alpha_f = atan(v_off / D) - alpha_i ignores h_off and
    # overestimates |q| by several percent at wide angles.
    dist = np.sqrt(v_off**2 + h_off**2 + detector_distance**2)
    sin_af = (v_off * np.cos(alpha_i) - detector_distance * np.sin(alpha_i)) / dist
    alpha_f = np.arcsin(np.clip(sin_af, -1.0, 1.0))
    tth = np.arctan2(
        h_off, detector_distance * np.cos(alpha_i) + v_off * np.sin(alpha_i)
    )

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
        "x": h_off / pixel_size,
        "y": -v_off / pixel_size,
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
