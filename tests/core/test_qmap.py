# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Tests for the reflection (grazing-incidence) q-map geometry."""

import numpy as np
import pytest

from pysimplemask.core.qmap import E2KCONST, compute_qmap, compute_reflection_qmap

ENERGY = 10.0
PIXEL = 1e-3
DIST = 0.2
SHAPE = (201, 211)
CENTER = (100.0, 105.0)
K0 = 2 * np.pi / (E2KCONST / ENERGY)
# A specular spot 50 px from the beam center sits at 2 * alpha_i.
SPEC_DIST = 50
ALPHA_I = 0.5 * np.rad2deg(np.arctan(SPEC_DIST * PIXEL / DIST))


def _qmap(specular, shape=SHAPE, center=CENTER):
    qmap, _unit = compute_reflection_qmap(
        ENERGY, center, specular, shape, PIXEL, DIST, alpha_i_deg=ALPHA_I
    )
    return qmap


def _angle(npix):
    return np.rad2deg(np.arctan(npix * PIXEL / DIST))


def test_q_magnitude_is_exact():
    # |q| = 2 k0 sin(TTH / 2) holds for every pixel, including wide angles
    # (the detector here spans about +-27 deg).
    qmap = _qmap((CENTER[0] - 30, CENTER[1] + 40))
    expected = 2 * K0 * np.sin(np.deg2rad(qmap["TTH"]) / 2)
    assert np.allclose(qmap["q"], expected, rtol=1e-12, atol=1e-12)


def test_specular_above_center_upright_geometry():
    qmap = _qmap((CENTER[0] - SPEC_DIST, CENTER[1]))
    v, h = np.indices(SHAPE)
    c = int(CENTER[1])
    # on the specular line the exit angle is the plain elevation angle
    assert np.allclose(qmap["alpha_f"][:, c], _angle(CENTER[0] - v[:, c]) - ALPHA_I)
    assert np.allclose(qmap["tth"][:, c], 0.0)
    # tth > 0 to the right, alpha_f symmetric across the specular line
    assert np.all(qmap["tth"][:, c + 1 :] > 0)
    assert np.allclose(qmap["tth"][:, c - 10], -qmap["tth"][:, c + 10])
    assert np.allclose(qmap["alpha_f"][:, c - 10], qmap["alpha_f"][:, c + 10])
    assert np.allclose(qmap["x"], h - CENTER[1])
    assert np.allclose(qmap["y"], v - CENTER[0])


def test_specular_spot_reflects_at_alpha_i():
    sp = (int(CENTER[0]) - SPEC_DIST, int(CENTER[1]))
    bc = (int(CENTER[0]), int(CENTER[1]))
    qmap = _qmap(sp)
    assert qmap["alpha_f"][sp] == pytest.approx(ALPHA_I)
    assert qmap["alpha_f"][bc] == pytest.approx(-ALPHA_I)
    assert qmap["tth"][sp] == pytest.approx(0.0, abs=1e-12)
    assert qmap["qz"][sp] == pytest.approx(2 * K0 * np.sin(np.deg2rad(ALPHA_I)))
    assert qmap["qr"][sp] == pytest.approx(0.0, abs=1e-12)


def test_specular_below_center_flips_normal():
    # e.g. a detector mounted upside down: the reflected beam lands at a
    # larger row index than the direct beam.
    sp = (int(CENTER[0]) + SPEC_DIST, int(CENTER[1]))
    qmap = _qmap(sp)
    r, c = int(CENTER[0]), int(CENTER[1])
    assert qmap["alpha_f"][sp] == pytest.approx(ALPHA_I)
    assert qmap["alpha_f"][r, c] == pytest.approx(-ALPHA_I)
    assert qmap["alpha_f"][r + 10, c] > qmap["alpha_f"][r, c]
    # the horizon direction rotates with the normal (180 deg rotation)
    assert qmap["tth"][r, c - 10] > 0


def test_tilted_specular_rotates_qmap():
    r, c = int(CENTER[0]), int(CENTER[1])
    upright = _qmap((r - SPEC_DIST, c))
    # "up" along (-3, 4) / 5 in (row, col): the upright offset (a, b) lands
    # at (0.6 a + 0.8 b, -0.8 a + 0.6 b)
    tilted = _qmap((r - 30, c + 40))
    for a, b in [(-50, 0), (-25, 0), (0, 25), (0, -25), (-10, 15), (20, -5), (35, 40)]:
        ta, tb = round(0.6 * a + 0.8 * b), round(-0.8 * a + 0.6 * b)
        for key in upright:
            if key in ("x", "y"):
                continue
            assert tilted[key][r + ta, c + tb] == pytest.approx(
                upright[key][r + a, c + b], abs=1e-9
            ), (key, a, b)
    assert tilted["alpha_f"][r - 30, c + 40] == pytest.approx(ALPHA_I)


def test_rotating_specular_by_90_rotates_qmap():
    shape, center = (201, 201), (100.0, 100.0)
    upright = _qmap((50.0, 100.0), shape=shape, center=center)
    # rotate the detector 90 deg counter-clockwise: "up" now points left
    rotated = _qmap((100.0, 50.0), shape=shape, center=center)
    for key in upright:
        assert np.allclose(rotated[key], np.rot90(upright[key]), atol=1e-9), key


def test_coincident_specular_raises():
    with pytest.raises(ValueError, match="specular"):
        _qmap(CENTER)


def test_compute_qmap_reads_specular_from_metadata():
    metadata = {
        "energy": ENERGY,
        "beam_center_x": CENTER[1],
        "beam_center_y": CENTER[0],
        "specular_x": CENTER[1] + 40,
        "specular_y": CENTER[0] - 30,
        "detector_shape_x": SHAPE[1],
        "detector_shape_y": SHAPE[0],
        "pixel_size": PIXEL,
        "detector_distance": DIST,
        "incident_angle": ALPHA_I,
    }
    qmap, unit = compute_qmap("Reflection", metadata)
    expected = _qmap((CENTER[0] - 30, CENTER[1] + 40))
    assert list(qmap) == list(expected)
    for key in expected:
        assert np.array_equal(qmap[key], expected[key]), key
    assert unit["alpha_f"] == "deg"
