# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import numpy as np
import pytest

from pysimplemask.core.partition import generate_partition, least_multiple


@pytest.mark.parametrize(
    "a,b,expected",
    [(10, 100, 100), (10, 95, 100), (36, 360, 360), (36, 350, 360), (7, 1, 7)],
)
def test_least_multiple(a, b, expected):
    assert least_multiple(a, b) == expected


def test_symmetry_fold_v_list_has_num_pts_entries_outside_first_fold():
    """Regression: with symmetry_fold>1, v_list came from a bincount over the
    first fold only, so a region with no pixels in [0, 360/fold) got a short
    (even empty) v_list that no longer lines up with num_pts."""
    phi = np.linspace(-180, 180, 361).reshape(1, -1)
    mask = (phi >= 100) & (phi <= 140)
    out = generate_partition("phi", mask, phi, num_pts=2, symmetry_fold=4)

    assert len(out["v_list"]) == 2
    assert np.all((out["v_list"] >= 100) & (out["v_list"] <= 140))
