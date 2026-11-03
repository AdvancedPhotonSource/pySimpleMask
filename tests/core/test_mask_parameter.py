# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import numpy as np

from pysimplemask.core.mask import MaskParameter


def _qmap_1x6():
    return {"q": np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])}


def _qmap_1x10():
    return {"q": np.arange(10.0).reshape(1, 10)}


def test_group_index_map_one_group_per_or_chained_row():
    """Typical usage: N rows OR-ed together, each defining one non-overlapping group."""
    mp = MaskParameter(shape=(1, 6))
    constraints = [
        ("q", "AND", "A^-1", 0, 1),
        ("q", "OR", "A^-1", 2, 3),
        ("q", "OR", "A^-1", 4, 5),
    ]
    mp.evaluate(qmap=_qmap_1x6(), constraints=constraints)

    assert mp.num_groups == 3
    np.testing.assert_array_equal(mp.group_index_map, [[1, 1, 2, 2, 3, 3]])
    # the OR chain keeps every pixel
    assert mp.get_mask().all()


def test_group_index_map_and_chain_collapses_to_last_row():
    """An AND-narrowed compound region collapses to one group (the last row),
    and pixels excluded by the final mask are zeroed out."""
    mp = MaskParameter(shape=(1, 6))
    constraints = [
        ("q", "AND", "A^-1", 0, 5),
        ("q", "AND", "A^-1", 2, 3),
    ]
    mp.evaluate(qmap=_qmap_1x6(), constraints=constraints)

    assert mp.num_groups == 2
    np.testing.assert_array_equal(mp.group_index_map, [[0, 0, 2, 2, 0, 0]])
    np.testing.assert_array_equal(mp.get_mask(), [[False, False, True, True, False, False]])


def test_group_index_map_empty_constraints():
    mp = MaskParameter(shape=(1, 6))
    mp.evaluate(qmap=_qmap_1x6(), constraints=[])

    assert mp.num_groups == 0
    np.testing.assert_array_equal(mp.group_index_map, np.zeros((1, 6)))


def test_find_overlaps_detects_overlapping_rows():
    """row0 claims q in [0,5] (6 px), row1 claims q in [3,9] (7 px); they
    overlap in q={3,4,5} (3 px). group_index_map silently gives all 3 to row1
    (last-write-wins) — find_overlaps is what surfaces that this happened."""
    mp = MaskParameter(shape=(1, 10))
    constraints = [
        ("q", "AND", "A^-1", 0, 5),
        ("q", "OR", "A^-1", 3, 9),
    ]
    mp.evaluate(qmap=_qmap_1x10(), constraints=constraints)

    np.testing.assert_array_equal(mp.group_index_map, [[1, 1, 1, 2, 2, 2, 2, 2, 2, 2]])
    assert mp.find_overlaps() == [(1, 2, 3)]


def test_find_overlaps_none_for_disjoint_or_chained_rows():
    mp = MaskParameter(shape=(1, 6))
    constraints = [
        ("q", "AND", "A^-1", 0, 1),
        ("q", "OR", "A^-1", 2, 3),
        ("q", "OR", "A^-1", 4, 5),
    ]
    mp.evaluate(qmap=_qmap_1x6(), constraints=constraints)

    assert mp.find_overlaps() == []


def test_find_overlaps_respects_mask_argument():
    """An overlap that falls entirely outside a caller-supplied mask (e.g. pixels
    excluded by some other mask worker) doesn't count as a conflict."""
    mp = MaskParameter(shape=(1, 10))
    constraints = [
        ("q", "AND", "A^-1", 0, 5),
        ("q", "OR", "A^-1", 3, 9),
    ]
    mp.evaluate(qmap=_qmap_1x10(), constraints=constraints)

    # exclude the overlap region {3,4,5} via an external mask
    external_mask = np.array([[True, True, True, False, False, False, True, True, True, True]])
    assert mp.find_overlaps(mask=external_mask) == []


def test_describe_constraint():
    mp = MaskParameter(shape=(1, 6))
    mp.evaluate(
        qmap=_qmap_1x6(),
        constraints=[("q", "AND", "A^-1", 0.0, 5.0)],
    )
    desc = mp.describe_constraint(1)
    assert "row 1" in desc
    assert "q" in desc
    assert "AND" in desc
    assert "0" in desc and "5" in desc


def test_warns_when_row_matches_no_pixels(caplog):
    """A row whose range lies outside the map (e.g. phi 230..236 on a
    -180..180 map) should warn and name the map's actual span."""
    phi = np.linspace(-180.0, 180.0, 361).reshape(1, 361)  # 1 deg steps
    mp = MaskParameter(shape=(1, 361))
    constraints = [
        ("phi", "AND", "deg", 50, 56),
        ("phi", "OR", "deg", 230, 236),
    ]
    with caplog.at_level("WARNING", logger="pysimplemask.core.mask"):
        mp.evaluate(qmap={"phi": phi}, constraints=constraints)

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "row 2" in warnings[0]
    assert "matches no pixels" in warnings[0]
    assert "[-180, 180]" in warnings[0]
    # the empty row contributes nothing, row 1 still defines the mask
    assert mp.num_groups == 2
    assert not mp.get_mask().all()


def test_warns_when_combined_constraints_leave_no_pixels(caplog):
    """Two disjoint AND rows each match pixels but their intersection is empty."""
    mp = MaskParameter(shape=(1, 10))
    constraints = [
        ("q", "AND", "A^-1", 0, 2),
        ("q", "AND", "A^-1", 5, 7),
    ]
    with caplog.at_level("WARNING", logger="pysimplemask.core.mask"):
        mp.evaluate(qmap=_qmap_1x10(), constraints=constraints)

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "leave no valid pixels" in warnings[0]
    assert not mp.get_mask().any()


def test_no_warning_when_every_row_matches(caplog):
    mp = MaskParameter(shape=(1, 6))
    constraints = [
        ("q", "AND", "A^-1", 0, 1),
        ("q", "OR", "A^-1", 4, 5),
    ]
    with caplog.at_level("WARNING", logger="pysimplemask.core.mask"):
        mp.evaluate(qmap=_qmap_1x6(), constraints=constraints)

    assert not [r for r in caplog.records if r.levelname == "WARNING"]
