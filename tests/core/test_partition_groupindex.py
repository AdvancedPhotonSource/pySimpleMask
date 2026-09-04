# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import numpy as np

from pysimplemask.core.partition import combine_partitions, generate_groupindex_partitions


def _xmap_1x6():
    return np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])


def _group_index_map_1x6():
    # 3 groups of 2 pixels each, matching the typical OR-chain-of-rings recipe
    return np.array([[1, 1, 2, 2, 3, 3]], dtype=np.uint32)


def test_dq_pack_labels_each_group_with_its_own_index():
    pack_dq, _ = generate_groupindex_partitions(
        "q", _group_index_map_1x6(), num_groups=3, xmap=_xmap_1x6(),
        dq_num_per_group=1, sq_num_per_group=2,
    )
    assert pack_dq["num_pts"] == 3
    np.testing.assert_array_equal(pack_dq["partition"], [[1, 1, 2, 2, 3, 3]])
    assert len(pack_dq["v_list"]) == 3


def test_sq_pack_has_disjoint_offset_ranges_per_group():
    _, pack_sq = generate_groupindex_partitions(
        "q", _group_index_map_1x6(), num_groups=3, xmap=_xmap_1x6(),
        dq_num_per_group=1, sq_num_per_group=2,
    )
    assert pack_sq["num_pts"] == 6
    partition = pack_sq["partition"][0]
    # group 1 -> bins {1,2}, group 2 -> bins {3,4}, group 3 -> bins {5,6}; each
    # group's own 2 pixels span its own local 2-bin range with no cross-group reuse
    assert set(partition[0:2]) <= {1, 2}
    assert set(partition[2:4]) <= {3, 4}
    assert set(partition[4:6]) <= {5, 6}
    assert len(pack_sq["v_list"]) == 6


def test_dq_pack_supports_multiple_bins_per_group():
    """dq_num_per_group > 1 gives each group its own independent multi-bin
    dynamic sub-partition, offset the same way the static (sq) axis already is."""
    pack_dq, _ = generate_groupindex_partitions(
        "q", _group_index_map_1x6(), num_groups=3, xmap=_xmap_1x6(),
        dq_num_per_group=2, sq_num_per_group=2,
    )
    assert pack_dq["num_pts"] == 6
    partition = pack_dq["partition"][0]
    # group 1 -> bins {1,2}, group 2 -> bins {3,4}, group 3 -> bins {5,6}; each
    # group's own 2 pixels span its own local 2-bin range with no cross-group reuse
    assert set(partition[0:2]) <= {1, 2}
    assert set(partition[2:4]) <= {3, 4}
    assert set(partition[4:6]) <= {5, 6}
    assert len(pack_dq["v_list"]) == 6


def test_empty_group_yields_zero_partition_for_that_group():
    group_index_map = np.array([[1, 1, 0, 0, 3, 3]], dtype=np.uint32)  # group 2 is empty
    pack_dq, pack_sq = generate_groupindex_partitions(
        "q", group_index_map, num_groups=3, xmap=_xmap_1x6(),
        dq_num_per_group=1, sq_num_per_group=2,
    )
    np.testing.assert_array_equal(pack_dq["partition"], [[1, 1, 0, 0, 3, 3]])
    assert pack_sq["partition"][0, 2] == 0
    assert pack_sq["partition"][0, 3] == 0


def test_combines_cleanly_with_existing_combine_partitions():
    """The dq/sq packs plug into combine_partitions exactly like ordinary
    generate_partition output — no special-casing needed downstream."""
    pack_dq, pack_sq = generate_groupindex_partitions(
        "q", _group_index_map_1x6(), num_groups=3, xmap=_xmap_1x6(),
        dq_num_per_group=1, sq_num_per_group=2,
    )
    phi_partition = np.ones((1, 6), dtype=np.uint32)
    pack_dp = {"map_name": "phi", "num_pts": 1, "partition": phi_partition, "v_list": np.array([0.0])}
    pack_sp = {"map_name": "phi", "num_pts": 1, "partition": phi_partition, "v_list": np.array([0.0])}

    dynamic_map = combine_partitions(pack_dq, pack_dp, prefix="dynamic")
    static_map = combine_partitions(pack_sq, pack_sp, prefix="static")

    assert dynamic_map["dynamic_num_pts"] == [3, 1]
    np.testing.assert_array_equal(dynamic_map["dynamic_roi_map"], [[1, 1, 2, 2, 3, 3]])
    assert static_map["static_num_pts"] == [6, 1]
    assert len(np.unique(static_map["static_roi_map"])) == 6


def test_multi_bin_dq_combines_cleanly_with_existing_combine_partitions():
    """dq_num_per_group > 1 also plugs into combine_partitions cleanly."""
    pack_dq, pack_sq = generate_groupindex_partitions(
        "q", _group_index_map_1x6(), num_groups=3, xmap=_xmap_1x6(),
        dq_num_per_group=2, sq_num_per_group=2,
    )
    phi_partition = np.ones((1, 6), dtype=np.uint32)
    pack_dp = {"map_name": "phi", "num_pts": 1, "partition": phi_partition, "v_list": np.array([0.0])}
    pack_sp = {"map_name": "phi", "num_pts": 1, "partition": phi_partition, "v_list": np.array([0.0])}

    dynamic_map = combine_partitions(pack_dq, pack_dp, prefix="dynamic")
    static_map = combine_partitions(pack_sq, pack_sp, prefix="static")

    assert dynamic_map["dynamic_num_pts"] == [6, 1]
    # every pixel gets its own distinct bin: 3 groups x 2 dq bins, 1 pixel each
    assert (dynamic_map["dynamic_roi_map"] > 0).all()
    assert len(np.unique(dynamic_map["dynamic_roi_map"])) == 6
    assert static_map["static_num_pts"] == [6, 1]
