# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import h5py
import numpy as np
import pytest

from pysimplemask.core.partition import combine_qmap_files


def _write_qmap_file(path, mask, dynamic, static, map_names=("q", "phi")):
    """Write the minimal subset of the /qmap schema that combine_qmap_files reads."""
    mask = np.asarray(mask)
    with h5py.File(path, "w") as f:
        f.create_dataset("/qmap/map_names", data=list(map_names))
        f.create_dataset("/qmap/mask", data=mask)
        for prefix, pack in (("dynamic", dynamic), ("static", static)):
            roi_map = np.asarray(pack["roi_map"], dtype=np.uint32)
            f.create_dataset(f"/qmap/{prefix}_num_pts", data=np.array(pack["num_pts"]))
            f.create_dataset(
                f"/qmap/{prefix}_v_list_dim0", data=np.asarray(pack["v_list_dim0"], dtype=np.float64)
            )
            f.create_dataset(
                f"/qmap/{prefix}_v_list_dim1", data=np.asarray(pack["v_list_dim1"], dtype=np.float64)
            )
            f.create_dataset(f"/qmap/{prefix}_roi_map", data=roi_map)
            valid = roi_map[roi_map > 0]
            index_mapping = np.arange(int(valid.max())) if valid.size else np.zeros(0, dtype=np.uint32)
            f.create_dataset(f"/qmap/{prefix}_index_mapping", data=index_mapping)
    return str(path)


def _single_bin_pack(roi_map, qval):
    """A trivial dim0=1, dim1=1 pack for one file's disjoint slice of pixels."""
    return {
        "num_pts": [1, 1],
        "v_list_dim0": [qval],
        "v_list_dim1": [0.0],
        "roi_map": roi_map,
    }


def test_combine_requires_at_least_two_files(tmp_path):
    a = _write_qmap_file(
        tmp_path / "a.h5",
        mask=[True, False],
        dynamic=_single_bin_pack([1, 0], 0.1),
        static=_single_bin_pack([1, 0], 0.1),
    )
    with pytest.raises(ValueError, match="at least two"):
        combine_qmap_files([a], str(tmp_path / "out.h5"))


def test_combine_rejects_mismatched_map_names(tmp_path):
    a = _write_qmap_file(
        tmp_path / "a.h5",
        mask=[True, False, False],
        dynamic=_single_bin_pack([1, 0, 0], 0.1),
        static=_single_bin_pack([1, 0, 0], 0.1),
        map_names=("q", "phi"),
    )
    b = _write_qmap_file(
        tmp_path / "b.h5",
        mask=[False, True, False],
        dynamic=_single_bin_pack([0, 1, 0], 0.2),
        static=_single_bin_pack([0, 1, 0], 0.2),
        map_names=("q", "phi"),
    )
    c = _write_qmap_file(
        tmp_path / "c.h5",
        mask=[False, False, True],
        dynamic=_single_bin_pack([0, 0, 1], 0.3),
        static=_single_bin_pack([0, 0, 1], 0.3),
        map_names=("x", "y"),
    )
    with pytest.raises(AssertionError):
        combine_qmap_files([a, b, c], str(tmp_path / "out.h5"))


def test_combine_two_files_unions_mask_and_reindexes_roi_map(tmp_path):
    a = _write_qmap_file(
        tmp_path / "a.h5",
        mask=[True, False, False],
        dynamic=_single_bin_pack([1, 0, 0], 0.1),
        static=_single_bin_pack([1, 0, 0], 0.1),
    )
    b = _write_qmap_file(
        tmp_path / "b.h5",
        mask=[False, True, False],
        dynamic=_single_bin_pack([0, 1, 0], 0.2),
        static=_single_bin_pack([0, 1, 0], 0.2),
    )
    out = tmp_path / "out.h5"
    combine_qmap_files([a, b], str(out))

    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(f["/qmap/mask"][()], [True, True, False])
        for prefix in ("dynamic", "static"):
            np.testing.assert_array_equal(f[f"/qmap/{prefix}_num_pts"][()], [2, 1])
            np.testing.assert_array_equal(f[f"/qmap/{prefix}_v_list_dim0"][()], [0.1, 0.2])
            np.testing.assert_array_equal(f[f"/qmap/{prefix}_roi_map"][()], [1, 2, 0])


def test_combine_three_files_sums_dim0_and_concatenates_in_order(tmp_path):
    a = _write_qmap_file(
        tmp_path / "a.h5",
        mask=[True, False, False],
        dynamic=_single_bin_pack([1, 0, 0], 0.1),
        static=_single_bin_pack([1, 0, 0], 0.1),
    )
    b = _write_qmap_file(
        tmp_path / "b.h5",
        mask=[False, True, False],
        dynamic=_single_bin_pack([0, 1, 0], 0.2),
        static=_single_bin_pack([0, 1, 0], 0.2),
    )
    c = _write_qmap_file(
        tmp_path / "c.h5",
        mask=[False, False, True],
        dynamic=_single_bin_pack([0, 0, 1], 0.3),
        static=_single_bin_pack([0, 0, 1], 0.3),
    )
    out = tmp_path / "out.h5"
    combine_qmap_files([a, b, c], str(out))

    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(f["/qmap/mask"][()], [True, True, True])
        for prefix in ("dynamic", "static"):
            np.testing.assert_array_equal(f[f"/qmap/{prefix}_num_pts"][()], [3, 1])
            np.testing.assert_array_equal(
                f[f"/qmap/{prefix}_v_list_dim0"][()], [0.1, 0.2, 0.3]
            )
            np.testing.assert_array_equal(f[f"/qmap/{prefix}_roi_map"][()], [1, 2, 3])


def test_combine_picks_dim1_v_list_from_file_with_most_dim1_bins(tmp_path):
    # dynamic stays uniform (dim1=1) across all three files; only static's
    # middle file (b) has the largest dim1 count, to prove the winner is
    # chosen by comparing all N files, not just first-vs-last.
    a = _write_qmap_file(
        tmp_path / "a.h5",
        mask=[True, False, False],
        dynamic=_single_bin_pack([1, 0, 0], 0.1),
        static={
            "num_pts": [1, 1],
            "v_list_dim0": [0.1],
            "v_list_dim1": [0.0],
            "roi_map": [1, 0, 0],
        },
    )
    b = _write_qmap_file(
        tmp_path / "b.h5",
        mask=[False, True, False],
        dynamic=_single_bin_pack([0, 1, 0], 0.2),
        static={
            "num_pts": [1, 3],
            "v_list_dim0": [0.2],
            "v_list_dim1": [10.0, 20.0, 30.0],
            "roi_map": [0, 1, 0],
        },
    )
    c = _write_qmap_file(
        tmp_path / "c.h5",
        mask=[False, False, True],
        dynamic=_single_bin_pack([0, 0, 1], 0.3),
        static={
            "num_pts": [1, 1],
            "v_list_dim0": [0.3],
            "v_list_dim1": [0.0],
            "roi_map": [0, 0, 1],
        },
    )
    out = tmp_path / "out.h5"
    combine_qmap_files([a, b, c], str(out))

    with h5py.File(out, "r") as f:
        np.testing.assert_array_equal(f["/qmap/dynamic_num_pts"][()], [3, 1])
        np.testing.assert_array_equal(f["/qmap/static_num_pts"][()], [3, 3])
        np.testing.assert_array_equal(
            f["/qmap/static_v_list_dim1"][()], [10.0, 20.0, 30.0]
        )
