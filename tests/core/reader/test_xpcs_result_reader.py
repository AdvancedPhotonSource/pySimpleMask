"""Tests for XPCSResultReader — scattering, metadata, partition loading."""

import numpy as np
import pytest

from pysimplemask.core.reader.beamlines.xpcs_result import XPCSResultReader

H, W = 16, 16

REQUIRED_PARTITION_KEYS = {
    "dynamic_roi_map",
    "static_roi_map",
    "dynamic_index_mapping",
    "dynamic_num_pts",
    "dynamic_v_list_dim0",
    "dynamic_v_list_dim1",
    "static_index_mapping",
    "static_num_pts",
    "static_v_list_dim0",
    "static_v_list_dim1",
    "mask",
    "blemish",
    "map_names",
    "map_units",
    "source_file",
}


def test_scattering_shape(make_xpcs_result):
    reader = XPCSResultReader(make_xpcs_result())
    scat = reader.get_scattering()
    assert scat.shape == (H, W)
    assert scat.dtype == np.float32


def test_metadata_from_xpcs_qmap(make_xpcs_result):
    reader = XPCSResultReader(make_xpcs_result())
    meta = reader.get_metadata()
    assert meta["beam_center_x"] == pytest.approx(100.0)
    assert meta["beam_center_y"] == pytest.approx(80.0)
    assert meta["energy"] == pytest.approx(10.0)
    assert meta["detector_distance"] == pytest.approx(5.0)
    assert meta["pixel_size"] == pytest.approx(7.5e-5)


def test_metadata_fallback_to_entry_instrument(make_xpcs_result):
    path = make_xpcs_result(with_qmap_scalars=False, with_partition=False)
    reader = XPCSResultReader(path)
    meta = reader.get_metadata()
    assert meta["beam_center_x"] == pytest.approx(100.0)
    assert meta["energy"] == pytest.approx(10.0)
    assert meta["pixel_size"] == pytest.approx(7.5e-5)


def test_frame_kwargs_silently_ignored(make_xpcs_result):
    reader = XPCSResultReader(make_xpcs_result())
    scat_default = reader.get_scattering()
    scat_with_kwargs = reader.get_scattering(begin_idx=5, num_frames=100)
    np.testing.assert_array_equal(scat_default, scat_with_kwargs)


def test_saved_partition_contains_required_keys(make_xpcs_result):
    reader = XPCSResultReader(make_xpcs_result())
    assert reader.saved_partition is not None
    assert REQUIRED_PARTITION_KEYS.issubset(reader.saved_partition.keys())


def test_saved_partition_is_none_when_arrays_absent(make_xpcs_result):
    path = make_xpcs_result(with_partition=False)
    reader = XPCSResultReader(path)
    assert reader.saved_partition is None
