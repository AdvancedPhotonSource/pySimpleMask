"""Tests for the NeXus/HDF5 metadata helpers and beamline metadata readers.

Covers :mod:`pysimplemask.reader.metadata` and the ``get_nexus_metadata`` /
``get_metadata`` entry points of the 8-ID-I and 9-ID-D beamline modules.
"""

import pytest
import numpy as np

from pysimplemask.reader.metadata import (
    _normalize,
    has_nexus_fields,
    read_keymap,
    find_metadata_file,
    read_nexus_metadata,
)
from pysimplemask.reader.beamlines.aps_8idi import (
    METADATA_KEYMAPS as KEYMAP_8IDI,
    get_nexus_metadata as nexus_8idi,
    get_metadata as meta_8idi,
    DEFAULT_METADATA as DEFAULT_8IDI,
)
from pysimplemask.reader.beamlines.aps_9idd import (
    get_nexus_metadata as nexus_9idd,
)

OPTIONAL_8IDI = ["swing_angle_horizontal", "swing_angle_vertical"]


# ---------------------------------------------------------------------------
# 1. _normalize
# ---------------------------------------------------------------------------


def test_normalize_bytes_decoded():
    assert _normalize(b"hi") == "hi"


def test_normalize_single_element_array_to_scalar():
    assert _normalize(np.array([3.5])) == 3.5


def test_normalize_plain_scalar_passthrough():
    assert _normalize(7) == 7
    assert _normalize(2.5) == 2.5


def test_normalize_multi_element_array_unchanged():
    arr = np.array([1.0, 2.0, 3.0])
    out = _normalize(arr)
    assert np.array_equal(out, arr)


# ---------------------------------------------------------------------------
# 2. read_keymap normalization end-to-end
# ---------------------------------------------------------------------------


def test_read_keymap_normalizes_values(make_h5_fields):
    path = make_h5_fields({"/s": b"hello", "/arr": np.array([2.5]), "/n": 7})
    out = read_keymap(path, {"s": "/s", "arr": "/arr", "n": "/n"})
    assert out == {"s": "hello", "arr": 2.5, "n": 7}


# ---------------------------------------------------------------------------
# 3. read_keymap optional handling
# ---------------------------------------------------------------------------


def test_read_keymap_optional_missing_returns_none(make_h5_fields):
    path = make_h5_fields({"/present": 1})
    out = read_keymap(
        path,
        {"present": "/present", "absent": "/absent"},
        optional_fields=["absent"],
    )
    assert out == {"present": 1, "absent": None}


def test_read_keymap_missing_required_raises(make_h5_fields):
    path = make_h5_fields({"/present": 1})
    with pytest.raises(KeyError):
        read_keymap(path, {"present": "/present", "required": "/required"})


# ---------------------------------------------------------------------------
# 4. has_nexus_fields
# ---------------------------------------------------------------------------


def test_has_nexus_fields_all_present(make_h5_fields):
    path = make_h5_fields({"/a": 1, "/b": 2})
    assert has_nexus_fields(path, {"a": "/a", "b": "/b"}) is True


def test_has_nexus_fields_required_missing(make_h5_fields):
    path = make_h5_fields({"/a": 1})
    assert has_nexus_fields(path, {"a": "/a", "b": "/b"}) is False


def test_has_nexus_fields_optional_missing_ok(make_h5_fields):
    path = make_h5_fields({"/a": 1})
    assert (
        has_nexus_fields(path, {"a": "/a", "b": "/b"}, optional_fields=["b"]) is True
    )


def test_has_nexus_fields_non_hdf5_file(tmp_path):
    bad = tmp_path / "not_hdf5.txt"
    bad.write_text("this is not an hdf5 file")
    assert has_nexus_fields(str(bad), {"a": "/a"}) is False


# ---------------------------------------------------------------------------
# 5. find_metadata_file
# ---------------------------------------------------------------------------


def test_find_metadata_file_locates_sibling(make_h5_fields, make_hdf):
    # Both fixtures write into the same tmp_path directory.
    data_path = make_hdf(np.zeros((1, 2, 2)), name="foo.h5")
    meta_path = make_h5_fields({"/x": 1}, name="foo_metadata.hdf")
    assert find_metadata_file(data_path) == meta_path


def test_find_metadata_file_missing_raises(tmp_path):
    data = tmp_path / "foo.h5"
    data.write_text("placeholder")
    with pytest.raises(FileNotFoundError):
        find_metadata_file(str(data))


# ---------------------------------------------------------------------------
# 6. read_nexus_metadata source selection
# ---------------------------------------------------------------------------


def test_read_nexus_metadata_uses_self_when_complete(make_nexus_8idi):
    path, _raw = make_nexus_8idi()
    metadata, meta_fname = read_nexus_metadata(
        path, KEYMAP_8IDI, OPTIONAL_8IDI
    )
    assert meta_fname == path
    assert metadata["meta_fname"] == path


# ---------------------------------------------------------------------------
# 7. 8-ID-I get_nexus_metadata
# ---------------------------------------------------------------------------


def test_get_nexus_metadata_8idi(make_nexus_8idi):
    path, raw = make_nexus_8idi()
    meta = nexus_8idi(path)

    expected_bcx = raw["bcx0"] + (raw["ccdx"] - raw["ccdx0"]) / raw["x_pixel_size"]
    expected_bcy = raw["bcy0"] + (raw["ccdy"] - raw["ccdy0"]) / raw["y_pixel_size"]

    assert np.isclose(meta["beam_center_x"], expected_bcx)
    assert np.isclose(meta["beam_center_y"], expected_bcy)
    assert np.isclose(meta["pixel_size"], raw["x_pixel_size"])
    assert np.isclose(meta["energy"], raw["energy"])

    for key in (
        "bcx0",
        "bcy0",
        "ccdx",
        "ccdy",
        "ccdx0",
        "ccdy0",
        "x_pixel_size",
        "y_pixel_size",
    ):
        assert key not in meta


# ---------------------------------------------------------------------------
# 8. 9-ID-D get_nexus_metadata
# ---------------------------------------------------------------------------


def test_get_nexus_metadata_9idd(make_nexus_9idd):
    path, raw = make_nexus_9idd()
    meta = nexus_9idd(path)

    expected_bcx = round(
        raw["_bcx"] + (raw["detector_x"] - raw["_bc_det_x0"]) / raw["pixel_size"], 3
    )
    expected_spx = round(
        raw["_spx"] + (raw["detector_x"] - raw["_sp_det_x0"]) / raw["pixel_size"], 3
    )

    assert np.isclose(meta["beam_center_x"], expected_bcx)
    assert np.isclose(meta["specular_x"], expected_spx)

    assert not any(key.startswith("_") for key in meta)


# ---------------------------------------------------------------------------
# 9. get_metadata fallback to defaults
# ---------------------------------------------------------------------------


def test_get_metadata_falls_back_to_defaults(tmp_path):
    bad = tmp_path / "bad.hdf"
    bad.write_text("garbage")
    meta = meta_8idi(str(bad))
    assert meta["meta_fname"] == "default_metadata"
    assert "energy" in meta
    assert meta["energy"] == DEFAULT_8IDI["energy"]
