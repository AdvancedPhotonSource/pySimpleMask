# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Synthetic-fixture builders for the reader test suite.

All fixtures are factory functions that write small in-memory-sized files into
pytest's ``tmp_path`` and return the path. No external data is required.
"""

import re
import struct

import h5py
import numpy as np
import pytest

from pysimplemask.core.reader.formats.imm import (
    _IMM_FIELDNAMES,
    _IMM_HEADER_FORMAT,
    _IMM_HEADER_SIZE,
)

# ---------------------------------------------------------------------------
# Rigaku 64-bit binary
# ---------------------------------------------------------------------------
# Bit layout per 64-bit word: count = bits[0:12], index = bits[16:37],
# frame = bits[40:]. pixel index = row * ncols + col.


def rigaku_words(events, ncols=1024):
    """Encode ``[(frame, row, col, count), ...]`` into a uint64 word array."""
    words = []
    for frame, row, col, count in events:
        index = row * ncols + col
        word = (
            (np.uint64(int(frame)) << np.uint64(40))
            | (np.uint64(int(index)) << np.uint64(16))
            | np.uint64(int(count))
        )
        words.append(word)
    return np.array(words, dtype=np.uint64)


@pytest.fixture
def make_rigaku(tmp_path):
    def _make(events, name="data.bin", ncols=1024):
        path = tmp_path / name
        rigaku_words(events, ncols).tofile(path)
        return str(path)

    return _make


@pytest.fixture
def make_rigaku_3m(tmp_path):
    """Write six Rigaku module files ``<stem>.bin.000`` ... ``.bin.005``.

    ``events_by_suffix`` maps suffix int (0..5) -> list of
    ``(frame, row, col, count)``. Each module file must contain at least one
    event (the 3M loader reads each file's last word to get the frame count).
    Returns the path to the ``.bin.000`` file.
    """

    def _make(events_by_suffix, basename="scan.bin.000", ncols=1024):
        base = str(tmp_path / basename)  # ends with ".bin.000"
        stem = base[:-3]  # ".../scan.bin."
        for suffix in range(6):
            events = events_by_suffix.get(suffix, [])
            rigaku_words(events, ncols).tofile(stem + f"00{suffix}")
        return base

    return _make


# ---------------------------------------------------------------------------
# HDF5 stacks
# ---------------------------------------------------------------------------


@pytest.fixture
def make_hdf(tmp_path):
    def _make(frames, data_path="/entry/data/data", name="data.h5"):
        path = tmp_path / name
        with h5py.File(path, "w") as h:
            h[data_path] = np.asarray(frames)
        return str(path)

    return _make


@pytest.fixture
def make_h5_fields(tmp_path):
    """Write arbitrary ``{hdf_path: value}`` datasets into a fresh HDF5 file."""

    def _make(fields, name="meta.h5"):
        path = tmp_path / name
        with h5py.File(path, "w") as h:
            for hdf_path, value in fields.items():
                h[hdf_path] = value
        return str(path)

    return _make


# ---------------------------------------------------------------------------
# NeXus metadata files (raw field values per beamline keymap)
# ---------------------------------------------------------------------------

# Known raw values written at each beamline's keymap paths. Tests compute the
# expected derived quantities (beam center etc.) from these explicitly.
RAW_8IDI = {
    "energy": 8.0,
    "detector_distance": 8.5,
    "swing_angle_horizontal": 0.0,
    "swing_angle_vertical": 0.0,
    "x_pixel_size": 7.5e-5,
    "y_pixel_size": 7.5e-5,
    "ccdx": 0.0015,
    "ccdy": 0.0030,
    "ccdx0": 0.0,
    "ccdy0": 0.0,
    "bcx0": 500.0,
    "bcy0": 600.0,
}

RAW_9IDD = {
    "energy": 10.92,
    "detector_distance": 0.228,
    "pixel_size": 1.72e-4,
    "incident_angle": 0.14,
    "exposure_time": 1.0,
    "detector_x": 0.0,
    "detector_y": 0.0,
    "orientation": 0.0,
    "_bcx": 800.0,
    "_bcy": 1000.0,
    "_bc_det_x0": 0.0,
    "_bc_det_y0": 0.0,
    "_spx": 810.0,
    "_spy": 1050.0,
    "_sp_det_x0": 0.0,
    "_sp_det_y0": 0.0,
}


@pytest.fixture
def make_nexus_8idi(tmp_path):
    """Write an 8-ID-I NeXus metadata file; return ``(path, raw_values)``."""
    from pysimplemask.core.reader.beamlines.aps_8idi import METADATA_KEYMAPS

    def _make(name="meta_8idi_metadata.hdf", **overrides):
        raw = dict(RAW_8IDI)
        raw.update(overrides)
        path = tmp_path / name
        with h5py.File(path, "w") as h:
            for key, hdf_path in METADATA_KEYMAPS.items():
                h[hdf_path] = raw[key]
        return str(path), raw

    return _make


@pytest.fixture
def make_nexus_9idd(tmp_path):
    """Write a 9-ID-D NeXus metadata file; return ``(path, raw_values)``."""
    from pysimplemask.core.reader.beamlines.aps_9idd import METADATA_KEYMAPS

    def _make(name="meta_9idd_metadata.hdf", **overrides):
        raw = dict(RAW_9IDD)
        raw.update(overrides)
        path = tmp_path / name
        with h5py.File(path, "w") as h:
            for key, hdf_path in METADATA_KEYMAPS.items():
                h[hdf_path] = raw[key]
        return str(path), raw

    return _make


# ---------------------------------------------------------------------------
# IMM files (dense and compression-6 sparse)
# ---------------------------------------------------------------------------

_IMM_TOKENS = re.findall(r"(\d*)([a-zA-Z])", _IMM_HEADER_FORMAT)


def _imm_default(num, ch):
    if ch in ("i", "I"):
        return 0
    if ch in ("d", "f"):
        return 0.0
    if ch == "c":
        return b"\x00"
    if ch == "s":
        return b"\x00" * (int(num) if num else 1)
    raise ValueError(f"unhandled struct token {ch!r}")


def pack_imm_header(**fields):
    """Pack a 1024-byte IMM header, defaulting unspecified fields to zero."""
    assert len(_IMM_TOKENS) == len(_IMM_FIELDNAMES)
    values = [
        fields.get(name, _imm_default(num, ch))
        for (num, ch), name in zip(_IMM_TOKENS, _IMM_FIELDNAMES)
    ]
    buf = struct.pack(_IMM_HEADER_FORMAT, *values)
    assert len(buf) == _IMM_HEADER_SIZE, (len(buf), _IMM_HEADER_SIZE)
    return buf


@pytest.fixture
def make_imm(tmp_path):
    """Write an IMM file from a 3-D ``(n_frames, rows, cols)`` array.

    ``sparse=True`` writes compression-6 sparse frames (int32 index + int16
    count per nonzero pixel); otherwise dense uint16 frames.
    """

    def _make(frames, sparse=False, name="data.imm"):
        frames = np.asarray(frames)
        assert frames.ndim == 3
        _n, rows, cols = frames.shape
        path = tmp_path / name
        with open(path, "wb") as f:
            for frame in frames:
                flat = frame.ravel()
                if sparse:
                    idx = np.nonzero(flat)[0].astype(np.int32)
                    cnt = flat[idx].astype(np.int16)
                    f.write(
                        pack_imm_header(
                            rows=rows, cols=cols, dlen=int(idx.size), compression=6
                        )
                    )
                    f.write(idx.tobytes())
                    f.write(cnt.tobytes())
                else:
                    payload = flat.astype(np.uint16)
                    f.write(
                        pack_imm_header(
                            rows=rows, cols=cols, dlen=int(payload.size), compression=0
                        )
                    )
                    f.write(payload.tobytes())
        return str(path)

    return _make


# ---------------------------------------------------------------------------
# XPCS result files
# ---------------------------------------------------------------------------

@pytest.fixture
def make_xpcs_result(tmp_path):
    """Write a minimal XPCS result HDF5 file with /xpcs structure.

    Parameters (all keyword-only via _make):
      with_qmap_scalars  – write beam_center/energy/distance/pixel_size under /xpcs/qmap
      with_partition     – write roi maps and index arrays under /xpcs/qmap
      with_entry_inst    – write /entry/instrument fallback metadata
    """
    H, W = 16, 16

    def _make(
        name="result.hdf",
        with_qmap_scalars=True,
        with_partition=True,
        with_entry_inst=True,
    ):
        path = tmp_path / name
        with h5py.File(path, "w") as f:
            f["/xpcs/temporal_mean/scattering_2d"] = np.ones(
                (1, H, W), dtype=np.float32
            ) * 5.0

            if with_qmap_scalars:
                f["/xpcs/qmap/beam_center_x"] = 100.0
                f["/xpcs/qmap/beam_center_y"] = 80.0
                f["/xpcs/qmap/energy"] = 10.0
                f["/xpcs/qmap/detector_distance"] = 5.0
                f["/xpcs/qmap/pixel_size"] = 7.5e-5

            if with_partition:
                dmap = np.zeros((H, W), dtype=np.int32)
                dmap[4:8, 4:8] = 1
                smap = np.zeros((H, W), dtype=np.int32)
                smap[4:8, 4:8] = 1
                f["/xpcs/qmap/dynamic_roi_map"] = dmap
                f["/xpcs/qmap/static_roi_map"] = smap
                f["/xpcs/qmap/dynamic_index_mapping"] = np.array([1], dtype=np.int32)
                f["/xpcs/qmap/dynamic_num_pts"] = np.array([1, 16], dtype=np.int32)
                f["/xpcs/qmap/dynamic_v_list_dim0"] = np.array([0.1])
                f["/xpcs/qmap/dynamic_v_list_dim1"] = np.array([0.0])
                f["/xpcs/qmap/static_index_mapping"] = np.array([1], dtype=np.int32)
                f["/xpcs/qmap/static_num_pts"] = np.array([1, 16], dtype=np.int32)
                f["/xpcs/qmap/static_v_list_dim0"] = np.array([0.1])
                f["/xpcs/qmap/static_v_list_dim1"] = np.array([0.0])
                f["/xpcs/qmap/mask"] = np.ones((H, W), dtype=np.uint8)
                f["/xpcs/qmap/blemish"] = np.ones((H, W), dtype=np.uint8)
                f["/xpcs/qmap/map_names"] = np.array([b"q", b"phi"])
                f["/xpcs/qmap/map_units"] = np.array([b"1/A", b"degree"])
                f["/xpcs/qmap/source_file"] = b"/path/to/raw.h5"

            if with_entry_inst:
                f["/entry/instrument/detector_1/beam_center_x"] = 100.0
                f["/entry/instrument/detector_1/beam_center_y"] = 80.0
                f["/entry/instrument/detector_1/x_pixel_size"] = 7.5e-5
                f["/entry/instrument/detector_1/distance"] = 5.0
                f["/entry/instrument/incident_beam/incident_energy"] = 10.0

        return str(path)

    return _make
