"""Tests for the reader dispatch layer.

Covers extension-based format dispatch (``get_format_loader``), beamline
dispatch (``get_reader``), and the error-swallowing handler (``get_handler``).
"""

import numpy as np
import pytest

from pysimplemask.core.file_handler import get_handler
from pysimplemask.core.reader import get_reader
from pysimplemask.core.reader.beamlines.aps_8idi import APS8IDIReader
from pysimplemask.core.reader.beamlines.aps_9idd import APS9IDDReader
from pysimplemask.core.reader.formats import (
    HdfDataset,
    ImmDataset,
    Rigaku3MDataset,
    RigakuDataset,
    get_format_loader,
)


# ---------------------------------------------------------------------------
# get_format_loader: extension-based dispatch to the correct loader type
# ---------------------------------------------------------------------------


def test_get_format_loader_h5(make_hdf):
    path = make_hdf(np.zeros((2, 4, 5)), name="data.h5")
    loader = get_format_loader(path)
    assert isinstance(loader, HdfDataset)
    assert loader.det_size == (4, 5)


def test_get_format_loader_hdf_extension(make_hdf):
    path = make_hdf(np.zeros((2, 4, 5)), name="data.hdf")
    loader = get_format_loader(path)
    assert isinstance(loader, HdfDataset)
    assert loader.det_size == (4, 5)


def test_get_format_loader_imm(make_imm):
    path = make_imm(np.zeros((2, 2, 3), dtype=np.uint16))
    loader = get_format_loader(path)
    assert isinstance(loader, ImmDataset)


def test_get_format_loader_bin_rigaku(make_rigaku):
    path = make_rigaku([(0, 0, 0, 1)])
    loader = get_format_loader(path)
    assert isinstance(loader, RigakuDataset)
    # Single-module .bin must not be promoted to the 3M loader.
    assert not isinstance(loader, Rigaku3MDataset)


def test_get_format_loader_bin000_rigaku_3m(make_rigaku_3m):
    events_by_suffix = {suffix: [(0, 0, 0, 1)] for suffix in range(6)}
    path = make_rigaku_3m(events_by_suffix)
    assert path.endswith(".bin.000")
    loader = get_format_loader(path)
    # ".bin.000" must dispatch to the 3M loader, taking precedence over ".bin".
    assert isinstance(loader, Rigaku3MDataset)


def test_get_format_loader_unsupported_extension():
    with pytest.raises(ValueError):
        get_format_loader("foo.xyz")


# ---------------------------------------------------------------------------
# get_reader: beamline dispatch
# ---------------------------------------------------------------------------


def test_get_reader_aps_8idi(make_hdf):
    path = make_hdf(np.zeros((2, 4, 5)))
    reader = get_reader("APS_8IDI", path)
    assert isinstance(reader, APS8IDIReader)
    assert reader.stype == "Transmission"


def test_get_reader_aps_9idd(make_hdf):
    path = make_hdf(np.zeros((2, 4, 5)))
    reader = get_reader("APS_9IDD", path)
    assert isinstance(reader, APS9IDDReader)
    assert reader.stype == "Reflection"


def test_get_reader_bogus_beamline(make_hdf):
    path = make_hdf(np.zeros((2, 4, 5)))
    with pytest.raises(ValueError):
        get_reader("BOGUS", path)


# ---------------------------------------------------------------------------
# get_handler: error-swallowing delegator
# ---------------------------------------------------------------------------


def test_get_handler_bogus_returns_none(make_hdf):
    path = make_hdf(np.zeros((2, 4, 5)))
    assert get_handler("BOGUS", path) is None


def test_get_handler_aps_8idi_returns_reader(make_hdf):
    path = make_hdf(np.zeros((2, 4, 5)))
    reader = get_handler("APS_8IDI", path)
    assert reader is not None
    assert isinstance(reader, APS8IDIReader)


# ---------------------------------------------------------------------------
# XPCS result auto-detection
# ---------------------------------------------------------------------------


def test_get_handler_xpcs_result_overrides_beamline(make_xpcs_result):
    from pysimplemask.core.reader.beamlines.xpcs_result import XPCSResultReader

    path = make_xpcs_result()
    reader = get_handler("APS_8IDI", path)
    assert isinstance(reader, XPCSResultReader)


def test_get_handler_plain_hdf_not_misdetected_as_xpcs(make_hdf):
    path = make_hdf(np.zeros((2, 4, 5)))
    reader = get_handler("APS_8IDI", path)
    assert isinstance(reader, APS8IDIReader)
