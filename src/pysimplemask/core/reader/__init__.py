# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Reader subsystem: beamline dispatch and the app-facing ``FileReader`` base."""

import logging

from .base_reader import FileReader

logger = logging.getLogger(__name__)

__all__ = ["FileReader", "get_reader"]


def get_reader(beamline, fname, **kwargs):
    """Construct the reader for a beamline.

    Args:
        beamline: Beamline identifier, e.g. ``"APS_8IDI"`` or ``"APS_9IDD"``.
        fname: Path to the data file.

    Raises:
        ValueError: If the beamline is not supported.
    """
    if beamline == "APS_8IDI":
        from .beamlines.aps_8idi import APS8IDIReader

        return APS8IDIReader(fname, **kwargs)
    if beamline == "APS_9IDD":
        from .beamlines.aps_9idd import APS9IDDReader

        return APS9IDDReader(fname, **kwargs)
    if beamline == "NativeFiles":
        from .beamlines.native_files import NativeFilesReader

        return NativeFilesReader(fname)
    raise ValueError(f"unsupported beamline: {beamline}")
