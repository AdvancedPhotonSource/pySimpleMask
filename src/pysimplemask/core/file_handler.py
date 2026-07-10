import logging

import h5py

from .reader import get_reader

logger = logging.getLogger(__name__)


def get_handler(beamline, fname, **kwargs):
    """Return a reader for the given file, or ``None`` on failure.

    If ``fname`` is an HDF5 file containing a top-level ``/xpcs`` group it is
    treated as an XPCS result file regardless of ``beamline``; otherwise the
    ``beamline`` string drives dispatch as usual.
    """
    try:
        if h5py.is_hdf5(fname):
            with h5py.File(fname, "r") as f:
                if "/xpcs" in f:
                    from .reader.beamlines.xpcs_result import XPCSResultReader

                    return XPCSResultReader(fname)
        return get_reader(beamline, fname, **kwargs)
    except Exception:
        logger.error(
            "failed to create a reader for beamline=%s file=%s",
            beamline,
            fname,
            exc_info=True,
        )
        return None
