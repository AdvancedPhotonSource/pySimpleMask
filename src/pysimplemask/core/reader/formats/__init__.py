# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Scattering-format loaders and extension-based dispatch."""

import logging

from . import rigaku, rigaku_extend
from .base import ScatteringDataset
from .hdf import HdfDataset, count_hdf_frames
from .imm import ImmDataset
from .rigaku import Rigaku3MDataset, RigakuDataset
from .rigaku_extend import RigakuExtendDataset

logger = logging.getLogger(__name__)

__all__ = [
    "ScatteringDataset",
    "HdfDataset",
    "ImmDataset",
    "RigakuDataset",
    "Rigaku3MDataset",
    "RigakuExtendDataset",
    "count_frames",
    "count_hdf_frames",
    "get_format_loader",
]

# Rigaku 3M ships six module files ending in .bin.000 ... .bin.005
_RIGAKU_3M_ENDINGS = tuple(f".bin.00{i}" for i in range(6))


def _load_timepix(fname):
    """Lazily construct the external Timepix loader (optional dependency)."""
    try:
        from timepix_dataset.dataset import TimepixRawDataset
    except ImportError as err:
        raise ImportError(
            "Timepix files require the 'timepix_dataset' package to be installed"
        ) from err
    return TimepixRawDataset(fname)


def get_format_loader(fname, **kwargs):
    """Return a scattering-format loader appropriate for ``fname``'s extension.

    Recognized extensions: ``.bin`` (Rigaku 500k), ``.bin.00N`` (Rigaku 3M),
    ``.bix`` (RigakuExtend), ``.imm``, ``.h5``/``.hdf``, and ``.tpx``/``.tpx.000``
    (Timepix).

    Raises:
        ValueError: If the extension is not recognized.
    """
    if fname.endswith(_RIGAKU_3M_ENDINGS):
        logger.info("Rigaku 3M (6 x 500k) dataset")
        return Rigaku3MDataset(fname, **kwargs)
    if fname.endswith(".bin"):
        logger.info("Rigaku 500k dataset")
        return RigakuDataset(fname, **kwargs)
    if fname.endswith(".bix"):
        logger.info("RigakuExtend dataset")
        return RigakuExtendDataset(fname, **kwargs)
    if fname.endswith(".imm"):
        logger.info("IMM dataset")
        return ImmDataset(fname, **kwargs)
    if fname.endswith((".h5", ".hdf")):
        logger.info("APS HDF dataset")
        return HdfDataset(fname, **kwargs)
    if fname.endswith((".tpx", ".tpx.000")):
        logger.info("Timepix dataset")
        return _load_timepix(fname)
    raise ValueError(f"unsupported dataset file: {fname}")


def count_frames(fname):
    """Return the number of frames in the dataset file ``fname``.

    Only what locates the frames is read (the HDF dataset shape, the last word
    of a Rigaku file, the IMM frame headers), never the frames themselves.
    Returns 0 when the count cannot be told: a missing or unreadable file, a
    single image, or a format without a cheap count (Timepix).
    """
    try:
        if fname.endswith(_RIGAKU_3M_ENDINGS):
            # as Rigaku3MDataset: the most that any of the six module files holds
            stem = fname[: -len(_RIGAKU_3M_ENDINGS[0])]
            return max(
                rigaku.get_number_of_frames_from_binfile(stem + ending)
                for ending in _RIGAKU_3M_ENDINGS
            )
        if fname.endswith(".bin"):
            return rigaku.get_number_of_frames_from_binfile(fname)
        if fname.endswith(".bix"):
            return rigaku_extend.get_number_of_frames_from_binfile(fname)
        if fname.endswith(".imm"):
            return len(ImmDataset(fname).toc)
    except (OSError, ValueError):
        return 0
    return count_hdf_frames(fname)
