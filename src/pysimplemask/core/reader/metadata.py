# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Shared NeXus/HDF5 metadata helpers used by the beamline readers."""

import glob
import logging
import os

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def _normalize(value):
    """Turn a raw HDF5 value into a plain Python/NumPy scalar where possible.

    Decodes byte strings and collapses single-element arrays so downstream
    arithmetic and formatting behave consistently.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.reshape(-1)[0]
        return value
    return value


def has_nexus_fields(fname, keymap, optional_fields=None):
    """Return True if ``fname`` is an HDF5 file containing every required field.

    Args:
        fname: Path to the candidate file.
        keymap: Mapping of metadata key -> HDF5 path.
        optional_fields: Keys allowed to be absent.
    """
    if not h5py.is_hdf5(fname):
        return False

    optional_fields = set(optional_fields or ())
    with h5py.File(fname, "r") as f:
        for key, hdf_path in keymap.items():
            if key in optional_fields:
                continue
            if hdf_path not in f:
                return False
    return True


def read_keymap(fname, keymap, optional_fields=None):
    """Read metadata values from an HDF5 file using a key -> path mapping.

    Optional fields that are missing are returned as ``None``. All values are
    normalized via :func:`_normalize`.
    """
    optional_fields = set(optional_fields or ())
    metadata = {}
    with h5py.File(fname, "r") as f:
        for key, hdf_path in keymap.items():
            if hdf_path not in f:
                if key in optional_fields:
                    metadata[key] = None
                    continue
                raise KeyError(f"required field {hdf_path!r} missing in {fname}")
            metadata[key] = _normalize(f[hdf_path][()])
    return metadata


def find_metadata_file(fname):
    """Find a ``*_metadata.hdf`` file in the same folder as ``fname``.

    Returns:
        str: Path to the metadata file (the first one if several exist).

    Raises:
        FileNotFoundError: If no metadata file is present.
    """
    pattern = os.path.join(os.path.dirname(fname), "*_metadata.hdf")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no *_metadata.hdf found in the folder of {fname}")
    if len(matches) > 1:
        logger.warning(
            "multiple *_metadata.hdf found in the folder of %s; using %s",
            fname,
            matches[0],
        )
    return matches[0]


def read_nexus_metadata(fname, keymap, optional_fields=None):
    """Locate and read NeXus metadata for ``fname``.

    The data file itself is used when it already contains the required fields;
    otherwise a sibling ``*_metadata.hdf`` file is located and validated.

    Returns:
        tuple: ``(metadata_dict, meta_fname)``.
    """
    if has_nexus_fields(fname, keymap, optional_fields):
        meta_fname = fname
    else:
        meta_fname = find_metadata_file(fname)
        if not has_nexus_fields(meta_fname, keymap, optional_fields):
            raise FileNotFoundError(f"No valid metadata found in {meta_fname}")

    logger.info("using metadata file: %s", meta_fname)
    metadata = read_keymap(meta_fname, keymap, optional_fields)
    metadata["meta_fname"] = meta_fname
    return metadata, meta_fname
