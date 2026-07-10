"""Reader for XPCS result HDF5 files (those containing a /xpcs group)."""

import logging

import h5py
import numpy as np

from ..base_reader import FileReader
from ..metadata import read_keymap
from .aps_8idi import DEFAULT_METADATA_WITHUNITS

logger = logging.getLogger(__name__)

# Metadata paths inside /xpcs/qmap (the preferred, already-refined values).
_XPCS_QMAP_KEYMAP = {
    "beam_center_x": "/xpcs/qmap/beam_center_x",
    "beam_center_y": "/xpcs/qmap/beam_center_y",
    "energy": "/xpcs/qmap/energy",
    "detector_distance": "/xpcs/qmap/detector_distance",
    "pixel_size": "/xpcs/qmap/pixel_size",
}

# Fallback paths in /entry/instrument when /xpcs/qmap scalars are absent.
_FALLBACK_KEYMAP = {
    "beam_center_x": "/entry/instrument/detector_1/beam_center_x",
    "beam_center_y": "/entry/instrument/detector_1/beam_center_y",
    "energy": "/entry/instrument/incident_beam/incident_energy",
    "detector_distance": "/entry/instrument/detector_1/distance",
    "pixel_size": "/entry/instrument/detector_1/x_pixel_size",
}

# Array datasets to load from /xpcs/qmap into saved_partition.
_PARTITION_ARRAY_KEYS = [
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
]


class XPCSResultReader(FileReader):
    """Reader for XPCS analysis result HDF5 files.

    Auto-detected by ``file_handler.get_handler`` when the file contains a
    top-level ``/xpcs`` group. Reads the temporally-averaged scattering image
    and restores the existing partition from ``/xpcs/qmap``.
    """

    ftype = "XPCS_Result"
    stype = "Transmission"

    def __init__(self, fname):
        super().__init__(fname)
        self.meta_units_fmts = DEFAULT_METADATA_WITHUNITS.copy()
        self.saved_partition = self._load_partition()

    def get_scattering(self, **kwargs):
        """Return the pre-averaged 2-D scattering image.

        All kwargs (``begin_idx``, ``num_frames``, etc.) are accepted and
        silently ignored — the image is already temporally averaged.
        """
        with h5py.File(self.fname, "r") as f:
            return f["/xpcs/temporal_mean/scattering_2d"][0].astype(np.float32)

    def _get_metadata(self):
        """Read instrument metadata.

        Tries ``/xpcs/qmap`` scalars first; falls back to
        ``/entry/instrument`` paths when any required key is absent.
        """
        try:
            return read_keymap(self.fname, _XPCS_QMAP_KEYMAP)
        except (KeyError, OSError):
            logger.info(
                "xpcs/qmap scalars incomplete in %s; falling back to /entry/instrument",
                self.fname,
            )
        return read_keymap(self.fname, _FALLBACK_KEYMAP)

    def _load_partition(self):
        """Load partition arrays from ``/xpcs/qmap``.

        Returns ``None`` if the group is absent or the required roi map
        arrays are missing — the image will still load normally.
        """
        try:
            with h5py.File(self.fname, "r") as f:
                if "/xpcs/qmap" not in f:
                    return None
                grp = f["/xpcs/qmap"]
                if "dynamic_roi_map" not in grp or "static_roi_map" not in grp:
                    return None
                partition = {}
                for key in _PARTITION_ARRAY_KEYS:
                    if key not in grp:
                        continue
                    val = grp[key][()]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    partition[key] = val
                for key in _XPCS_QMAP_KEYMAP:
                    if key in grp:
                        partition[key] = float(grp[key][()])
                return partition
        except Exception:
            logger.warning(
                "failed to load partition from %s", self.fname, exc_info=True
            )
            return None
