# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""APS 9-ID-D reader (reflection / GISAXS-XPCS)."""

import logging

from ..base_reader import FileReader
from ..formats.hdf import HdfDataset
from ..metadata import read_nexus_metadata

logger = logging.getLogger(__name__)


# Each entry is (default_value, unit, format_string).
DEFAULT_METADATA_WITHUNITS = {
    "energy": (10.92, "keV", "%.6f"),
    "detector_distance": (0.228165, "m", "%.6f"),
    "pixel_size": (0.000172, "m", "%.6f"),
    "incident_angle": (0.14, "degree", "%.4f"),
    "exposure_time": (1.0, "s", "%.4f"),
    "detector_x": (0.00, "m", "%.6f"),
    "detector_y": (0.00, "m", "%.6f"),
    "orientation": (0.00, "degree", "%.4f"),
    "beam_center_x": (813.0, "pixel", "%.4f"),
    "beam_center_y": (1020.0, "pixel", "%.4f"),
    "specular_x": (813.10, "pixel", "%.4f"),
    "specular_y": (1050.2, "pixel", "%.4f"),
    "detector_shape_x": (981, "pixel", "%.4f"),
    "detector_shape_y": (1043, "pixel", "%.4f"),
}

DEFAULT_METADATA = {key: value[0] for key, value in DEFAULT_METADATA_WITHUNITS.items()}

# Metadata key -> NeXus HDF5 path. Keys prefixed with "_" are intermediate.
METADATA_KEYMAPS = {
    "energy": "/entry/instrument/incident_beam/incident_energy",
    "detector_distance": "/entry/instrument/detector_1/current_stage_z",
    "pixel_size": "/entry/instrument/detector_1/x_pixel_size",
    "incident_angle": "/entry/sample/rotation_x",
    "exposure_time": "/entry/instrument/detector_1/count_time",
    "detector_x": "/entry/instrument/detector_1/current_stage_x",
    "detector_y": "/entry/instrument/detector_1/current_stage_y",
    "orientation": "/entry/instrument/detector_1/rotation_z",
    "_bcx": "/entry/instrument/detector_1/direct_beam_image_x",
    "_bcy": "/entry/instrument/detector_1/direct_beam_image_y",
    "_bc_det_x0": "/entry/instrument/detector_1/direct_beam_stage_x",
    "_bc_det_y0": "/entry/instrument/detector_1/direct_beam_stage_y",
    "_spx": "/entry/instrument/detector_1/specular_beam_image_x",
    "_spy": "/entry/instrument/detector_1/specular_beam_image_y",
    "_sp_det_x0": "/entry/instrument/detector_1/specular_beam_stage_x",
    "_sp_det_y0": "/entry/instrument/detector_1/specular_beam_stage_y",
}


def get_nexus_metadata(fname):
    """Read 9-ID-D NeXus metadata and derive beam-center / specular positions."""
    meta, _meta_fname = read_nexus_metadata(fname, METADATA_KEYMAPS)

    pixel_size = meta["pixel_size"]
    beam_center_x = (
        meta["_bcx"] + (meta["detector_x"] - meta["_bc_det_x0"]) / pixel_size
    )
    beam_center_y = (
        meta["_bcy"] + (meta["detector_y"] - meta["_bc_det_y0"]) / pixel_size
    )
    specular_x = meta["_spx"] + (meta["detector_x"] - meta["_sp_det_x0"]) / pixel_size
    specular_y = meta["_spy"] + (meta["detector_y"] - meta["_sp_det_y0"]) / pixel_size

    for key in [k for k in meta if k.startswith("_")]:
        meta.pop(key)

    meta["beam_center_x"] = round(beam_center_x, 3)
    meta["beam_center_y"] = round(beam_center_y, 3)
    meta["specular_x"] = round(specular_x, 3)
    meta["specular_y"] = round(specular_y, 3)
    return meta


def get_metadata(fname):
    """Return 9-ID-D metadata, falling back to defaults on any failure."""
    try:
        return get_nexus_metadata(fname)
    except Exception:
        logger.info(
            "Failed to read metadata from %s; using defaults.", fname, exc_info=True
        )
        meta = DEFAULT_METADATA.copy()
        meta["meta_fname"] = "default_metadata"
        return meta


class APS9IDDReader(FileReader):
    def __init__(self, fname) -> None:
        super().__init__(fname)
        self.ftype = "APS_9IDD"
        self.stype = "Reflection"
        self.meta_units_fmts = DEFAULT_METADATA_WITHUNITS.copy()
        self.loader = HdfDataset(fname, data_path="/entry/data/data")
        self.shape = tuple(self.loader.det_size)

    def get_scattering(self, **kwargs):
        return self.loader.get_scattering(**kwargs)

    def _get_metadata(self):
        return get_metadata(self.fname)
