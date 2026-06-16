"""APS 8-ID-I reader (transmission SAXS/XPCS)."""

import logging

from ..base_reader import FileReader
from ..formats import get_format_loader
from ..metadata import read_nexus_metadata

logger = logging.getLogger(__name__)


# Each entry is (default_value, unit, format_string).
DEFAULT_METADATA_WITHUNITS = {
    "energy": (10.0, "keV", "%.6f"),
    "detector_distance": (5.0, "m", "%.6f"),
    "swing_angle_horizontal": (0.0, "degree", "%.4f"),
    "swing_angle_vertical": (0.0, "degree", "%.4f"),
    "beam_center_x": (512.0, "pixel", "%.4f"),
    "beam_center_y": (256.0, "pixel", "%.4f"),
    "pixel_size": (0.000075, "m", "%.6f"),
    "detector_shape_x": (1024, "pixel", "%.4f"),
    "detector_shape_y": (512, "pixel", "%.4f"),
}

DEFAULT_METADATA = {key: value[0] for key, value in DEFAULT_METADATA_WITHUNITS.items()}

# Metadata key -> NeXus HDF5 path.
METADATA_KEYMAPS = {
    "energy": "/entry/instrument/incident_beam/incident_energy",
    "detector_distance": "/entry/instrument/detector_1/distance",
    "swing_angle_horizontal": "/entry/instrument/detector_1/flightpath_swing",
    "swing_angle_vertical": "/entry/instrument/detector_1/flightpath_swing_vertical",
    "x_pixel_size": "/entry/instrument/detector_1/x_pixel_size",
    "y_pixel_size": "/entry/instrument/detector_1/y_pixel_size",
    "ccdx": "/entry/instrument/detector_1/position_x",
    "ccdy": "/entry/instrument/detector_1/position_y",
    "ccdx0": "/entry/instrument/detector_1/beam_center_position_x",
    "ccdy0": "/entry/instrument/detector_1/beam_center_position_y",
    "bcx0": "/entry/instrument/detector_1/beam_center_x",
    "bcy0": "/entry/instrument/detector_1/beam_center_y",
}

OPTIONAL_FIELDS = ["swing_angle_horizontal", "swing_angle_vertical"]


def get_nexus_metadata(fname):
    """Read 8-ID-I NeXus metadata and derive the beam center."""
    meta, _meta_fname = read_nexus_metadata(fname, METADATA_KEYMAPS, OPTIONAL_FIELDS)

    for key in ("swing_angle_horizontal", "swing_angle_vertical"):
        if meta.get(key) is None:
            logger.warning("%s not found in metadata, set to 0.0 degree", key)
            meta[key] = 0.0

    # Beam center = recorded center + detector translation in pixels.
    meta["beam_center_x"] = (
        meta["bcx0"] + (meta["ccdx"] - meta["ccdx0"]) / meta["x_pixel_size"]
    )
    meta["beam_center_y"] = (
        meta["bcy0"] + (meta["ccdy"] - meta["ccdy0"]) / meta["y_pixel_size"]
    )
    meta["pixel_size"] = meta["x_pixel_size"]

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
        meta.pop(key, None)

    return meta


def get_metadata(fname):
    """Return 8-ID-I metadata, falling back to defaults on any failure."""
    try:
        return get_nexus_metadata(fname)
    except Exception:
        logger.info(
            "Failed to read metadata from %s; using defaults.", fname, exc_info=True
        )
        meta = DEFAULT_METADATA.copy()
        meta["meta_fname"] = "default_metadata"
        return meta


class APS8IDIReader(FileReader):
    def __init__(self, fname) -> None:
        super().__init__(fname)
        self.ftype = "APS_8IDI"
        self.stype = "Transmission"
        self.meta_units_fmts = DEFAULT_METADATA_WITHUNITS.copy()
        self.loader = get_format_loader(fname)
        self.shape = tuple(self.loader.det_size)

    def get_scattering(self, **kwargs):
        return self.loader.get_scattering(**kwargs)

    def _get_metadata(self):
        return get_metadata(self.fname)
