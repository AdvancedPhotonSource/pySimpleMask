import logging

from ..base_reader import FileReader
from . import HdfDataset, ImmDataset, Rigaku3MDataset, RigakuDataset
from ..utils import (
    get_metadata_from_keymap,
    has_nexus_fields,
    find_metadata_same_folder,
)
import traceback


logger = logging.getLogger(__file__)


# the value of each entry should be (value, unit, format_string)
DEFAULT_METADATA_WITHUNITS = {
    "energy": (10.0, "keV", "%.6f"),
    "detector_distance": (5.0, "m", "%.6f"),
    "swing_angle": (0.0, "degree", "%.4f"),
    "beam_center_x": (512.0, "pixel", "%.4f"),
    "beam_center_y": (256.0, "pixel", "%.4f"),
    "pixel_size": (0.000075, "m", "%.6f"),
    "detector_shape_x": (1024, "pixel", "%.4f"),
    "detector_shape_y": (512, "pixel", "%.4f"),
}

DEFAULT_METADATA = {key: value[0] for key, value in DEFAULT_METADATA_WITHUNITS.items()}

# Metadata key mappings for different HDF5 data formats
METADATA_KEYMAPS = {
    "energy": "/entry/instrument/incident_beam/incident_energy",
    "detector_distance": "/entry/instrument/detector_1/distance",
    "swing_angle": "/entry/instrument/detector_1/flightpath_swing",
    "x_pixel_size": "/entry/instrument/detector_1/x_pixel_size",
    "y_pixel_size": "/entry/instrument/detector_1/y_pixel_size",
    "ccdx": "/entry/instrument/detector_1/position_x",
    "ccdy": "/entry/instrument/detector_1/position_y",
    "ccdx0": "/entry/instrument/detector_1/beam_center_position_x",
    "ccdy0": "/entry/instrument/detector_1/beam_center_position_y",
    "bcx0": "/entry/instrument/detector_1/beam_center_x",
    "bcy0": "/entry/instrument/detector_1/beam_center_y",
}


def get_nexus_metadata(fname):
    """
    Read metadata from HDF5 files with fallback to default values.

    Args:
        fname: HDF5 file path

    Returns:
        dict: Metadata dictionary
    """
    optional_fields = ["swing_angle"]
    if has_nexus_fields(fname, METADATA_KEYMAPS, optional_fields):
        meta_fname = fname
    else:
        meta_fname = find_metadata_same_folder(fname)
        if not has_nexus_fields(meta_fname, METADATA_KEYMAPS, optional_fields):
            logger.error(f"{meta_fname} does not have the minimal nexus fields.")
            raise FileNotFoundError(f"No valid metadata found in {meta_fname}")

    logger.info(f"using metadata file: {meta_fname}")
    # Use the keymap-based reader; swing_angle is optional and defaults to None
    meta = get_metadata_from_keymap(meta_fname, METADATA_KEYMAPS, optional_fields)

    # Handle special case for swing_angle
    if meta.get("swing_angle") is None:
        logger.warning("flight path swing angle not found metadata, set to 0.0 degree")
        meta["swing_angle"] = 0.0

    # Calculate beam center positions with null checks
    ccdx, ccdx0 = meta["ccdx"], meta["ccdx0"]
    ccdy, ccdy0 = meta["ccdy"], meta["ccdy0"]

    meta["beam_center_x"] = meta["bcx0"] + (ccdx - ccdx0) / meta["x_pixel_size"]
    meta["beam_center_y"] = meta["bcy0"] + (ccdy - ccdy0) / meta["y_pixel_size"]
    meta["pixel_size"] = meta["x_pixel_size"]

    # Clean up intermediate values
    for key in [
        "bcx0",
        "bcy0",
        "ccdx",
        "ccdy",
        "ccdx0",
        "ccdy0",
        "x_pixel_size",
        "y_pixel_size",
    ]:
        meta.pop(key, None)

    meta["meta_fname"] = meta_fname
    return meta


def get_metadata(fname):
    try:
        meta = get_nexus_metadata(fname)
    except Exception as e:
        logger.info(
            f"Failed to read metadata from file: {fname} because {e}. using default metadata."
        )
        traceback.print_exc()
        meta = DEFAULT_METADATA.copy()
        meta["meta_fname"] = "default_metadata"
    meta["scattering_type"] = "Transmission"
    return meta


class APS8IDIReader(FileReader):
    def __init__(self, fname) -> None:
        super(APS8IDIReader, self).__init__(fname)
        self.handler = None
        self.ftype = "APS_8IDI"
        self.stype = "Transmission"
        self.shape = (0, 0)
        self.meta_units_fmts = DEFAULT_METADATA_WITHUNITS.copy()

        rigaku_endings = tuple(f".bin.00{i}" for i in range(6))

        if fname.endswith(".bin"):
            logger.info("Rigaku 500k dataset")
            self.handler = RigakuDataset(fname, batch_size=1000)
        elif fname.endswith(rigaku_endings):
            logger.info("Rigaku 3M (6 x 500K) dataset")
            self.handler = Rigaku3MDataset(fname, batch_size=1000)
        elif fname.endswith(".imm"):
            logger.info("IMM dataset")
            self.handler = ImmDataset(fname, batch_size=100)
        elif fname.endswith(".h5") or fname.endswith(".hdf"):
            logger.info("APS HDF dataset")
            self.handler = HdfDataset(fname, batch_size=100)
        elif fname.endswith(".tpx") or fname.endswith(".tpx.000"):
            from timepix_dataset.dataset import TimepixRawDataset

            self.handler = TimepixRawDataset(fname)
        else:
            logger.error("Unsupported APS dataset")
            return None
        self.shape = self.handler.det_size

    def get_scattering(self, **kwargs):
        return self.handler.get_scattering(**kwargs)

    def _get_metadata(self):
        return get_metadata(self.fname)
