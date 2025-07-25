import logging

from ..base_reader import FileReader
from . import HdfDataset, ImmDataset, Rigaku3MDataset, RigakuDataset
from ..utils import get_metadata_from_keymap
import os
import glob
import h5py


logger = logging.getLogger(__file__)


# the value of each entry should be (value, unit, format_string)
DEFAULT_METADATA_WITHUNITS = {
    "energy": (10.0, "keV", "%.6f"),
    "detector_distance": (5.0, "m", "%.6f"),
    "pixel_size": (0.000172, "m", "%.6f"),
    "beam_center_x": (512.0, "pixel", "%.4f"),
    "beam_center_y": (256.0, "pixel", "%.4f"),
    "detector_shape_x": (1024, "pixel", "%.4f"),
    "detector_shape_y": (512, "pixel", "%.4f"),
    "swing_angle": (0.0, "degree", "%.4f"),
}

DEFAULT_METADATA = {key: value[0] for key, value in DEFAULT_METADATA_WITHUNITS.items()}

# Metadata key mappings for different HDF5 data formats
METADATA_KEYMAPS = {
    "energy": "/entry/instrument/incident_beam/incident_energy",
    "ccdx": "/entry/instrument/detector_1/position_x",
    "ccdy": "/entry/instrument/detector_1/position_y",
    "ccdx0": "/entry/instrument/detector_1/beam_center_position_x",
    "ccdy0": "/entry/instrument/detector_1/beam_center_position_y",
    "x_pixel_size": "/entry/instrument/detector_1/x_pixel_size",
    "y_pixel_size": "/entry/instrument/detector_1/y_pixel_size",
    "bcx0": "/entry/instrument/detector_1/beam_center_x",
    "bcy0": "/entry/instrument/detector_1/beam_center_y",
    "detector_distance": "/entry/instrument/detector_1/distance",
    "swing_angle": "/entry/instrument/detector_1/flightpath_swing",
}


def get_hdf_metadata(fname):
    """
    Read metadata from HDF5 files with fallback to default values.

    Args:
        fname: HDF5 file path

    Returns:
        dict: Metadata dictionary
    """
    prefix = os.path.join(os.path.dirname(fname), "*_metadata.hdf")
    meta_fnames = glob.glob(prefix)
    assert len(meta_fnames) > 0, f"no *_metadata.hdf found in the folder of {fname}"
    if len(meta_fnames) > 1:
        logger.warning(
            f"multiple *_metadata.hdf found in the folder of {fname}. using the first one"
        )
    meta_fname = meta_fnames[0]
    logger.info(f"using metadata file: {meta_fname}")

    try:
        # Use the keymap-based reader
        meta = get_metadata_from_keymap(meta_fname, METADATA_KEYMAPS)

        # Handle special case for swing_angle
        if meta.get("swing_angle") is None:
            logger.warning(
                "flight path swing angle not found metadata, set to 0.0 degree"
            )
            meta["swing_angle"] = 0.0

        meta["data_name"] = os.path.basename(meta_fname)

        # Calculate beam center positions with null checks
        try:
            ccdx, ccdx0 = meta["ccdx"], meta["ccdx0"]
            ccdy, ccdy0 = meta["ccdy"], meta["ccdy0"]

            meta["beam_center_x"] = meta["bcx0"] + (ccdx - ccdx0) / meta["x_pixel_size"]
            meta["beam_center_y"] = meta["bcy0"] + (ccdy - ccdy0) / meta["y_pixel_size"]
            meta["pixel_size"] = meta["x_pixel_size"]
        except (TypeError, KeyError) as e:
            logger.warning(f"Error calculating beam center positions: {e}")
            # Use default values if calculation fails
            meta["beam_center_x"] = DEFAULT_METADATA["beam_center_x"]
            meta["beam_center_y"] = DEFAULT_METADATA["beam_center_y"]
            meta["pixel_size"] = DEFAULT_METADATA["pixel_size"]

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

        # Fill in any missing values with defaults
        for key, default_value in DEFAULT_METADATA.items():
            if meta.get(key) is None:
                logger.debug(f"Using default value for {key}: {default_value}")
                meta[key] = default_value

    except Exception as e:
        logger.warning(f"Failed to read metadata from {meta_fname}: {e}")
        logger.info("Using default metadata values")
        # Use DEFAULT_METADATA as fallback
        meta = DEFAULT_METADATA.copy()
        meta["data_name"] = os.path.basename(meta_fname) if meta_fnames else "unknown"

    return meta


class APS8IDIReader(FileReader):
    def __init__(self, fname) -> None:
        super(APS8IDIReader, self).__init__(fname)
        self.handler = None
        self.ftype = "APS_8IDI"
        self.stype = "Transmission"
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
        else:
            logger.error("Unsupported APS dataset")
            return None
        self.shape = self.handler.det_size

    def get_scattering(self, **kwargs):
        return self.handler.get_scattering(**kwargs)

    def _get_metadata(self):
        return get_hdf_metadata(self.fname)
