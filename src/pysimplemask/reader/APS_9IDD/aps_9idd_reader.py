import logging

import h5py
import os
import glob
from ..base_reader import FileReader
from ..utils import sum_frames_parallel
import re

logger = logging.getLogger(__name__)


# the value of each entry should be (value, unit)
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

# Metadata key mappings for different HDF5 data formats
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


def create_pg_parameter_list(data_dict):
    def get_param_type(value):
        """Determines the parameter type based on the value's Python type."""
        if isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        else:
            return "str"

    # Convert the dictionary to a list of parameter definitions
    params = []
    for key, value in data_dict.items():
        param_type = get_param_type(value)
        line = {"name": key, "type": param_type, "value": value}

        if key in DEFAULT_METADATA_WITHUNITS:
            # Unpack the metadata tuple
            _, unit, fmt_str = DEFAULT_METADATA_WITHUNITS[key]
            # Set the unit as a suffix for display
            line["suffix"] = f" {unit}"  # Add a leading space for readability
            line["siPrefix"] = False
            # This is crucial for custom formatting of floats
            # 2. Parse format string to set the number of decimals
            match = re.search(r"%\.(\d+)f", fmt_str)
            if match:
                line["decimals"] = int(match.group(1))

        params.append(line)

    return params


def get_metadata_from_keymap(fname):
    """
    Generic metadata reader using configurable key mappings.

    Args:
        fname: HDF5 file path
        keymap_name: Name of predefined keymap ("nexus" or "standard")
        custom_keymap: Optional custom keymap dictionary to override predefined ones

    Returns:
        dict: Metadata dictionary with standardized keys
    """
    metadata = {}
    with h5py.File(fname, "r") as f:
        for key, hdf_path in METADATA_KEYMAPS.items():
            try:
                metadata[key] = f[hdf_path][()]
            except KeyError:
                logger.debug(
                    f"Could not find HDF5 path '{hdf_path}' for key '{key}' in file {fname}"
                )
                metadata[key] = None

    return metadata


def get_metadata(fname, flag_samefile=True, detector_shape=(1000, 1000)):
    """
    Read metadata from NeXus format HDF5 files.

    Args:
        fname: HDF5 file path
        flag_samefile: If True, read metadata from same file; if False, look for *_metadata.hdf

    Returns:
        dict: Metadata dictionary
    """
    if flag_samefile:
        meta_fname = fname
    else:
        realpath = os.path.realpath(fname)
        prefix = os.path.join(os.path.dirname(realpath), "*_metadata.hdf")
        meta_fnames = glob.glob(prefix)
        assert len(meta_fnames) > 0, f"no *_metadata.hdf found in the folder of {fname}"
        if len(meta_fnames) > 1:
            logger.warning("More than one metadata file found, using the first one")
        meta_fname = meta_fnames[0]

    # Use the keymap-based reader
    metadata = get_metadata_from_keymap(meta_fname)
    m = metadata

    # Calculate beam center and specular positions with null checks
    try:
        beam_center_x = (
            m["_bcx"] + (m["detector_x"] - m["_bc_det_x0"]) / m["pixel_size"]
        )
        beam_center_y = (
            m["_bcy"] + (m["detector_y"] - m["_bc_det_y0"]) / m["pixel_size"]
        )
        specular_x = m["_spx"] + (m["detector_x"] - m["_sp_det_x0"]) / m["pixel_size"]
        specular_y = m["_spy"] + (m["detector_y"] - m["_sp_det_y0"]) / m["pixel_size"]
    except (TypeError, KeyError) as e:
        logger.warning(f"Error calculating beam/specular positions: {e}")
        # Use default values if calculation fails
        beam_center_x = DEFAULT_METADATA["beam_center_x"]
        beam_center_y = DEFAULT_METADATA["beam_center_y"]
        specular_x = DEFAULT_METADATA["specular_x"]
        specular_y = DEFAULT_METADATA["specular_y"]

    # remove unused keys
    delete_keys = [key for key in metadata if key.startswith("_")]
    for key in delete_keys:
        metadata.pop(key)

    metadata["beam_center_x"] = round(beam_center_x, 3)
    metadata["beam_center_y"] = round(beam_center_y, 3)
    metadata["specular_x"] = round(specular_x, 3)
    metadata["specular_y"] = round(specular_y, 3)
    metadata["detector_shape_x"] = detector_shape[1]
    metadata["detector_shape_y"] = detector_shape[0]

    return metadata


class APS9IDDReader(FileReader):
    def __init__(self, fname) -> None:
        super().__init__(fname)
        self.ftype = "APS_9IDD"
        self.stype = "Reflection"

    def get_scattering(self, num_frames=-1, begin_idx=0, num_processes=None):
        return sum_frames_parallel(
            self.fname,
            dataset_name="/entry/data/data",
            start_frame=begin_idx,
            num_frames=num_frames,
            chunk_size=32,
            num_processes=num_processes,
        )

    def _get_metadata(self, *args, **kwargs):
        """
        Read metadata from HDF5 file.

        Args:
            fname: HDF5 file path
            detector_shape: Shape of detector data for metadata calculation

        Returns:
            meta: metadata dictionary
        """
        with h5py.File(self.fname, "r") as f:
            dset = f["/entry/data/data"]
            if dset.ndim == 2:
                detector_shape = dset.shape
            elif dset.ndim == 3:
                detector_shape = dset.shape[-2:]  # Use last two dimensions

        try:
            meta = get_metadata(self.fname, detector_shape=detector_shape)
        except Exception as e:
            logger.warning(f"Failed to read metadata from {self.fname}: {e}")
            logger.info("Using default metadata values")
            # Use METADATA_DEFAULT_WITHUNITS and extract values from tuples
            meta = DEFAULT_METADATA.copy()
            # Update detector shape if we have it
            if detector_shape is not None:
                meta["detector_shape_x"] = detector_shape[1]
                meta["detector_shape_y"] = detector_shape[0]
        meta["bcx"] = meta["beam_center_x"]
        meta["bcy"] = meta["beam_center_y"]
        meta["pix_dim"] = meta["pixel_size"]
        meta["det_dist"] = meta["detector_distance"]
        return meta
