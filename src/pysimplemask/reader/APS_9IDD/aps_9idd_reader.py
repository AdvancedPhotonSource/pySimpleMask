import logging

from ..base_reader import FileReader
from ..utils import (find_metadata_same_folder, get_metadata_from_keymap,
                     has_nexus_fields, sum_frames_parallel)

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


def get_nexus_metadata(fname):
    """
    Read metadata from NeXus format HDF5 files.

    Args:
        fname: HDF5 file path
        flag_samefile: If True, read metadata from same file; if False, look for *_metadata.hdf

    Returns:
        dict: Metadata dictionary
    """
    if has_nexus_fields(fname, METADATA_KEYMAPS):
        meta_fname = fname
    else:
        meta_fname = find_metadata_same_folder(fname)
        if not has_nexus_fields(meta_fname, METADATA_KEYMAPS):
            raise FileNotFoundError(f"No valid metadata file found for {fname}.")

    # Use the keymap-based reader
    metadata = get_metadata_from_keymap(meta_fname, METADATA_KEYMAPS)
    m = metadata

    # Calculate beam center and specular positions with null checks
    beam_center_x = m["_bcx"] + (m["detector_x"] - m["_bc_det_x0"]) / m["pixel_size"]
    beam_center_y = m["_bcy"] + (m["detector_y"] - m["_bc_det_y0"]) / m["pixel_size"]
    specular_x = m["_spx"] + (m["detector_x"] - m["_sp_det_x0"]) / m["pixel_size"]
    specular_y = m["_spy"] + (m["detector_y"] - m["_sp_det_y0"]) / m["pixel_size"]

    # remove unused keys
    delete_keys = [key for key in metadata if key.startswith("_")]
    for key in delete_keys:
        metadata.pop(key)

    metadata["beam_center_x"] = round(beam_center_x, 3)
    metadata["beam_center_y"] = round(beam_center_y, 3)
    metadata["specular_x"] = round(specular_x, 3)
    metadata["specular_y"] = round(specular_y, 3)
    metadata["meta_fname"] = meta_fname
    return metadata


def get_metadata(fname):
    try:
        meta = get_nexus_metadata(fname)
    except Exception as e:
        logger.info("Failed to read metadata from file: %s. using default metadata.", e)
        meta = DEFAULT_METADATA.copy()
        meta["meta_fname"] = "default_metadata"
    meta["scattering_type"] = "Reflection"
    return meta


class APS9IDDReader(FileReader):
    def __init__(self, fname) -> None:
        super().__init__(fname)
        self.ftype = "APS_9IDD"
        self.stype = "Reflection"
        self.meta_units_fmts = DEFAULT_METADATA_WITHUNITS.copy()

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
        return get_metadata(self.fname)
