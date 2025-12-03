import numpy as np
import h5py
import hdf5plugin
import logging
import os
import glob
import re


logger = logging.getLogger(__name__)


from multiprocessing import Pool, cpu_count


def create_pg_parameter_list(data_dict, metadata_withunits):
    """
    Create a parameter list for PyQtGraph from a data dictionary.

    Args:
        data_dict: Dictionary of parameter values
        metadata_withunits: Dictionary mapping parameter names to (value, unit, format_string) tuples

    Returns:
        list: List of parameter definitions for PyQtGraph
    """

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

        if key in metadata_withunits:
            # Unpack the metadata tuple
            _, unit, fmt_str = metadata_withunits[key]
            # Set the unit as a suffix for display
            line["suffix"] = f" {unit}"  # Add a leading space for readability
            line["siPrefix"] = False
            # This is crucial for custom formatting of floats
            # Parse format string to set the number of decimals
            match = re.search(r"%\.(\d+)f", fmt_str)
            if match:
                line["decimals"] = int(match.group(1))

        params.append(line)

    return params


def get_metadata_from_keymap(fname, metadata_keymaps, optional_fields=None):
    """
    Generic metadata reader using configurable key mappings.

    Args:
        fname: HDF5 file path
        metadata_keymaps: Dictionary mapping metadata keys to HDF5 paths
        optional_fields: List of field names that are optional and can be None if not found

    Returns:
        dict: Metadata dictionary with standardized keys
    """
    metadata = {}

    if optional_fields is None:
        optional_fields = []

    with h5py.File(fname, "r") as f:
        for key, hdf_path in metadata_keymaps.items():
            if key in optional_fields and hdf_path not in f:
                metadata[key] = None
            else:
                metadata[key] = f[hdf_path][()]

    return metadata


def process_chunk(args):
    """
    Process a chunk of frames.

    Args:
        args: tuple containing (file_path, dataset_name, start_idx, end_idx)
    """
    file_path, dataset_name, start_idx, end_idx = args
    with h5py.File(file_path, "r") as f:
        chunk = f[dataset_name][start_idx:end_idx].astype(np.float32)
        return np.sum(chunk, axis=0)


def sum_frames_parallel(
    file_path,
    dataset_name="/entry/data/data",
    start_frame=0,
    num_frames=-1,
    chunk_size=32,
    num_processes=None,
):
    """
    Sum frames from a HDF5 dataset using parallel processing.

    Parameters:
    -----------
    file_path : str
        Path to the HDF5 file
    dataset_name : str
        Name of the dataset in the HDF5 file
    start_frame : int
        Starting frame index (default: 0)
    num_frames : int or None
        Number of frames to sum. If None, will sum all remaining frames
    chunk_size : int
        Number of frames to process in each chunk
    num_processes : int or None
        Number of processes to use. If None, uses cpu_count()
    """
    # Open file to get shape information
    with h5py.File(file_path, "r") as f:
        dataset = f[dataset_name]
        assert dataset.ndim in [2, 3], "Dataset must be 2D or 3D"
        if dataset.ndim == 2:
            return dataset[()]

        total_frames = dataset.shape[0]
        logger.info(f"Total frames in dataset: {total_frames}")

        # Validate inputs
        if start_frame < 0 or start_frame >= total_frames:
            raise ValueError(f"start_frame must be between 0 and {total_frames - 1}")

        # If num_frames is None, use all remaining frames
        if num_frames is None or num_frames == 0:
            num_frames = total_frames - start_frame
        elif num_frames < 0:
            num_frames = max(1000, total_frames // 5)

        # Validate num_frames
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if (start_frame + num_frames) > total_frames:
            num_frames = total_frames - start_frame

        # If num_frames is small, process directly
        if num_frames < chunk_size:
            return np.sum(dataset[start_frame : start_frame + num_frames], axis=0)

        # Create chunks
        chunks = []
        for i in range(start_frame, start_frame + num_frames, chunk_size):
            end_idx = min(i + chunk_size, start_frame + num_frames)
            chunks.append((file_path, dataset_name, i, end_idx))

    # Set number of processes
    if num_processes is None:
        num_processes = cpu_count() // 2

    num_processes = min(len(chunks), num_processes)
    logger.info(f"using {num_processes} cores to load {num_frames} frames")
    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)

    # Sum all results
    return np.sum(np.array(results), axis=0)


def has_nexus_fields(fname, metadata_keymaps, optional_fields=None):
    """
    Check if an HDF5 file contains all required NeXus fields.

    Args:
        fname: Path to the HDF5 file
        fields: List of field names to check for
        optional_fields: List of field names that are optional and can be skipped

    Returns:
        bool: True if all required fields are present, False otherwise
    """
    if not h5py.is_hdf5(fname):
        return False

    if optional_fields is None:
        optional_fields = []

    with h5py.File(fname, "r") as f:
        for key, hdf_path in metadata_keymaps.items():
            if key in optional_fields:
                continue
            elif hdf_path not in f:
                return False
    return True


def find_metadata_same_folder(fname):
    """
    Find a metadata HDF5 file in the same folder as the given file.

    Args:
        fname: Path to the data file

    Returns:
        str: Path to the metadata file (*_metadata.hdf)

    Raises:
        FileNotFoundError: If no metadata file is found
        AssertionError: If no metadata file is found (legacy behavior)
    """
    prefix = os.path.join(os.path.dirname(fname), "*_metadata.hdf")
    meta_fnames = glob.glob(prefix)
    num_found = len(meta_fnames)
    assert num_found > 0, f"no *_metadata.hdf found in the folder of {fname}"
    if num_found >= 1:
        if num_found > 1:
            logger.warning(
                f"multiple *_metadata.hdf found in the folder of {fname}. using the first one"
            )
        return meta_fnames[0]
    elif num_found == 0:
        raise FileNotFoundError(f"no *_metadata.hdf found in the folder of {fname}")
