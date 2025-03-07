import hashlib
import numpy as np
from typing import Dict, Union
import json


def hash_numpy_dict(input_dictionary):
    """
    Computes a stable SHA256 hash for a dictionary containing NumPy arrays and lists of strings.

    Parameters:
        dictionary (dict): Dictionary with NumPy arrays and lists of strings.

    Returns:
        str: A SHA256 hash of the dictionary.
    """
    hasher = hashlib.sha256()

    for key in sorted(input_dictionary.keys()):  # Sort keys for consistency
        hasher.update(str(key).encode())

        value = input_dictionary[key]

        if isinstance(value, np.ndarray):
            # Ensure consistent dtype & memory layout
            value = np.ascontiguousarray(value)
            hasher.update(value.astype(value.dtype.newbyteorder('='))  # Force consistent endianness
                         .tobytes())

        elif isinstance(value, list):
            # Convert list of strings to JSON for consistent encoding
            hasher.update(json.dumps(value, sort_keys=True).encode())

        else:
            # Convert other types to a JSON string for stability
            hasher.update(json.dumps(value, sort_keys=True).encode())

    return hasher.hexdigest()


def optimize_integer_array(arr):
    """
    Optimizes the data type of a NumPy array of integers to minimize memory usage.

    Args:
        arr: A NumPy array of integers.

    Returns:
        A NumPy array with the optimized data type, or the original array if
        the input is not a NumPy array of integers or if it's empty.
    """

    if not isinstance(arr, np.ndarray) or arr.size == 0:
        return arr  # Return original if not a numpy array or if empty

    if not np.issubdtype(arr.dtype, np.integer):
        return arr  # Return original if not an integer type

    min_val, max_val = arr.min(), arr.max()  # Get min/max values

    # Choose smallest dtype based on min/max
    if min_val >= 0:  # Unsigned types
        if max_val <= np.iinfo(np.uint8).max:
            new_dtype = np.uint8
        elif max_val <= np.iinfo(np.uint16).max:
            new_dtype = np.uint16
        elif max_val <= np.iinfo(np.uint32).max:
            new_dtype = np.uint32
        else:
            new_dtype = np.uint64
    else:  # Signed types
        if min_val >= np.iinfo(np.int8).min and max_val <= np.iinfo(np.int8).max:
            new_dtype = np.int8
        elif min_val >= np.iinfo(np.int16).min and max_val <= np.iinfo(np.int16).max:
            new_dtype = np.int16
        elif min_val >= np.iinfo(np.int32).min and max_val <= np.iinfo(np.int32).max:
            new_dtype = np.int32
        else:
            new_dtype = np.int64

    return arr.astype(new_dtype) if new_dtype != arr.dtype else arr


def generate_partition(
    map_name: str,
    mask: np.ndarray,
    xmap: np.ndarray,
    num_pts: int,
    style: str = 'linear',
    phi_offset: Union[float, None] = None,
    symmetry_fold: int = 1,
) -> Dict[str, Union[str, int, np.ndarray]]:
    """
    Generates a partition map for X-ray scattering analysis.
    """
    if map_name == 'phi':
        xmap_phi = xmap.copy()
        if phi_offset is not None:
            xmap = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(xmap + phi_offset))))
        if symmetry_fold > 1:
            unit_xmap =  (xmap < (360 / symmetry_fold)) * (xmap >= 0)
            xmap = xmap + 180.0  # [0, 360]
            xmap = np.mod(xmap, 360.0 / symmetry_fold)

    roi = mask > 0
    v_min = np.nanmin(xmap[roi])
    v_max = np.nanmax(xmap[roi])

    if map_name == 'q' and style == 'logarithmic':
        mask = mask * (xmap > 0)
        valid_xmap = xmap[mask > 0]
        if valid_xmap.size == 0 or np.all(np.isnan(valid_xmap)):
            raise ValueError("Invalid `xmap` values for logarithmic binning. All values are non-positive.")
        v_min = np.nanmin(valid_xmap)
        xmap = np.where(xmap > 0, xmap, np.nan)  # Avoid modifying input
        v_span = np.logspace(np.log10(v_min), np.log10(v_max), num_pts + 1,
                             base=10)
        v_list = np.sqrt(v_span[1:] * v_span[:-1])
    else:
        v_span = np.linspace(v_min, v_max, num_pts + 1)
        v_list = (v_span[1:] + v_span[:-1]) / 2.0

    partition = np.digitize(xmap, v_span).astype(np.uint32) * mask
    partition[partition > num_pts] = 0
    # Ensure the maximum value (excluding unmasked) is assigned to the last bin
    partition[(xmap == v_max) * mask] = num_pts

    if map_name == 'phi' and symmetry_fold > 1:
        # get the average phi value for each partition, at the first fold
        idx_map = unit_xmap * partition
        sum_value = np.bincount(idx_map.flatten(), weights=xmap_phi.flatten()) 
        norm_factor = np.bincount(idx_map.flatten())
        v_list = sum_value / np.clip(norm_factor, 1, None)
        v_list = v_list[1:]

    return {'map_name': map_name, 'num_pts': num_pts, 'partition': partition,
            'v_list': v_list}


def combine_partitions(
    pack1: Dict[str, Union[str, int, np.ndarray]],
    pack2: Dict[str, Union[str, int, np.ndarray]],
    prefix: str = 'dynamic'
) -> Dict[str, Union[list, np.ndarray]]:
    """
    Combines two partition maps into a single partition space.

    This function merges two partition dictionaries (typically representing
    different dimensions such as 'q' and 'phi', or 'x' and 'y') into a
    combined partition index map.

    Parameters
    ----------
    pack1 : Dict[str, Union[str, int, np.ndarray]]
        First partition dictionary containing:
        - 'map_name' (str): Name of the first partition (e.g., 'q', 'x').
        - 'num_pts' (int): Number of points in the first partition.
        - 'partition' (np.ndarray): The partition array.
        - 'v_list' (np.ndarray): The bin center values.

    pack2 : Dict[str, Union[str, int, np.ndarray]]
        Second partition dictionary containing:
        - 'map_name' (str): Name of the second partition (e.g., 'phi', 'y').
        - 'num_pts' (int): Number of points in the second partition.
        - 'partition' (np.ndarray): The partition array.
        - 'v_list' (np.ndarray): The bin center values.

    prefix : str, optional
        Prefix to be used for naming keys in the output dictionary.
        Default is `'dynamic'`.

    Returns
    -------
    Dict[str, Union[list, np.ndarray]]
        A dictionary containing:
        - `'{prefix}_map_names'` (list of str): The names of the combined maps.
        - `'{prefix}_num_pts'` (list of int): The number of bins for each partition.
        - `'{prefix}_roi_map'` (np.ndarray): The combined partition index map.
        - `'{prefix}_v_list_dim0'` (np.ndarray): Bin center values of the first partition.
        - `'{prefix}_v_list_dim1'` (np.ndarray): Bin center values of the second partition.
        - `'{prefix}_index_mapping'` (np.ndarray): Unique partition indices after combination.

    Raises
    ------
    AssertionError
        If the provided `map_name` pairs are not valid. Only allowed pairs are:
        - ('q', 'phi')
        - ('x', 'y')
    """
    assert (pack1['map_name'], pack2['map_name']) in [('q', 'phi'), ('x', 'y')], \
        "Invalid partition pair. Allowed pairs: ('q', 'phi') or ('x', 'y')"

    # Convert partitions to zero-based indexing, then merge
    partition = (pack1['partition'].astype(np.int64) - 1) * pack2['num_pts'] + \
                (pack2['partition'].astype(np.int64) - 1) + 1  # Convert back to one-based

    # Ensure valid range
    partition = np.clip(partition, a_min=0, a_max=None).astype(np.uint32)

    # Get unique values and remap indices
    unique_idx, inverse = np.unique(partition, return_inverse=True)
    partition_natural_order = inverse.reshape(partition.shape).astype(np.uint32)

    # Construct output dictionary with correct prefix
    partition_pack = {
        f'{prefix}_num_pts': [pack1['num_pts'], pack2['num_pts']],
        f'{prefix}_roi_map': partition_natural_order,
        f'{prefix}_v_list_dim0': pack1['v_list'],
        f'{prefix}_v_list_dim1': pack2['v_list'],
        f'{prefix}_index_mapping': unique_idx[unique_idx >= 1] - 1,
    }

    return partition_pack


def check_consistency(dqmap: np.ndarray, sqmap: np.ndarray) -> bool:
    """
    Check the consistency of dqmap and sqmap efficiently.

    Ensures that each unique value in `sqmap` corresponds to only one unique value in `dqmap`.

    Parameters
    ----------
    dqmap : np.ndarray
        A 2D NumPy array representing the dqmap.
    sqmap : np.ndarray
        A 2D NumPy array representing the sqmap.

    Returns
    -------
    bool
        True if each unique value in `sqmap` maps to exactly one unique value in `dqmap`,
        False otherwise.

    Raises
    ------
    ValueError
        If `dqmap` and `sqmap` do not have the same shape.
    """
    if dqmap.shape != sqmap.shape:
        raise ValueError("dqmap and sqmap must have the same shape")

    # Flatten arrays for efficient processing
    sq_flat = sqmap.ravel()
    dq_flat = dqmap.ravel()

    # Dictionary to store mapping from sqmap values to dqmap values
    sq_to_dq: dict[int, int] = {}

    for sq_value, dq_value in zip(sq_flat, dq_flat):
        if sq_value in sq_to_dq:
            if sq_to_dq[sq_value] != dq_value:
                print(sq_value, dq_value)
                return False  # Inconsistent mapping found
        else:
            sq_to_dq[sq_value] = dq_value

    return True