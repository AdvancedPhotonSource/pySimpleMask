# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import contextlib
import hashlib
import json
import logging
import shutil
from typing import Dict, Union

import h5py
import numpy as np

logger = logging.getLogger(__name__)


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
            hasher.update(
                value.astype(
                    value.dtype.newbyteorder("=")
                ).tobytes()  # Force consistent endianness
            )

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
    style: str = "linear",
    phi_offset: Union[float, None] = None,
    symmetry_fold: int = 1,
) -> Dict[str, Union[str, int, np.ndarray]]:
    """
    Generates a partition map for X-ray scattering analysis.
    """
    if map_name == "phi":
        xmap_phi = xmap.copy()
        if phi_offset is not None:
            xmap = np.rad2deg(np.angle(np.exp(1j * np.deg2rad(xmap + phi_offset))))
        if symmetry_fold > 1:
            unit_xmap = (xmap < (360 / symmetry_fold)) * (xmap >= 0)
            xmap = xmap + 180.0  # [0, 360]
            xmap = np.mod(xmap, 360.0 / symmetry_fold)

    roi = mask > 0
    if not roi.any():
        # No unmasked pixels — return an all-zero partition with empty value list.
        return {
            "map_name": map_name,
            "num_pts": num_pts,
            "partition": np.zeros_like(mask, dtype=np.uint32),
            "v_list": np.zeros(num_pts, dtype=np.float64),
        }
    v_min = np.nanmin(xmap[roi])
    v_max = np.nanmax(xmap[roi])

    if map_name == "q" and style == "logarithmic":
        mask = mask * (xmap > 0)
        valid_xmap = xmap[mask > 0]
        if valid_xmap.size == 0 or np.all(np.isnan(valid_xmap)):
            raise ValueError(
                "Invalid `xmap` values for logarithmic binning. All values are non-positive."
            )
        v_min = np.nanmin(valid_xmap)
        xmap = np.where(xmap > 0, xmap, np.nan)  # Avoid modifying input
        v_span = np.logspace(np.log10(v_min), np.log10(v_max), num_pts + 1, base=10)
        v_list = np.sqrt(v_span[1:] * v_span[:-1])
    else:
        v_span = np.linspace(v_min, v_max, num_pts + 1)
        v_list = (v_span[1:] + v_span[:-1]) / 2.0

    # np.digitize is very sensitive to floating point precision, so we round the values
    # to 12 decimal places to avoid issues with values that are very close to the bin edges.
    # e.g. xmap = 0.0015398376679056104, v_span 0.0015398376679056109 yield 0
    partition = (
        np.digitize(np.round(xmap, 12), np.round(v_span, 12)).astype(np.uint32) * mask
    )
    partition[partition > num_pts] = 0
    # Ensure the maximum value (excluding unmasked) is assigned to the last bin
    partition[(xmap == v_max) * mask] = num_pts
    # Floating-point rounding can leave a small number of valid pixels unassigned
    # (bin=0) even after the above correction; assign them to the first bin.
    leftover = roi & (partition == 0)
    if leftover.any():
        partition[leftover] = 1

    if map_name == "phi" and symmetry_fold > 1:
        # get the average phi value for each partition, at the first fold
        idx_map = unit_xmap * partition
        sum_value = np.bincount(idx_map.flatten(), weights=xmap_phi.flatten())
        norm_factor = np.bincount(idx_map.flatten())
        v_list = sum_value / np.clip(norm_factor, 1, None)
        v_list = v_list[1:]

    return {
        "map_name": map_name,
        "num_pts": num_pts,
        "partition": partition,
        "v_list": v_list,
    }


def generate_groupindex_partitions(
    map_name: str,
    group_index_map: np.ndarray,
    num_groups: int,
    xmap: np.ndarray,
    dq_num_per_group: int,
    sq_num_per_group: int,
    style: str = "linear",
    phi_offset: Union[float, None] = None,
    symmetry_fold: int = 1,
) -> tuple:
    """
    Builds dq/sq-equivalent partition packs from a pre-labeled group-index map
    instead of a linear/log rebin of the whole ROI. Used for either axis of a
    partition (radial or angular) — call it once per axis with that axis's own
    ``map_name``/``xmap``.

    Each group (a value 1..num_groups in ``group_index_map``, 0 = excluded) is
    treated as an independent sub-ROI: it gets its own ``dq_num_per_group`` dynamic
    bins and ``sq_num_per_group`` static bins, computed from that group's own value
    range, then all groups' sub-partitions are offset and combined into one overall
    pack. Groups are assumed spatially disjoint. Returned packs have the same shape
    as ``generate_partition``'s output and plug directly into ``combine_partitions``
    — including when called once per axis: combining two per-group-offset packs
    naturally confines each group's combined bins to their own block, since a given
    pixel's two axis values are both offset by that same pixel's own group.

    ``phi_offset``/``symmetry_fold`` are forwarded to ``generate_partition`` for
    each group and only take effect when ``map_name == "phi"``.

    As with the plain (non-group) dq/sq path, ``check_consistency`` only holds if
    each group's local static bin-edges refine its local dynamic bin-edges, which
    for linear/log binning requires ``sq_num_per_group`` to be a multiple of
    ``dq_num_per_group``.

    Returns
    -------
    tuple[dict, dict]
        ``(pack_dq, pack_sq)`` — ``pack_dq`` has ``num_pts == num_groups *
        dq_num_per_group``; ``pack_sq`` has ``num_pts == num_groups *
        sq_num_per_group``.
    """
    dq_partition = np.zeros(group_index_map.shape, dtype=np.uint32)
    dq_v_list = []
    sq_partition = np.zeros(group_index_map.shape, dtype=np.uint32)
    sq_v_list = []

    for g in range(1, num_groups + 1):
        sub_mask = group_index_map == g

        pack_dq_g = generate_partition(
            map_name, sub_mask, xmap, dq_num_per_group, style=style,
            phi_offset=phi_offset, symmetry_fold=symmetry_fold,
        )
        local_dq = pack_dq_g["partition"]
        dq_offset = (g - 1) * dq_num_per_group
        dq_partition[local_dq > 0] = local_dq[local_dq > 0] + dq_offset
        dq_v_list.append(pack_dq_g["v_list"])

        pack_sq_g = generate_partition(
            map_name, sub_mask, xmap, sq_num_per_group, style=style,
            phi_offset=phi_offset, symmetry_fold=symmetry_fold,
        )
        local_sq = pack_sq_g["partition"]
        sq_offset = (g - 1) * sq_num_per_group
        sq_partition[local_sq > 0] = local_sq[local_sq > 0] + sq_offset
        sq_v_list.append(pack_sq_g["v_list"])

    pack_dq = {
        "map_name": map_name,
        "num_pts": num_groups * dq_num_per_group,
        "partition": dq_partition,
        "v_list": np.concatenate(dq_v_list) if dq_v_list else np.zeros(0),
    }
    pack_sq = {
        "map_name": map_name,
        "num_pts": num_groups * sq_num_per_group,
        "partition": sq_partition,
        "v_list": np.concatenate(sq_v_list) if sq_v_list else np.zeros(0),
    }
    return pack_dq, pack_sq


def combine_partitions(
    pack1: Dict[str, Union[str, int, np.ndarray]],
    pack2: Dict[str, Union[str, int, np.ndarray]],
    prefix: str = "dynamic",
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
    # assert (pack1["map_name"], pack2["map_name"]) in [
    #     ("q", "phi"),
    #     ("x", "y"),
    # ], "Invalid partition pair. Allowed pairs: ('q', 'phi') or ('x', 'y')"

    # Convert partitions to zero-based indexing, then merge
    partition = (
        (pack1["partition"].astype(np.int64) - 1) * pack2["num_pts"]
        + (pack2["partition"].astype(np.int64) - 1)
        + 1
    )  # Convert back to one-based

    # Ensure valid range
    partition = np.clip(partition, a_min=0, a_max=None).astype(np.uint32)

    # some qmap may not have any bad pixels, so the partition may start from 1
    start_index = np.min(partition)
    # Get unique values and remap indices
    unique_idx, inverse = np.unique(partition, return_inverse=True)
    partition_natural_order = inverse.reshape(partition.shape).astype(np.uint32)

    # if start_index is 0, then the partition_natural_order is already correct
    # otherwise, we need to shift the partition_natural_order by 1, so that the
    # first index is 0. otherwise the first index will be 0, which marks this
    # partition as bad pixels.
    if start_index > 0:
        partition_natural_order += 1

    # Construct output dictionary with correct prefix
    partition_pack = {
        f"{prefix}_num_pts": [pack1["num_pts"], pack2["num_pts"]],
        f"{prefix}_roi_map": partition_natural_order,
        f"{prefix}_v_list_dim0": pack1["v_list"],
        f"{prefix}_v_list_dim1": pack2["v_list"],
        f"{prefix}_index_mapping": unique_idx[unique_idx >= 1] - 1,
    }

    return partition_pack


def check_consistency(dqmap: np.ndarray, sqmap: np.ndarray, mask: np.ndarray) -> bool:
    """
    Check the consistency of dqmap and sqmap efficiently.

    Ensures that each unique value in `sqmap` corresponds to only one unique value in `dqmap`.

    Parameters
    ----------
    dqmap : np.ndarray
        A 2D NumPy array representing the dqmap.
    sqmap : np.ndarray
        A 2D NumPy array representing the sqmap.
    mask : np.ndarray
        A 2D NumPy array representing the mask.

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
    if dqmap.shape != mask.shape:
        raise ValueError("dqmap and mask must have the same shape")

    if not np.all((mask > 0) == (dqmap > 0)):
        logger.warning("dqmap and mask have mismatched valid pixels")
        return False
    if not np.all((mask > 0) == (sqmap > 0)):
        logger.warning("sqmap and mask have mismatched valid pixels")
        return False

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


def combine_qmap_files(qmap_files, output_file):
    """
    Combine two or more qmap files into a single qmap file.

    Parameters
    ----------
    qmap_files : list[str]
        Paths to the qmap files to combine, in order. Must contain at least two.
    output_file : str
        Path to the output qmap file.
    """
    if len(qmap_files) < 2:
        raise ValueError(
            f"combine_qmap_files requires at least two input files, got {len(qmap_files)}"
        )

    logger.info("Combining %d qmap files:", len(qmap_files))
    for i, fname in enumerate(qmap_files):
        logger.info("  file[%d]: %s", i, fname)
    logger.info("  output  : %s", output_file)

    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(h5py.File(fname, "r")) for fname in qmap_files]

        map_names0 = tuple(handles[0]["/qmap/map_names"][()])
        for fname, hf in zip(qmap_files[1:], handles[1:]):
            map_names_i = tuple(hf["/qmap/map_names"][()])
            assert map_names_i == map_names0, (
                f"map_names must be the same across all files: "
                f"{map_names0!r} != {map_names_i!r} ({fname!r})"
            )
        logger.info("map_names validated: %s", map_names0)

        logger.info("Copying %s -> output file as base ...", qmap_files[0])
        shutil.copy(qmap_files[0], output_file)

        with h5py.File(output_file, "r+") as fo:
            # Combine masks
            masks = [hf["/qmap/mask"][()] for hf in handles]
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = np.logical_or(combined_mask, mask)
            logger.info(
                "Mask: valid per file=%s  combined valid=%d",
                [int(m.sum()) for m in masks],
                combined_mask.sum(),
            )
            del fo["/qmap/mask"]
            fo["/qmap/mask"] = combined_mask

            for prefix in ["static", "dynamic"]:
                logger.info("--- Processing '%s' partition ---", prefix)

                num_pts_list = [hf[f"/qmap/{prefix}_num_pts"][()] for hf in handles]
                logger.debug(
                    "  num_pts per file: %s", [n.tolist() for n in num_pts_list]
                )

                dim0_num_pts = int(sum(n[0] for n in num_pts_list))
                dim1_num_pts = int(max(n[1] for n in num_pts_list))
                logger.info(
                    "  Combined num_pts: dim0=%d (sum of %s)  dim1=%d (max of %s)",
                    dim0_num_pts,
                    [int(n[0]) for n in num_pts_list],
                    dim1_num_pts,
                    [int(n[1]) for n in num_pts_list],
                )
                del fo[f"/qmap/{prefix}_num_pts"]
                fo[f"/qmap/{prefix}_num_pts"] = np.array([dim0_num_pts, dim1_num_pts])

                # Combine dim0 value list (concatenate all ranges, in file order)
                v_list_dim0 = np.concatenate(
                    [hf[f"/qmap/{prefix}_v_list_dim0"][()] for hf in handles]
                )
                logger.debug(
                    "  v_list_dim0: range [%.6g, %.6g], %d entries",
                    v_list_dim0.min(),
                    v_list_dim0.max(),
                    len(v_list_dim0),
                )
                del fo[f"/qmap/{prefix}_v_list_dim0"]
                fo[f"/qmap/{prefix}_v_list_dim0"] = v_list_dim0

                # Keep the dim1 value list from whichever file has the most dim1
                # bins (ties keep the earliest file, already the output's base).
                winner = max(range(len(handles)), key=lambda i: num_pts_list[i][1])
                if winner != 0:
                    logger.debug(
                        "  v_list_dim1: using file[%d]'s list (%d entries)",
                        winner,
                        num_pts_list[winner][1],
                    )
                    del fo[f"/qmap/{prefix}_v_list_dim1"]
                    fo[f"/qmap/{prefix}_v_list_dim1"] = handles[winner][
                        f"/qmap/{prefix}_v_list_dim1"
                    ][()]

                # Merge roi maps: fold files in one at a time, offsetting each
                # newcomer's non-zero indices past the running total so they
                # don't collide, then adding it in (files are assumed to cover
                # disjoint pixels, same assumption the 2-file version made).
                roi_map = handles[0][f"/qmap/{prefix}_roi_map"][()].astype(np.int64)
                for hf in handles[1:]:
                    next_roi = hf[f"/qmap/{prefix}_roi_map"][()].astype(np.int64)
                    running_max = np.max(roi_map[roi_map > 0], initial=0)
                    next_roi = next_roi.copy()
                    next_roi[next_roi > 0] += running_max
                    roi_map = roi_map + next_roi

                start_index = np.min(roi_map)

                # Re-index to natural order (0 = masked, 1-based = valid)
                unique_idx, inverse = np.unique(roi_map, return_inverse=True)
                partition_natural_order = inverse.reshape(roi_map.shape).astype(
                    np.uint32
                )

                # If no masked pixels exist, shift up so index 0 stays reserved
                if start_index > 0:
                    partition_natural_order += 1

                num_partitions = int((unique_idx > 0).sum())
                logger.info("  Combined roi_map: %d valid partitions", num_partitions)

                del fo[f"/qmap/{prefix}_roi_map"]
                fo[f"/qmap/{prefix}_roi_map"] = partition_natural_order

                del fo[f"/qmap/{prefix}_index_mapping"]
                fo[f"/qmap/{prefix}_index_mapping"] = unique_idx[unique_idx > 0] - 1

    logger.info("Done. Output written to: %s", output_file)


def least_multiple(a: int, b: int) -> int:
    """Smallest multiple of ``a`` that is >= ``b`` (aligns static bins to dynamic)."""
    return ((b + a - 1) // a) * a
