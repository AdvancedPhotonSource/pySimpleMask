"""Raw-frame IO helpers shared across format loaders."""

import logging
from multiprocessing import Pool, cpu_count

import h5py
import hdf5plugin  # noqa: F401  # registers HDF5 compression plugins (bitshuffle/lz4)
import numpy as np

logger = logging.getLogger(__name__)


def process_chunk(args):
    """Sum a contiguous chunk of frames from an HDF5 dataset.

    Args:
        args: ``(file_path, dataset_name, start_idx, end_idx)``.

    Returns:
        np.ndarray: per-pixel sum over ``[start_idx, end_idx)``.
    """
    file_path, dataset_name, start_idx, end_idx = args
    with h5py.File(file_path, "r") as f:
        chunk = f[dataset_name][start_idx:end_idx].astype(np.float64)
        return np.sum(chunk, axis=0)


def _resolve_frame_range(total_frames, start_frame, num_frames):
    """Clamp a requested frame range to what the dataset actually contains.

    Returns the number of frames to read, starting at ``start_frame``.
    """
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"start_frame must be between 0 and {total_frames - 1}")

    if num_frames is None or num_frames == 0:
        num_frames = total_frames - start_frame
    elif num_frames < 0:
        # negative sentinel: read a representative subset
        num_frames = max(1000, total_frames // 5)

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if (start_frame + num_frames) > total_frames:
        num_frames = total_frames - start_frame
    return num_frames


def average_frames_parallel(
    file_path,
    dataset_name="/entry/data/data",
    start_frame=0,
    num_frames=-1,
    chunk_size=32,
    num_processes=None,
):
    """Return the per-pixel mean image over a range of frames in an HDF5 dataset.

    Chunks are summed in parallel and the total is divided by the number of
    frames so that the result is independent of the frame count.

    Parameters:
        file_path: Path to the HDF5 file.
        dataset_name: Path to the dataset inside the file.
        start_frame: First frame index to include (default 0).
        num_frames: Number of frames to average. ``0``/``None`` means all
            remaining frames; a negative value selects a representative subset.
        chunk_size: Frames per parallel chunk.
        num_processes: Worker count. ``None`` uses half of the available cores.

    Returns:
        np.ndarray: 2-D ``float32`` mean image.
    """
    with h5py.File(file_path, "r") as f:
        dataset = f[dataset_name]
        if dataset.ndim not in (2, 3):
            raise ValueError("Dataset must be 2D or 3D")
        if dataset.ndim == 2:
            return dataset[()].astype(np.float32)

        total_frames = dataset.shape[0]
        logger.info("Total frames in dataset: %d", total_frames)

        num_frames = _resolve_frame_range(total_frames, start_frame, num_frames)

        # Small ranges are cheaper to read in a single process.
        if num_frames < chunk_size:
            frames = dataset[start_frame : start_frame + num_frames].astype(np.float64)
            mean = np.sum(frames, axis=0) / num_frames
            return mean.astype(np.float32)

        chunks = []
        for i in range(start_frame, start_frame + num_frames, chunk_size):
            end_idx = min(i + chunk_size, start_frame + num_frames)
            chunks.append((file_path, dataset_name, i, end_idx))

    if num_processes is None:
        num_processes = max(1, cpu_count() // 2)
    num_processes = min(len(chunks), num_processes)
    logger.info("using %d cores to load %d frames", num_processes, num_frames)

    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)

    total = np.sum(np.array(results), axis=0)
    return (total / num_frames).astype(np.float32)
