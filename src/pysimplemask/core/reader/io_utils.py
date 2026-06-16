"""Raw-frame IO helpers shared across format loaders."""

import logging
from multiprocessing import Pool, cpu_count

import h5py
import hdf5plugin  # noqa: F401  # registers HDF5 compression plugins (bitshuffle/lz4)
import numpy as np

logger = logging.getLogger(__name__)

# Target in-memory size of a single parallel read chunk.
_CHUNK_MEMORY_BUDGET = 256 * 1024 * 1024  # 256 MiB


def process_chunk(file_path, dataset_name, start_idx, end_idx):
    """Return the per-pixel float32 sum over ``[start_idx, end_idx)`` of a dataset."""
    with h5py.File(file_path, "r") as f:
        chunk = f[dataset_name][start_idx:end_idx]
        return np.sum(chunk, axis=0, dtype=np.float32)


def resolve_frame_range(total_frames, start_frame, num_frames):
    """Clamp a requested frame range to what the dataset contains.

    ``num_frames`` semantics, shared by every format loader:
      * ``> 0`` — exactly that many frames (clamped to the end);
      * ``0`` / ``None`` — all remaining frames from ``start_frame``;
      * ``< 0`` — a representative subset (``max(1000, total_frames // 5)``).

    Returns the number of frames to read, starting at ``start_frame``.
    """
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"start_frame must be between 0 and {total_frames - 1}")
    if num_frames is None or num_frames == 0:
        num_frames = total_frames - start_frame
    elif num_frames < 0:
        num_frames = max(1000, total_frames // 5)
    if start_frame + num_frames > total_frames:
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
    """Return the per-pixel mean image over a range of frames in a 3-D HDF5 stack.

    Chunks are summed in parallel and the total is divided by the number of
    frames, so the result is independent of the frame count. See
    :func:`resolve_frame_range` for ``num_frames`` semantics.

    Returns:
        np.ndarray: 2-D ``float32`` mean image.
    """
    with h5py.File(file_path, "r") as f:
        dataset = f[dataset_name]
        if dataset.ndim != 3:
            raise ValueError("expected a 3-D (frame, y, x) dataset")

        total_frames = dataset.shape[0]
        logger.info("Total frames in dataset: %d", total_frames)
        num_frames = resolve_frame_range(total_frames, start_frame, num_frames)

        # Small ranges are cheaper to read in a single process.
        if num_frames < chunk_size:
            frames = dataset[start_frame : start_frame + num_frames]
            return (np.sum(frames, axis=0, dtype=np.float32) / num_frames).astype(
                np.float32
            )

        # Grow chunks toward one-per-worker to amortize the per-chunk file open,
        # but cap each chunk's in-memory size so wide detectors don't blow up RAM.
        if num_processes is None:
            num_processes = max(1, cpu_count() // 2)
        frame_nbytes = int(np.prod(dataset.shape[1:])) * dataset.dtype.itemsize
        budget_frames = max(1, _CHUNK_MEMORY_BUDGET // max(frame_nbytes, 1))
        per_worker = -(-num_frames // num_processes)  # ceil division
        chunk_size = max(chunk_size, min(per_worker, budget_frames))

        stop = start_frame + num_frames
        chunks = [
            (file_path, dataset_name, i, min(i + chunk_size, stop))
            for i in range(start_frame, stop, chunk_size)
        ]

    num_processes = min(len(chunks), num_processes)
    logger.info("using %d cores to load %d frames", num_processes, num_frames)
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_chunk, chunks)

    # results are per-chunk (H, W) float32 sums; fold without stacking into an
    # (n_chunks, H, W) intermediate.
    return (sum(results) / num_frames).astype(np.float32)
