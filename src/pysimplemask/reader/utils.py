import numpy as np
import h5py
import hdf5plugin
import logging


logger = logging.getLogger(__name__)


from multiprocessing import Pool, cpu_count


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
            raise ValueError(f"start_frame must be between 0 and {total_frames-1}")

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
            return np.sum(dataset[start_frame:start_frame + num_frames], axis=0)

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

