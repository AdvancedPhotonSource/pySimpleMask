import numpy as np

# required for bloc compression
import hdf5plugin
import h5py
import logging
from .xpcs_dataset import XpcsDataset


logger = logging.getLogger(__name__)


from multiprocessing import Pool, cpu_count

def process_chunk(args):
    """
    Process a chunk of frames.
    
    Args:
        args: tuple containing (file_path, dataset_name, start_idx, end_idx)
    """
    file_path, dataset_name, start_idx, end_idx = args
    with h5py.File(file_path, 'r') as f:
        chunk = f[dataset_name][start_idx:end_idx].astype(np.float32)
        return np.sum(chunk, axis=0)


def sum_frames_parallel(file_path, dataset_name='/entry/data/data',
                        start_frame=0, num_frames=-1, 
                        chunk_size=32, num_processes=None):
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
    with h5py.File(file_path, 'r') as f:
        dataset = f[dataset_name]
        total_frames = dataset.shape[0]
        
        # Validate inputs
        if start_frame < 0 or start_frame >= total_frames:
            raise ValueError(f"start_frame must be between 0 and {total_frames-1}")
        
        # If num_frames is None, use all remaining frames
        if num_frames is None or num_frames <= 0:
            num_frames = total_frames - start_frame
        
        # Validate num_frames
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if (start_frame + num_frames) > total_frames:
            num_frames = total_frames - start_frame
        
        # Create chunks
        chunks = []
        for i in range(start_frame, start_frame + num_frames, chunk_size):
            end_idx = min(i + chunk_size, start_frame + num_frames)
            chunks.append((file_path, dataset_name, i, end_idx))
    
    # Set number of processes
    if num_processes is None:
        num_processes = cpu_count() // 2

    num_processes = min(len(chunks), num_processes) 
    logger.info(f'using {num_processes} cores to load {num_frames} frames')
    # Process chunks in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_chunk, chunks)
    
    # Sum all results
    return np.sum(np.array(results), axis=0)


class HdfDataset(XpcsDataset):
    """
    Parameters
    ----------
    filename: string
    """
    def __init__(self,
                 *args,
                 preload_size=8,
                 dtype=np.uint8,
                 data_path='/entry/data/data',
                 **kwargs):

        super(HdfDataset, self).__init__(*args, dtype=dtype, **kwargs)
        self.dataset_type = "HDF5 Dataset"
        self.is_sparse = False
        with h5py.File(self.fname, 'r') as f:
            data = f[data_path]
            self.shape = data.shape

            # update data type;
            if data.dtype == np.uint8:
                self.dtype = np.int16
            elif data.dtype == np.uint16:
                # lambda2m's uint16 is actually 12bit. it's safe to to int16
                if self.shape[1] * self.shape[2] == 1813 * 1558:
                    self.dtype = np.int16
                # likely eiger detectors
                else:
                    self.dtype = np.int32
            elif data.dtype == np.uint32:
                self.dtype = np.int32
                logger.warn('cast uint32 to int32. it may cause ' + 
                            'overflow when the maximal value >= 2^31')
            else:
                self.dtype = data.dtype
        
        if 'avg_frame' in kwargs and kwargs['avg_frame'] > 1:
            self.dtype = np.float32
        self.fhdl = None
        self.data = None
        self.data_cache = None
        self.data_path = data_path

        self.update_batch_info(self.shape[0])
        self.update_det_size(self.shape[1:])

        if self.mask_crop is not None:
            self.mask_crop = self.mask_crop.cpu().numpy()

        self.current_group = None
        self.preload_size = preload_size

    def __reset__(self):
        if self.fhdl is not None:
            self.fhdl.close()
            self.data = None
            self.fhdl = None

    def __getbatch__(self, idx):
        """
        return numpy array, which will be converted to tensor with dataloader
        """
        if self.fhdl is None:
            self.fhdl = h5py.File(self.fname, 'r', rdcc_nbytes=1024*1024*256)
            self.data = self.fhdl[self.data_path]

        beg, end, size = self.get_raw_index(idx)
        idx_list = np.arange(beg, end, self.stride)

        if self.mask_crop is not None:
            # x = self.data[beg:end, self.sl_v, self.sl_h].reshape(end - beg, -1)
            x = self.data[idx_list].reshape(size, -1)
            x = x[:, self.mask_crop].astype(self.dtype)
            # if idx == 0:
            #     print(np.max(x), x.dtype)
        else:
            x = self.data[idx_list].reshape(-1, self.pixel_num)

        if x.dtype != self.dtype:
            x = x.astype(self.dtype)
        return x

    def get_scattering(self, num_frames=-1, begin_idx=0, num_processes=None):
        return sum_frames_parallel(self.fname,
                                   dataset_name='/entry/data/data',
                                   start_frame=begin_idx,
                                   num_frames=num_frames, 
                                   chunk_size=32,
                                   num_processes=num_processes)


def test():
    fname = (
        "/clhome/MQICHU/ssd/xpcs_data_raw/A003_Cu3Au_att0_001/A003_Cu3Au_att0_001.imm"
    )
    ds = HdfDataset(fname)
    # for n in range(len(ds)):
    #     print(n, ds[n].shape, ds[n].device)


def test_bin(idx):
    fname = f"/clhome/MQICHU/ssd/APSU_TestData_202106/APSU_TestData_{idx:03d}/APSU_TestData_{idx:03d}.h5"
    ds = HdfDataset(fname)
    ds.to_rigaku_bin(f"hdf2bin_{idx:03d}.bin")


if __name__ == '__main__':
    test_bin(4)
