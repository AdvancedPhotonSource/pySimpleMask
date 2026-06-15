"""HDF5 scattering loader (APS NeXus-style ``/entry/data/data``)."""

import logging

import h5py
import hdf5plugin  # noqa: F401  # registers HDF5 compression plugins

from ..io_utils import average_frames_parallel
from .base import ScatteringDataset

logger = logging.getLogger(__name__)


class HdfDataset(ScatteringDataset):
    """Loader for 3-D (frame, y, x) detector stacks stored in HDF5."""

    def __init__(self, fname, data_path="/entry/data/data", **kwargs):
        super().__init__(fname)
        self.data_path = data_path
        with h5py.File(self.fname, "r") as f:
            if data_path not in f:
                raise KeyError(f"dataset {data_path!r} not found in {fname}")
            shape = f[data_path].shape
        if len(shape) == 2:
            self.det_size = tuple(shape)
        elif len(shape) == 3:
            self.det_size = tuple(shape[1:])
        else:
            raise ValueError(f"unexpected dataset rank {len(shape)} for {data_path!r}")

    def get_scattering(self, num_frames=-1, begin_idx=0, num_processes=None):
        return average_frames_parallel(
            self.fname,
            dataset_name=self.data_path,
            start_frame=begin_idx,
            num_frames=num_frames,
            chunk_size=32,
            num_processes=num_processes,
        )
