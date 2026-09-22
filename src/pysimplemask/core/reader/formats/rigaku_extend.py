# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""RigakuExtend 64-bit sparse binary loader (single 512x448 module, ``.bix``)."""

import logging
import os
import struct

import numpy as np
from scipy.sparse import coo_array

from ..io_utils import resolve_frame_range
from .base import ScatteringDataset

logger = logging.getLogger(__name__)

# RigakuExtend module geometry (height, width).
MODULE_SHAPE = (512, 448)


def convert_sparse(raw):
    """Unpack a RigakuExtend 64-bit word stream into (index, frame, count) arrays.

    Bit layout per 64-bit word: ``[frame:32][index:20][count:12]``.
    """
    index = ((raw >> 12) & (2**20 - 1)).astype(np.uint32)
    frame = (raw >> 32).astype(np.uint32)
    count = (raw & (2**12 - 1)).astype(np.uint32)
    return index, frame, count


def get_number_of_frames_from_binfile(filepath, endianness="<"):
    """Read the number of frames from the last 8-byte word of a RigakuExtend binary.

    Args:
        filepath: Path to the binary file.
        endianness: ``'<'`` little-endian (default) or ``'>'`` big-endian.

    Returns:
        int: Number of frames (last frame index + 1).
    """
    file_size = os.path.getsize(filepath)
    if file_size < 8:
        raise ValueError("file is smaller than one 8-byte word")

    with open(filepath, "rb") as f:
        f.seek(file_size - 8)
        last_word = struct.unpack(endianness + "Q", f.read(8))[0]

    _index, frame, _count = convert_sparse(np.array([last_word], dtype=np.uint64))
    return int(frame[0]) + 1


class RigakuExtendDataset(ScatteringDataset):
    """Loader for a single RigakuExtend 512x448 64-bit binary file."""

    def __init__(self, fname, det_size=MODULE_SHAPE, total_frames=None, **kwargs):
        super().__init__(fname)
        self.det_size = tuple(det_size)
        self.index, self.frame, self.count, self.num_frames_total = self._read(
            total_frames
        )

    def _read(self, total_frames):
        with open(self.fname, "rb") as f:
            raw = np.fromfile(f, dtype=np.uint64)
        index, frame, count = convert_sparse(raw)

        max_frames = int(frame[-1]) + 1 if frame.size else 0
        if total_frames is None:
            total_frames = max_frames
        else:
            total_frames = max(total_frames, max_frames)

        return index, frame, count, total_frames

    def get_scattering(self, num_frames=-1, begin_idx=0, num_processes=None):
        total_frames = self.num_frames_total
        n_frames = resolve_frame_range(total_frames, begin_idx, num_frames)
        end_idx = begin_idx + n_frames

        pixel_num = self.det_size[0] * self.det_size[1]
        smat = coo_array(
            (self.count.astype(np.float64), (self.frame, self.index)),
            shape=(total_frames, pixel_num),
        ).tocsr()

        summed = np.asarray(smat[begin_idx:end_idx].sum(axis=0)).reshape(self.det_size)
        return (summed / n_frames).astype(np.float32)
