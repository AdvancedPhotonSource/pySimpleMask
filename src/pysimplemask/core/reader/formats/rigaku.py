"""Rigaku 64-bit sparse binary loaders (500k single module and 3M six-module)."""

import logging
import os
import struct

import numpy as np
from scipy.sparse import coo_array

from ..io_utils import resolve_frame_range
from .base import ScatteringDataset

logger = logging.getLogger(__name__)

# Single Rigaku 500k module geometry.
MODULE_SHAPE = (512, 1024)
# A module index is packed into 21 bits; 3M module files offset by one mega-pixel.
_MODULE_INDEX_OFFSET = 1024 * 1024


def convert_sparse(raw):
    """Unpack a Rigaku 64-bit word stream into (index, frame, count) arrays.

    Bit layout per 64-bit word: ``[frame:24][index:21]...[count:12]``.
    """
    index = ((raw >> 16) & (2**21 - 1)).astype(np.uint32)
    frame = (raw >> 40).astype(np.uint32)
    # Count occupies 12 bits; keep the full range (the original code truncated
    # to 8 bits, dropping per-pixel-per-frame counts above 255).
    count = (raw & (2**12 - 1)).astype(np.uint32)
    return index, frame, count


def get_number_of_frames_from_binfile(filepath, endianness="<"):
    """Read the number of frames from the last 8-byte word of a Rigaku binary.

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


class RigakuDataset(ScatteringDataset):
    """Loader for a single Rigaku 500k 64-bit binary file."""

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

        # Rigaku 3M module files offset the index by one mega-pixel; fold back.
        if index.size and np.min(index) >= _MODULE_INDEX_OFFSET:
            index = index - _MODULE_INDEX_OFFSET

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


class Rigaku3MDataset(ScatteringDataset):
    """Loader for a Rigaku 3M dataset: six 500k modules in a 3x2 layout."""

    # Physical module ordering of the six ``*.bin.00N`` files.
    _MODULE_ORDER = (5, 0, 4, 1, 3, 2)

    def __init__(self, fname, gap=(70, 52), layout=(3, 2), **kwargs):
        super().__init__(fname)
        self.gap = gap
        self.layout = layout
        self.modules = self._load_modules()

        rows = MODULE_SHAPE[0] * layout[0] + gap[0] * (layout[0] - 1)
        cols = MODULE_SHAPE[1] * layout[1] + gap[1] * (layout[1] - 1)
        self.det_size = (rows, cols)

    def _load_modules(self):
        flist = [self.fname[:-3] + f"00{n}" for n in self._MODULE_ORDER]
        missing = [f for f in flist if not os.path.isfile(f)]
        if missing:
            raise FileNotFoundError(f"missing Rigaku 3M module files: {missing}")

        total_frames = max(get_number_of_frames_from_binfile(f) for f in flist)
        for f in flist:
            logger.info("Rigaku 3M module %s", f)
        return [
            RigakuDataset(f, det_size=MODULE_SHAPE, total_frames=total_frames)
            for f in flist
        ]

    def _stitch(self, module_images):
        rows, cols = self.layout
        canvas = np.zeros(self.det_size, dtype=module_images[0].dtype)
        for row in range(rows):
            top = row * (MODULE_SHAPE[0] + self.gap[0])
            v = slice(top, top + MODULE_SHAPE[0])
            for col in range(cols):
                left = col * (MODULE_SHAPE[1] + self.gap[1])
                h = slice(left, left + MODULE_SHAPE[1])
                canvas[v, h] = module_images[row * cols + col]
        return canvas

    def get_scattering(self, num_frames=-1, begin_idx=0, num_processes=None):
        module_images = [
            m.get_scattering(num_frames=num_frames, begin_idx=begin_idx)
            for m in self.modules
        ]
        return self._stitch(module_images)
