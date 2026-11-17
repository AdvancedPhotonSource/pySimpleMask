# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Rigaku 64-bit sparse binary loaders (500k single module and 3M six-module).

A Rigaku file is a list of events, one 64-bit word for each pixel of each
frame that counted photons, written frame by frame. The loaders do not hold it
in memory: the events of a range of frames are one stretch of the file, which
is found by bisection and summed as it is read.
"""

import logging
import os
import struct
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..io_utils import default_num_workers, gc_paused, read_into, resolve_frame_range
from .base import ScatteringDataset

logger = logging.getLogger(__name__)

# Single Rigaku 500k module geometry.
MODULE_SHAPE = (512, 1024)
# A module index is packed into 21 bits; 3M module files offset by one mega-pixel.
_MODULE_INDEX_OFFSET = 1024 * 1024
# A single frame holds too few photons to show the scattering pattern, so
# browsing the frames averages this many of them into one image by default.
DEFAULT_FRAME_AVERAGE = 100

# Bit layout of an event word: [frame:24][index:21]...[count:12]
_WORD = struct.Struct("<Q")
_FRAME_SHIFT = 40
_INDEX_SHIFT = 16
_INDEX_MASK = 2**21 - 1
_COUNT_MASK = 2**12 - 1

# Events that a thread reads and sums at a time (8 MiB of the file). Smaller
# chunks stay in cache but make the numpy calls short, and many threads then
# queue for the GIL between them; larger ones only cost memory bandwidth.
_CHUNK_EVENTS = 1024 * 1024


def convert_sparse(raw):
    """Unpack a Rigaku 64-bit word stream into (index, frame, count) arrays.

    Bit layout per 64-bit word: ``[frame:24][index:21]...[count:12]``.
    """
    index = ((raw >> _INDEX_SHIFT) & _INDEX_MASK).astype(np.uint32)
    frame = (raw >> _FRAME_SHIFT).astype(np.uint32)
    # Count occupies 12 bits; keep the full range (the original code truncated
    # to 8 bits, dropping per-pixel-per-frame counts above 255).
    count = (raw & _COUNT_MASK).astype(np.uint32)
    return index, frame, count


def _read_word(f, n):
    """The ``n``-th 64-bit word of the open file ``f``."""
    f.seek(n * _WORD.size)
    return _WORD.unpack(f.read(_WORD.size))[0]


def _first_event_of_frame(f, num_events, frame):
    """Number of events of the open file ``f`` that precede frame ``frame``,
    found by bisection: the events are in frame order."""
    lo, hi = 0, num_events
    while lo < hi:
        mid = (lo + hi) // 2
        if _read_word(f, mid) >> _FRAME_SHIFT < frame:
            lo = mid + 1
        else:
            hi = mid
    return lo


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

    default_frame_average = DEFAULT_FRAME_AVERAGE

    def __init__(self, fname, det_size=MODULE_SHAPE, total_frames=None, **kwargs):
        super().__init__(fname)
        self.det_size = tuple(det_size)
        self.num_events = os.path.getsize(fname) // _WORD.size
        max_frames = 0
        self._index_offset = 0
        if self.num_events:
            with open(fname, "rb") as f:
                first, last = _read_word(f, 0), _read_word(f, self.num_events - 1)
            max_frames = (last >> _FRAME_SHIFT) + 1
            # Rigaku 3M module files offset the index by one mega-pixel; fold back.
            if (first >> _INDEX_SHIFT) & _INDEX_MASK >= _MODULE_INDEX_OFFSET:
                self._index_offset = _MODULE_INDEX_OFFSET
        self.num_frames_total = max(total_frames or 0, max_frames)
        self.num_browsable_frames = self.num_frames_total

    def _sum_events(self, begin, end):
        """Per-pixel sum of the counts of the events ``[begin, end)``, flat."""
        num_pixels = self.det_size[0] * self.det_size[1]
        total = np.zeros(num_pixels, dtype=np.float64)
        if begin == end:  # frames without events
            return total
        size = min(_CHUNK_EVENTS, end - begin)
        words = np.empty(size, dtype=_WORD.format)
        index = np.empty(size, dtype=np.uint64)
        # one file object per thread: seek + readinto is then safe, and portable
        with open(self.fname, "rb", buffering=0) as f:
            for lo in range(begin, end, size):
                n = min(size, end - lo)
                word, idx = words[:n], index[:n]
                read_into(f, lo * _WORD.size, word.view(np.uint8))
                np.right_shift(word, _INDEX_SHIFT, out=idx)
                np.bitwise_and(idx, _INDEX_MASK, out=idx)
                if self._index_offset:
                    # an index below the offset wraps, and is negative as int64
                    np.subtract(idx, self._index_offset, out=idx)
                np.bitwise_and(word, _COUNT_MASK, out=word)
                counts = np.bincount(
                    idx.view(np.int64), weights=word, minlength=num_pixels
                )
                if counts.size != num_pixels:
                    raise ValueError(
                        f"{self.fname} has events outside the {self.det_size} detector"
                    )
                total += counts
        return total

    def _sum_frames(self, begin_idx, end_idx, num_workers=None):
        """Per-pixel sum of the counts in the frames ``[begin_idx, end_idx)``,
        flat; a long range is read by ``num_workers`` threads."""
        with open(self.fname, "rb") as f:
            begin = _first_event_of_frame(f, self.num_events, begin_idx)
            end = _first_event_of_frame(f, self.num_events, end_idx)
        if num_workers is None:
            num_workers = default_num_workers()
        num_workers = min(num_workers, -(-(end - begin) // _CHUNK_EVENTS))
        if num_workers <= 1:
            return self._sum_events(begin, end)
        edges = [int(n) for n in np.linspace(begin, end, num_workers + 1)]
        with gc_paused(), ThreadPoolExecutor(max_workers=num_workers) as pool:
            return sum(pool.map(self._sum_events, edges[:-1], edges[1:]))

    def get_scattering(self, num_frames=-1, begin_idx=0, num_processes=None):
        n_frames = resolve_frame_range(self.num_frames_total, begin_idx, num_frames)
        summed = self._sum_frames(begin_idx, begin_idx + n_frames, num_processes)
        return (summed / n_frames).reshape(self.det_size).astype(np.float32)

    def get_frames_mean(self, begin_idx, end_idx):
        self._check_frame_range(begin_idx, end_idx)
        summed = self._sum_frames(begin_idx, end_idx)
        return (summed / (end_idx - begin_idx)).reshape(self.det_size).astype(np.float32)


class Rigaku3MDataset(ScatteringDataset):
    """Loader for a Rigaku 3M dataset: six 500k modules in a 3x2 layout."""

    default_frame_average = DEFAULT_FRAME_AVERAGE

    # Physical module ordering of the six ``*.bin.00N`` files.
    _MODULE_ORDER = (5, 0, 4, 1, 3, 2)

    def __init__(self, fname, gap=(70, 52), layout=(3, 2), **kwargs):
        super().__init__(fname)
        self.gap = gap
        self.layout = layout
        self.modules = self._load_modules()
        # every module is given the frame count of the longest module file
        self.num_browsable_frames = self.modules[0].num_browsable_frames

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
            m.get_scattering(
                num_frames=num_frames, begin_idx=begin_idx, num_processes=num_processes
            )
            for m in self.modules
        ]
        return self._stitch(module_images)

    def get_frames_mean(self, begin_idx, end_idx):
        return self._stitch(
            [m.get_frames_mean(begin_idx, end_idx) for m in self.modules]
        )
