"""IMM (legacy APS) scattering loader, supporting dense and sparse frames."""

import logging
import struct

import numpy as np

from ..io_utils import resolve_frame_range
from .base import ScatteringDataset

logger = logging.getLogger(__name__)

_IMM_HEADER_SIZE = 1024
_IMM_HEADER_FORMAT = (
    "ii32s16si16siiiiiiiiiiiiiddiiIiiI40sf40sf40sf40s"
    "f40sf40sf40sf40sf40sf40sfffiiifc295s84s12s"
)
_IMM_FIELDNAMES = (
    "mode",
    "compression",
    "date",
    "prefix",
    "number",
    "suffix",
    "monitor",
    "shutter",
    "row_beg",
    "row_end",
    "col_beg",
    "col_end",
    "row_bin",
    "col_bin",
    "rows",
    "cols",
    "bytes",
    "kinetics",
    "kinwinsize",
    "elapsed",
    "preset",
    "topup",
    "inject",
    "dlen",
    "roi_number",
    "buffer_number",
    "systick",
    "pv1",
    "pv1VAL",
    "pv2",
    "pv2VAL",
    "pv3",
    "pv3VAL",
    "pv4",
    "pv4VAL",
    "pv5",
    "pv5VAL",
    "pv6",
    "pv6VAL",
    "pv7",
    "pv7VAL",
    "pv8",
    "pv8VAL",
    "pv9",
    "pv9VAL",
    "pv10",
    "pv10VAL",
    "imageserver",
    "CPUspeed",
    "immversion",
    "corecotick",
    "cameratype",
    "threshhold",
    "byte632",
    "empty_space",
    "ZZZZ",
    "FFFF",
)
# Sparse frames store an int32 index + int16 count per event (6 bytes);
# dense frames store one uint16 per pixel (2 bytes).
_SPARSE_COMPRESSION = 6
_SPARSE_EVENT_BYTES = 6
_DENSE_PIXEL_BYTES = 2


def _unpack_imm_header(buf):
    """Parse a 1024-byte IMM header block into a field dictionary."""
    values = struct.unpack(_IMM_HEADER_FORMAT, buf)
    return dict(zip(_IMM_FIELDNAMES, values))


def read_imm_header(file):
    """Read and parse one IMM header from an open binary file object."""
    return _unpack_imm_header(file.read(_IMM_HEADER_SIZE))


class ImmDataset(ScatteringDataset):
    """Loader for ``.imm`` files (dense or compression-6 sparse)."""

    def __init__(self, fname, **kwargs):
        super().__init__(fname)
        self.toc, self.det_size, self.is_sparse = self._read_toc()

    def _read_toc(self):
        """Build a table of contents: (payload_start, event_count) per frame."""
        toc = []
        with open(self.fname, "rb") as f:
            header = read_imm_header(f)
            det_size = (header["rows"], header["cols"])
            is_sparse = header["compression"] == _SPARSE_COMPRESSION
            event_bytes = _SPARSE_EVENT_BYTES if is_sparse else _DENSE_PIXEL_BYTES

            f.seek(0)
            while True:
                buf = f.read(_IMM_HEADER_SIZE)
                if len(buf) == 0:
                    break
                if len(buf) < _IMM_HEADER_SIZE:
                    raise IOError("IMM file is corrupted (truncated header)")
                header = _unpack_imm_header(buf)
                payload_start = f.tell()
                dlen = header["dlen"]
                toc.append((payload_start, dlen))
                f.seek(payload_start + dlen * event_bytes)
                if not f.peek(4):
                    break

        if not toc:
            raise IOError(f"no frames found in IMM file {self.fname}")
        return np.array(toc, dtype=np.int64), det_size, is_sparse

    def get_scattering(self, num_frames=-1, begin_idx=0, num_processes=None):
        n_frames = resolve_frame_range(len(self.toc), begin_idx, num_frames)
        end_idx = begin_idx + n_frames

        pixel_num = self.det_size[0] * self.det_size[1]
        accum = np.zeros(pixel_num, dtype=np.float64)

        with open(self.fname, "rb") as f:
            for i in range(begin_idx, end_idx):
                start_byte, dlen = (int(v) for v in self.toc[i])
                f.seek(start_byte)
                if self.is_sparse:
                    index = np.fromfile(f, dtype=np.int32, count=dlen)
                    count = np.fromfile(f, dtype=np.int16, count=dlen)
                    accum[index] += count
                else:
                    accum += np.fromfile(f, dtype=np.uint16, count=dlen)

        return (accum / n_frames).reshape(self.det_size).astype(np.float32)
