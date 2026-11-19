# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Raw-frame IO helpers shared across format loaders."""

import contextlib
import ctypes
import gc
import logging
import pathlib
import struct
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from multiprocessing import Pool, cpu_count
from typing import Callable, NamedTuple, Optional

import h5py
import hdf5plugin  # noqa: F401  # registers HDF5 compression plugins (bitshuffle/lz4)
import numpy as np

logger = logging.getLogger(__name__)


def _cast_to_signed(arr: np.ndarray) -> np.ndarray:
    """Cast an unsigned integer array to its signed counterpart.

    uint16 → int16, uint32 → int32, etc.  Signed and non-integer arrays are
    returned unchanged.  The cast preserves bit patterns so that detector
    overflow values (e.g. 65535 for uint16) become negative rather than
    staying as large positive counts.
    """
    if arr.dtype.kind == "u":
        return arr.astype(np.dtype(f"int{arr.dtype.itemsize * 8}"))
    return arr


_CHUNK_MEMORY_BUDGET = 256 * 1024 * 1024  # 256 MiB

# ---------------------------------------------------------------------------
# Direct chunk read — frame-per-chunk datasets decoded without the HDF5 library.
#
# h5py serialises every read behind a global lock, so reading through it only
# scales with worker *processes*, which pay to start up, to decode into large
# intermediate stacks and to pickle their sums back.  For the layouts the
# detectors write (one frame per chunk; bitshuffle, LZ4 or Blosc) the chunk
# index is read once through h5py instead, and worker threads then read and
# decompress the chunks themselves.  File reads, the decompressors (called
# through ctypes) and numpy all release the GIL, so the threads run in
# parallel on a regular interpreter as well as on a free-threaded one.
# ---------------------------------------------------------------------------
_BSHUF_FILTER_ID = 32008
_LZ4_FILTER_ID = 32004
_BLOSC_FILTER_ID = 32001
# cd_values[4] of the bitshuffle filter -> decompress function of its library
_BSHUF_DECODERS = {2: "bshuf_decompress_lz4", 3: "bshuf_decompress_zstd"}
# The bitshuffle and LZ4 filters start a chunk with (big-endian) the
# uncompressed size in bytes (uint64) and the block size in bytes (uint32);
# the LZ4 filter then prefixes every block with its compressed size (uint32).
_CHUNK_HEADER = struct.Struct(">QI")
_LZ4_BLOCK_HEADER = struct.Struct(">I")
# A Blosc chunk is one Blosc buffer; its header holds (little-endian, after
# four bytes of versions and flags) the uncompressed size, the block size and
# the compressed size (uint32 each).
_BLOSC_HEADER = struct.Struct("<4xIII")

# Frame dtype (unsigned types already reinterpreted as signed, see
# _cast_to_signed) -> dtype of a worker's running sum.  The threads are limited
# by memory bandwidth, so the sum is kept narrow; a worker adds up at most
# _SUM_WINDOW frames before its sum is folded into the float64 total, which
# keeps the integer sums exact (256 int8 frames fit int16, 256 int16 frames
# fit int32) and the float32 sum of int32 counts exact up to 2**24 per pixel.
_SUM_DTYPE = {"i1": "i2", "i2": "i4", "i4": "f4", "i8": "f8", "f4": "f4", "f8": "f8"}
_SUM_WINDOW = 256
# Default cap on the threads: more only contend for memory bandwidth.
_DIRECT_MAX_WORKERS = 32

_filter_libs: dict = {}


def _filter_func(lib, name, restype, argtypes):
    """Return function ``name`` of the ``h5<lib>`` filter library bundled in hdf5plugin."""
    if lib not in _filter_libs:
        plugin_dir = pathlib.Path(hdf5plugin.__file__).parent / "plugins"
        paths = sorted(plugin_dir.glob(f"*h5{lib}.*"))
        if not paths:
            raise OSError(f"no h5{lib} filter library in {plugin_dir}")
        _filter_libs[lib] = ctypes.CDLL(str(paths[0]))
    func = getattr(_filter_libs[lib], name)
    func.restype, func.argtypes = restype, argtypes
    return func


def _bshuf_decoder(name):
    """Return ``decode(chunk, frame)`` for chunks of the bitshuffle filter."""
    # int64_t f(const void *in, void *out, size_t size, size_t elem_size, size_t block_size)
    func = _filter_func(
        "bshuf", name, ctypes.c_int64, [ctypes.c_void_p] * 2 + [ctypes.c_size_t] * 3
    )

    def decode(chunk, frame):
        nbytes, block_nbytes = _CHUNK_HEADER.unpack_from(chunk)
        if nbytes != frame.nbytes:
            raise ValueError(f"chunk holds {nbytes} bytes, expected {frame.nbytes}")
        ret = func(
            chunk.ctypes.data + _CHUNK_HEADER.size,
            frame.ctypes.data,
            frame.size,
            frame.itemsize,
            block_nbytes // frame.itemsize,
        )
        if ret < 0:
            raise RuntimeError(f"bitshuffle decompression failed with error {ret}")

    return decode


def _lz4_decoder():
    """Return ``decode(chunk, frame)`` for chunks of the LZ4 filter."""
    # int LZ4_decompress_safe(const char *src, char *dst, int compressedSize, int dstCapacity)
    func = _filter_func(
        "lz4", "LZ4_decompress_safe", ctypes.c_int, [ctypes.c_void_p] * 2 + [ctypes.c_int] * 2
    )

    def decode(chunk, frame):
        nbytes, block_nbytes = _CHUNK_HEADER.unpack_from(chunk)
        if nbytes != frame.nbytes or block_nbytes == 0:
            raise ValueError(f"chunk holds {nbytes} bytes, expected {frame.nbytes}")
        src, dst = chunk.ctypes.data, frame.ctypes.data
        pos, done = _CHUNK_HEADER.size, 0
        while done < nbytes:
            block = min(block_nbytes, nbytes - done)
            (size,) = _LZ4_BLOCK_HEADER.unpack_from(chunk, pos)
            pos += _LZ4_BLOCK_HEADER.size
            if pos + size > chunk.size:
                raise ValueError("LZ4 block runs past the end of its chunk")
            if size == block:  # the filter stores an incompressible block as is
                ctypes.memmove(dst + done, src + pos, block)
            elif func(src + pos, dst + done, size, block) != block:
                raise RuntimeError("LZ4 decompression failed")
            pos += size
            done += block

    return decode


def _blosc_decoder():
    """Return ``decode(chunk, frame)`` for chunks of the Blosc filter."""
    # int blosc_decompress_ctx(const void *src, void *dest, size_t destsize,
    #                          int numinternalthreads)
    func = _filter_func(
        "blosc",
        "blosc_decompress_ctx",
        ctypes.c_int,
        [ctypes.c_void_p] * 2 + [ctypes.c_size_t, ctypes.c_int],
    )

    def decode(chunk, frame):
        nbytes, _, compressed_nbytes = _BLOSC_HEADER.unpack_from(chunk)
        if nbytes != frame.nbytes or compressed_nbytes > chunk.size:
            raise ValueError(f"chunk holds {nbytes} bytes, expected {frame.nbytes}")
        if func(chunk.ctypes.data, frame.ctypes.data, frame.nbytes, 1) != frame.nbytes:
            raise RuntimeError("Blosc decompression failed")

    return decode


def _chunk_decoder(filter_id, cd_values, itemsize):
    """Return ``decode(chunk, frame)`` for an HDF5 filter, or ``None`` if the
    chunks it writes cannot be decoded here."""
    try:
        if filter_id == _LZ4_FILTER_ID:
            return _lz4_decoder()
        if filter_id == _BLOSC_FILTER_ID:
            return _blosc_decoder()
        # cd_values: (major, minor, elem_size, block_size, compression)
        if (
            filter_id == _BSHUF_FILTER_ID
            and len(cd_values) >= 5
            and cd_values[2] == itemsize
            and cd_values[4] in _BSHUF_DECODERS
        ):
            return _bshuf_decoder(_BSHUF_DECODERS[cd_values[4]])
    except (AttributeError, OSError) as exc:
        logger.debug("filter library is not usable for a direct chunk read: %s", exc)
    return None


class _DirectLayout(NamedTuple):
    """Where a range of frames lives in its file, and how to decode it."""

    fname: str
    shape: tuple  # (rows, cols) of one frame
    dtype: np.dtype  # frame dtype; unsigned types are reinterpreted as signed
    addr: np.ndarray  # file offset of each frame
    nbytes: np.ndarray  # stored size of each frame
    stored: np.ndarray  # True where a frame was written without the filter
    decode: Optional[Callable]  # chunk decoder; None for a dataset without filter


def _chunk_table(dataset, start_frame, num_frames):
    """Return ``(addr, nbytes, stored)`` of the one-frame chunks in a frame
    range, or ``None`` if one of them was never written."""
    addr = np.full(num_frames, -1, dtype=np.int64)
    nbytes = np.zeros(num_frames, dtype=np.int64)
    stored = np.zeros(num_frames, dtype=bool)

    def record(n, info):
        if 0 <= n < num_frames and info.byte_offset is not None:
            addr[n], nbytes[n] = info.byte_offset, info.size
            # bit 0 of the mask set -> the (only) filter was skipped for this chunk
            stored[n] = info.filter_mask & 1

    if hasattr(dataset.id, "chunk_iter"):
        dataset.id.chunk_iter(lambda info: record(info.chunk_offset[0] - start_frame, info))
    else:  # HDF5 < 1.12.3
        for n in range(num_frames):
            record(n, dataset.id.get_chunk_info_by_coord((start_frame + n, 0, 0)))

    if np.any(addr < 0):
        return None
    return addr, nbytes, stored


def _probe_direct(dataset, start_frame, num_frames):
    """Return the :class:`_DirectLayout` of a frame range of a 3-D h5py
    dataset if it can be read without the HDF5 library, else ``None``.

    Qualifying datasets hold a native fixed-size type and are either
    contiguous, or chunked one frame per chunk with no filter or with the
    bitshuffle (LZ4/ZSTD), the LZ4 or the Blosc filter alone.
    """
    dtype = dataset.dtype
    if dtype.kind == "u":
        dtype = np.dtype(f"int{dtype.itemsize * 8}")
    if not (
        dataset.dtype.isnative
        and dtype.str[1:] in _SUM_DTYPE
        and not dataset.is_virtual
        and dataset.external is None
        and dataset.file.driver == "sec2"
        and dataset.file.userblock_size == 0
    ):
        return None

    # the file that actually holds the data; differs from the opened file when
    # the dataset is reached through an external link
    fname = dataset.file.filename
    shape = tuple(int(n) for n in dataset.shape[1:])
    frame_nbytes = shape[0] * shape[1] * dtype.itemsize

    if dataset.chunks is None:
        offset = dataset.id.get_offset()
        if offset is None:
            return None
        addr = offset + frame_nbytes * np.arange(
            start_frame, start_frame + num_frames, dtype=np.int64
        )
        nbytes = np.full(num_frames, frame_nbytes, dtype=np.int64)
        return _DirectLayout(
            fname, shape, dtype, addr, nbytes, np.ones(num_frames, dtype=bool), None
        )

    if dataset.chunks != (1, *shape):
        return None
    dcpl = dataset.id.get_create_plist()
    decode = None
    if dcpl.get_nfilters() == 1:
        filter_id, _, cd_values, _ = dcpl.get_filter(0)
        decode = _chunk_decoder(filter_id, cd_values, dtype.itemsize)
        if decode is None:
            return None
    elif dcpl.get_nfilters() != 0:
        return None

    table = _chunk_table(dataset, start_frame, num_frames)
    if table is None:
        return None
    return _DirectLayout(fname, shape, dtype, *table, decode)


def default_num_workers():
    """Number of threads that read a file when the caller does not say: half
    the CPUs, capped because more only contend for memory bandwidth."""
    return min(max(1, cpu_count() // 2), _DIRECT_MAX_WORKERS)


@contextlib.contextmanager
def gc_paused():
    """Hold the garbage collector off while worker threads run.

    The collector runs in whichever thread trips it, and would then destroy
    the caller's unreachable objects from a worker thread; the GUI's Qt
    objects do not survive that. It is left disabled if it already was.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def read_into(f, offset, buf):
    """Fill the uint8 array ``buf`` from byte ``offset`` of the open file ``f``."""
    f.seek(offset)
    done = 0
    while done < buf.size:
        n = f.readinto(buf[done:])
        if not n:
            raise OSError(f"unexpected end of file while reading {f.name}")
        done += n


def _sum_frames(layout, frames, out):
    """Sum the frames ``frames`` (indices into ``layout``) into the flat array ``out``."""
    frame = np.empty(out.size, dtype=layout.dtype)
    frame_bytes = frame.view(np.uint8)
    chunk = np.empty(int(layout.nbytes[frames].max()), dtype=np.uint8)
    # one file object per thread: seek + readinto is then safe, and portable
    with open(layout.fname, "rb", buffering=0) as f:
        for n, i in enumerate(frames):
            nbytes, offset = int(layout.nbytes[i]), int(layout.addr[i])
            if layout.decode is None or layout.stored[i]:
                if nbytes != frame_bytes.size:
                    raise ValueError(f"frame {i}: unexpected raw chunk size {nbytes}")
                read_into(f, offset, frame_bytes)
            else:
                read_into(f, offset, chunk[:nbytes])
                layout.decode(chunk[:nbytes], frame)
            if n == 0:
                out[:] = frame
            else:
                np.add(out, frame, out=out, dtype=out.dtype, casting="unsafe")


def _average_direct(layout, num_workers=None):
    """Return the float32 mean image of the frames of a :class:`_DirectLayout`,
    summed by ``num_workers`` threads."""
    num_frames = len(layout.addr)
    if num_workers is None:
        num_workers = default_num_workers()
    num_workers = min(num_workers, num_frames)
    size = layout.shape[0] * layout.shape[1]
    sums = np.empty((num_workers, size), dtype=_SUM_DTYPE[layout.dtype.str[1:]])
    total = np.zeros(size, dtype=np.float64)
    # folding the sums in bands of pixels keeps each band of the total in
    # cache, and lets the threads do it as well
    edges = np.linspace(0, size, num_workers + 1, dtype=np.int64)

    def fold(lo, hi, num_sums):
        total[lo:hi] += sums[:num_sums, lo:hi].sum(axis=0, dtype=np.float64)

    logger.info("using %d threads to load %d frames", num_workers, num_frames)
    with gc_paused(), ThreadPoolExecutor(max_workers=num_workers) as pool:
        for start in range(0, num_frames, num_workers * _SUM_WINDOW):
            stop = min(start + num_workers * _SUM_WINDOW, num_frames)
            jobs = [
                job
                for job in np.array_split(np.arange(start, stop), num_workers)
                if len(job)
            ]
            list(pool.map(_sum_frames, repeat(layout), jobs, sums))
            list(pool.map(fold, edges[:-1], edges[1:], repeat(len(jobs))))
    return (total / num_frames).reshape(layout.shape).astype(np.float32)


# ---------------------------------------------------------------------------
# Multiprocessing worker (fallback for layouts the direct read cannot handle)
# ---------------------------------------------------------------------------

def process_chunk(file_path, dataset_name, start_idx, end_idx):
    """Return the per-pixel float32 sum over ``[start_idx, end_idx)`` of a dataset."""
    with h5py.File(file_path, "r") as f:
        chunk = _cast_to_signed(f[dataset_name][start_idx:end_idx])
        return np.sum(chunk, axis=0, dtype=np.float32)


# A subset of the frames (see resolve_frame_range) is never smaller than this,
# and is a fifth of the dataset unless a fraction is given.
_MIN_SUBSET_FRAMES = 1000
_DEFAULT_SUBSET_FRACTION = 0.2


def resolve_frame_range(total_frames, start_frame, num_frames):
    """Clamp a requested frame range to what the dataset contains.

    ``num_frames`` semantics, shared by every format loader:
      * ``>= 1`` — exactly that many frames (clamped to the end);
      * ``0`` / ``None`` — all remaining frames from ``start_frame``;
      * a fraction between 0 and 1 — a representative subset: that share of
        the dataset's frames, but at least 1000 of them;
      * ``< 0`` — the default subset, the same as ``0.2``.

    Returns the number of frames to read, starting at ``start_frame``.
    """
    if start_frame < 0 or start_frame >= total_frames:
        raise ValueError(f"start_frame must be between 0 and {total_frames - 1}")
    if num_frames is None or num_frames == 0:
        num_frames = total_frames - start_frame
    elif num_frames < 1:
        fraction = num_frames if num_frames > 0 else _DEFAULT_SUBSET_FRACTION
        num_frames = max(_MIN_SUBSET_FRAMES, int(total_frames * fraction))
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

    A dataset stored one frame per chunk with the bitshuffle, LZ4 or Blosc
    filter (or uncompressed) is read by worker threads that bypass the HDF5
    library: the chunk byte-offsets are scanned once via h5py, then each thread
    reads and decompresses its share of the chunks directly (see
    :func:`_probe_direct`).  Any other layout is read through h5py in worker
    processes.

    ``num_processes`` is the number of workers of either kind; the default is
    half the CPU count, and no more than 32 threads.  See
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

        try:
            layout = _probe_direct(dataset, start_frame, num_frames)
            if layout is not None:
                return _average_direct(layout, num_processes)
        except Exception as exc:
            logger.warning(
                "direct chunk read failed (%s), reading through h5py instead", exc
            )

        # Small ranges are cheaper to read in a single process.
        if num_frames < chunk_size:
            frames = _cast_to_signed(dataset[start_frame: start_frame + num_frames])
            return (np.sum(frames, axis=0, dtype=np.float32) / num_frames).astype(
                np.float32
            )

        if num_processes is None:
            num_processes = max(1, cpu_count() // 2)
        # Grow chunks toward one-per-worker to amortize the per-chunk file open,
        # but cap each chunk's in-memory size so wide detectors don't blow up RAM.
        frame_nbytes = int(np.prod(dataset.shape[1:])) * dataset.dtype.itemsize
        budget_frames = max(1, _CHUNK_MEMORY_BUDGET // max(frame_nbytes, 1))
        per_worker = -(-num_frames // num_processes)  # ceil division
        chunk_size = max(chunk_size, min(per_worker, budget_frames))

        stop = start_frame + num_frames
        chunks = [
            (file_path, dataset_name, i, min(i + chunk_size, stop))
            for i in range(start_frame, stop, chunk_size)
        ]
    # h5py file is closed here; all needed metadata is in local variables.

    num_processes = min(len(chunks), num_processes)
    logger.info("using %d cores to load %d frames", num_processes, num_frames)
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_chunk, chunks)

    # results are per-chunk (H, W) float32 sums; fold without stacking into an
    # (n_chunks, H, W) intermediate.
    return (sum(results) / num_frames).astype(np.float32)
