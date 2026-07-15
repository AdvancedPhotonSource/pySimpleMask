"""Raw-frame IO helpers shared across format loaders."""

import ctypes
import logging
import os
import pathlib
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count

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

# ---------------------------------------------------------------------------
# GIL detection — evaluated once at module import (= app startup).
# sys._is_gil_enabled() is present on CPython 3.13+ free-threaded builds;
# older interpreters always have the GIL so we default to True.
# ---------------------------------------------------------------------------
_GIL_ENABLED: bool = getattr(sys, "_is_gil_enabled", lambda: True)()
if not _GIL_ENABLED:
    logger.info(
        "Free-threaded Python detected (GIL disabled) — "
        "thread-parallel HDF5-bypass path will be used for LZ4-chunked datasets."
    )

_CHUNK_MEMORY_BUDGET = 256 * 1024 * 1024  # 256 MiB

# HDF5 pure-LZ4 block filter (id 32004).  Wire format per chunk:
#   bytes  0–7:  total uncompressed bytes  (uint64, big-endian)
#   bytes  8–11: block size               (uint32, big-endian)
#   bytes 12–15: compressed block size    (uint32, big-endian)
#   bytes 16–:   LZ4 data
_LZ4_FILTER_ID = 32004
_NOGIL_DEFAULT_WORKERS = 16   # empirical elbow on Lustre: 8–16 threads


# ---------------------------------------------------------------------------
# LZ4 shared library — loaded lazily; only needed on the no-GIL path.
# Deferred so the module stays importable even without libh5lz4.so.
# ---------------------------------------------------------------------------
_lz4_lib = None


def _get_lz4_lib():
    """Return the LZ4 ctypes handle, loading it from hdf5plugin on first call."""
    global _lz4_lib
    if _lz4_lib is not None:
        return _lz4_lib
    plugin_dir = pathlib.Path(hdf5plugin.__file__).parent / "plugins"
    lib = ctypes.CDLL(str(plugin_dir / "libh5lz4.so"))
    lib.LZ4_decompress_safe.restype = ctypes.c_int
    lib.LZ4_decompress_safe.argtypes = [
        ctypes.c_char_p,  # src  (compressed bytes)
        ctypes.c_char_p,  # dst  (output buffer)
        ctypes.c_int,     # compressedSize
        ctypes.c_int,     # maxDecompressedSize
    ]
    _lz4_lib = lib
    return lib


# ---------------------------------------------------------------------------
# No-GIL workers — pure os.pread + ctypes LZ4; no h5py, no HDF5 mutex.
# ---------------------------------------------------------------------------

def _decode_lz4_chunk(raw: bytes, out: np.ndarray) -> None:
    """Parse 16-byte LZ4 filter header and decompress *raw* into *out* (flat uint32)."""
    lz4 = _get_lz4_lib()
    total_bytes = struct.unpack_from(">Q", raw, 0)[0]
    comp_size   = struct.unpack_from(">I", raw, 12)[0]
    ret = lz4.LZ4_decompress_safe(
        raw[16: 16 + comp_size],
        out.ctypes.data_as(ctypes.c_char_p),
        comp_size,
        total_bytes,
    )
    if ret != total_bytes:
        raise RuntimeError(f"LZ4 decompression: got {ret} bytes, expected {total_bytes}")


def _read_and_sum_nogil(args):
    """pread + LZ4-decompress + accumulate for one worker's slice of frames."""
    fd, chunk_list, partial, H, W = args
    frame_u32 = np.empty(H * W, dtype=np.uint32)
    for byte_offset, raw_size in chunk_list:
        _decode_lz4_chunk(os.pread(fd, int(raw_size), int(byte_offset)), frame_u32)
        partial += _cast_to_signed(frame_u32).reshape(H, W)
    return len(chunk_list)


def _try_scan_lz4(dataset, start_frame, num_frames):
    """Return a list of (byte_offset, size) for each frame if the dataset uses
    the LZ4 filter with one frame per chunk; otherwise return None.

    Called while the h5py file handle is still open.
    """
    try:
        plist = dataset.id.get_create_plist()
        chunk_shape = dataset.chunks
        if (plist.get_nfilters() < 1
                or plist.get_filter(0)[0] != _LZ4_FILTER_ID
                or chunk_shape is None
                or chunk_shape[0] != 1):
            return None
        _get_lz4_lib()  # verify the shared lib is loadable before committing
        logger.info("no-GIL: scanning %d LZ4 chunk byte-offsets", num_frames)
        return [
            (info.byte_offset, info.size)
            for i in range(start_frame, start_frame + num_frames)
            for info in (dataset.id.get_chunk_info_by_coord((i, 0, 0)),)
        ]
    except Exception as exc:
        logger.debug("no-GIL LZ4 path unavailable (%s), falling back to multiprocessing", exc)
        return None


def _run_nogil_average(file_path, lz4_frames, H, W, num_frames):
    """Thread-parallel reduction over a pre-scanned list of LZ4 chunk offsets.

    Phase 1 (already done by caller): chunk byte-offsets collected via h5py.
    Phase 2 (here): os.pread + LZ4_decompress_safe in ThreadPoolExecutor —
    threads run freely without GIL serialisation.
    """
    splits = [s for s in np.array_split(lz4_frames, _NOGIL_DEFAULT_WORKERS) if len(s)]
    n_workers = len(splits)
    partials = np.zeros((n_workers, H, W), dtype=np.float32)
    fd = os.open(str(file_path), os.O_RDONLY)
    try:
        jobs = [(fd, list(s), p, H, W) for s, p in zip(splits, partials)]
        logger.info("averaging %d frames with %d threads (no-GIL HDF5-bypass)", num_frames, n_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(executor.map(_read_and_sum_nogil, jobs))
    finally:
        os.close(fd)
    return (partials.sum(axis=0) / num_frames).astype(np.float32)


# ---------------------------------------------------------------------------
# Multiprocessing worker (GIL-enabled / non-LZ4 fallback path)
# ---------------------------------------------------------------------------

def process_chunk(file_path, dataset_name, start_idx, end_idx):
    """Return the per-pixel float32 sum over ``[start_idx, end_idx)`` of a dataset."""
    with h5py.File(file_path, "r") as f:
        chunk = _cast_to_signed(f[dataset_name][start_idx:end_idx])
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

    On free-threaded Python (no GIL) with a dataset compressed by the LZ4 HDF5
    filter (id 32004, one frame per chunk), this uses a thread-parallel
    HDF5-bypass: chunk byte-offsets are scanned once via h5py, then worker
    threads call os.pread + LZ4_decompress_safe directly, bypassing the HDF5
    serialisation mutex entirely.  On GIL-enabled Python or non-LZ4 datasets
    the original multiprocessing path is used.

    See :func:`resolve_frame_range` for ``num_frames`` semantics.

    Returns:
        np.ndarray: 2-D ``float32`` mean image.
    """
    _lz4_frames = None  # set to a list of (byte_offset, size) on the no-GIL path

    with h5py.File(file_path, "r") as f:
        dataset = f[dataset_name]
        if dataset.ndim != 3:
            raise ValueError("expected a 3-D (frame, y, x) dataset")

        total_frames = dataset.shape[0]
        logger.info("Total frames in dataset: %d", total_frames)
        num_frames = resolve_frame_range(total_frames, start_frame, num_frames)

        # Small ranges are cheaper to read in a single process.
        if num_frames < chunk_size:
            frames = _cast_to_signed(dataset[start_frame: start_frame + num_frames])
            return (np.sum(frames, axis=0, dtype=np.float32) / num_frames).astype(
                np.float32
            )

        H, W = int(dataset.shape[1]), int(dataset.shape[2])

        # No-GIL fast path: scan LZ4 chunk byte-offsets while the file is open.
        if not _GIL_ENABLED:
            _lz4_frames = _try_scan_lz4(dataset, start_frame, num_frames)

        if _lz4_frames is None:
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
    # h5py file is closed here; all needed metadata is in local variables.

    # No-GIL thread path (LZ4-chunked dataset on free-threaded Python).
    if _lz4_frames is not None:
        return _run_nogil_average(file_path, _lz4_frames, H, W, num_frames)

    # Multiprocessing path (GIL-enabled Python or non-LZ4 dataset).
    num_processes = min(len(chunks), num_processes)
    logger.info("using %d cores to load %d frames", num_processes, num_frames)
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_chunk, chunks)

    # results are per-chunk (H, W) float32 sums; fold without stacking into an
    # (n_chunks, H, W) intermediate.
    return (sum(results) / num_frames).astype(np.float32)
