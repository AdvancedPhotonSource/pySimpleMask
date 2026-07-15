# XPCS Result File Reader — Design Spec

**Date:** 2026-07-10  
**Branch:** mc_dev  
**Status:** Approved

## Problem

pySimpleMask currently loads raw detector datasets (HDF5 + sibling `*_metadata.hdf`, IMM,
Rigaku binary, TIFF). XPCS analysis pipelines (e.g. boost-corr) produce result HDF5 files
that already contain a temporally-averaged scattering image and a ready-made partition
(`/xpcs/qmap`). Users want to re-mask, re-partition, or inspect these result files without
having access to the original raw frames.

## Detection Criterion

An HDF5 file is treated as an XPCS result file if and only if it contains a top-level
`/xpcs` group. Detection happens in `file_handler.get_handler` before any beamline
dispatch. Example result file:

```
/scratch/MQICHU/Datasets/xpcs_edge_computing_datasets/eiger4m/cluster_results/
  D0131_US-Cup2_a0010_f005000_r00001_results.hdf
```

## File Layout (expected)

```
/xpcs/temporal_mean/scattering_2d   Dataset {1, H, W}   # pre-averaged image
/xpcs/qmap/beam_center_x            Dataset {SCALAR}
/xpcs/qmap/beam_center_y            Dataset {SCALAR}
/xpcs/qmap/energy                   Dataset {SCALAR}
/xpcs/qmap/detector_distance        Dataset {SCALAR}
/xpcs/qmap/pixel_size               Dataset {SCALAR}
/xpcs/qmap/map_names                Dataset {2}
/xpcs/qmap/map_units                Dataset {2}
/xpcs/qmap/mask                     Dataset {H, W}
/xpcs/qmap/blemish                  Dataset {H, W}
/xpcs/qmap/dynamic_roi_map          Dataset {H, W}
/xpcs/qmap/static_roi_map           Dataset {H, W}
/xpcs/qmap/dynamic_index_mapping    Dataset {N}
/xpcs/qmap/dynamic_num_pts          Dataset {2}
/xpcs/qmap/dynamic_v_list_dim0      Dataset {N}
/xpcs/qmap/dynamic_v_list_dim1      Dataset {1}
/xpcs/qmap/static_index_mapping     Dataset {M}
/xpcs/qmap/static_num_pts           Dataset {2}
/xpcs/qmap/static_v_list_dim0       Dataset {M}
/xpcs/qmap/static_v_list_dim1       Dataset {1}
/xpcs/rawdata_path                  Dataset {SCALAR}    # optional, informational
```

Fallback metadata source when `/xpcs/qmap` scalars are absent:
`/entry/instrument/detector_1` + `/entry/instrument/incident_beam`
(same paths used by `APS8IDIReader`).

## Architecture

### Detection & Dispatch (`core/file_handler.py`)

```python
import h5py

def get_handler(beamline, fname, **kwargs):
    try:
        if h5py.is_hdf5(fname):
            with h5py.File(fname, "r") as f:
                if "/xpcs" in f:
                    from .reader.beamlines.xpcs_result import XPCSResultReader
                    return XPCSResultReader(fname)
        return get_reader(beamline, fname, **kwargs)
    except Exception:
        logger.error("failed to create a reader for beamline=%s file=%s",
                     beamline, fname, exc_info=True)
        return None
```

The beamline dropdown selection is ignored when auto-detect triggers. No GUI changes are
required. The peek costs one read-only HDF5 open, closed immediately after the group check.

### New Reader (`core/reader/beamlines/xpcs_result.py`)

Subclasses `FileReader`. Key responsibilities:

**`get_scattering(**kwargs)`**  
Reads `/xpcs/temporal_mean/scattering_2d[0]` and returns a 2D `float32` array `(H, W)`.
All `begin_idx`/`num_frames` kwargs are accepted and silently discarded — the image is
already temporally averaged.

**`_get_metadata()`**  
Two-tier fallback:
1. Read scalar fields from `/xpcs/qmap` (preferred — these are the refined values used
   for the XPCS analysis). Triggered when the group exists **and** all required scalar
   keys are present within it.
2. If `/xpcs/qmap` is missing **or** any required scalar key within it is absent, fall
   back to `/entry/instrument/detector_1` and `/entry/instrument/incident_beam` using the
   shared `read_keymap` helper. If fallback also fails, `get_metadata()` in the base
   class catches and returns `get_fake_metadata()`.

Required scalars: `beam_center_x`, `beam_center_y`, `energy`, `detector_distance`,
`pixel_size`. Uses `DEFAULT_METADATA_WITHUNITS` from `aps_8idi.py` for units/formats
(same detector convention).

**`self.saved_partition`** (populated in `__init__`)  
A dict matching the structure produced by `compute_partition()`, loaded from `/xpcs/qmap`.
If the partition arrays (`dynamic_roi_map`, `static_roi_map`, etc.) are absent from the
file, `saved_partition` is set to `None` — the image still loads normally, but no
partition is restored. `source_file` bytes values are decoded to `str`.

```python
{
    "beam_center_x": ..., "beam_center_y": ..., "energy": ...,
    "detector_distance": ..., "pixel_size": ...,
    "map_names": ..., "map_units": ...,
    "mask": ndarray, "blemish": ndarray, "source_file": str,
    "dynamic_roi_map": ndarray, "static_roi_map": ndarray,
    "dynamic_index_mapping": ndarray, "dynamic_num_pts": ndarray,
    "dynamic_v_list_dim0": ndarray, "dynamic_v_list_dim1": ndarray,
    "static_index_mapping": ndarray, "static_num_pts": ndarray,
    "static_v_list_dim0": ndarray, "static_v_list_dim1": ndarray,
}
```

Class sketch:

```python
class XPCSResultReader(FileReader):
    ftype = "XPCS_Result"
    stype = "Transmission"

    def __init__(self, fname):
        super().__init__(fname)
        self.meta_units_fmts = DEFAULT_METADATA_WITHUNITS.copy()
        self.saved_partition = self._load_partition()

    def get_scattering(self, **kwargs):
        with h5py.File(self.fname, "r") as f:
            return f["/xpcs/temporal_mean/scattering_2d"][0].astype(np.float32)

    def _get_metadata(self): ...
    def _load_partition(self): ...
```

### Model Partition Restoration (`core/model.py`)

One new block added to `read_data()` after `mask_kernel` is set up:

```python
if hasattr(self.dset, "saved_partition"):
    p = self.dset.saved_partition
    self.dset.update_partitions(p["dynamic_roi_map"], p["static_roi_map"])
    self.new_partition = p
```

Duck-typed on `saved_partition` — no `isinstance` check. The qmap (per-pixel q/phi/x/y
maps) is still computed from the scalar metadata via `compute_qmap()` in the normal path,
so coordinate readout, display channels, and recomputing a different partition all work
normally. Users can modify parameters and recompute a fresh partition at any time.

## GUI Impact

None. The beamline comboBox (`APS_8IDI`, `APS_9IDD`, `NativeFiles`) is unaffected.
Frame-range spinners (`begin_idx`, `num_frames`) are silently ignored for result files.
No `.ui` changes, no `make ui` needed.

## Testing (`tests/core/reader/test_xpcs_result_reader.py`)

Synthetic HDF5 fixtures built in-memory with `h5py` (16×16 arrays). No real data files.

| Test | What it checks |
|------|----------------|
| Auto-detect overrides beamline | `get_handler("APS_8IDI", result_fixture)` → `XPCSResultReader` |
| Non-result file unaffected | `get_handler("APS_8IDI", raw_fixture)` → `APS8IDIReader` |
| Scattering shape | `get_scattering()` returns `(H, W)`, leading dim squeezed |
| Metadata from `/xpcs/qmap` | Scalar fields read correctly |
| Metadata fallback | Missing `/xpcs/qmap` scalars → reads from `/entry/instrument` |
| `saved_partition` keys | All expected keys present after `__init__` |
| Frame kwargs ignored | `get_scattering(begin_idx=5, num_frames=100)` == no-kwargs result |
| Model integration | `model.read_data(result_fixture)` sets `new_partition` and non-zero partition display channels |

## Files Changed

| File | Change |
|------|--------|
| `src/pysimplemask/core/file_handler.py` | Add `/xpcs` peek before beamline dispatch |
| `src/pysimplemask/core/reader/beamlines/xpcs_result.py` | **New** — `XPCSResultReader` |
| `src/pysimplemask/core/model.py` | Restore `saved_partition` after `prepare_data` |
| `tests/core/reader/test_xpcs_result_reader.py` | **New** — 8 test cases, synthetic fixtures |

No changes to `reader/__init__.py`, `base_reader.py`, GUI, or `.ui` files.
