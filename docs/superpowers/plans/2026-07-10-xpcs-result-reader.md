# XPCS Result File Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `XPCSResultReader` so pySimpleMask can load XPCS result HDF5 files (those containing `/xpcs`), automatically restoring the pre-computed scattering image and partition maps.

**Architecture:** A new `XPCSResultReader` subclasses `FileReader` and handles the `_results.hdf` format. Auto-detection lives in `file_handler.get_handler`, which peeks for `/xpcs` before delegating to the beamline-based dispatch. `model.read_data` restores the saved partition after `prepare_data` if the reader provides one.

**Tech Stack:** Python, h5py, numpy, pytest

## Global Constraints

- Environment: `/local/MQICHU/envs/l2606_simplemask_refact/bin/`
- Run tests with: `/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest`
- `core/` must stay Qt-free — never import PySide6/pyqtgraph inside `core/`
- Follow existing fixture pattern: synthetic in-memory HDF5 via `h5py`, no real data files
- Linter: `ruff check src tests` must pass (existing baseline)
- `saved_partition` is `None` when partition arrays are absent — graceful degradation
- Fallback metadata source order: `/xpcs/qmap` scalars → `/entry/instrument` paths

---

## File Map

| Status | File | Change |
|--------|------|--------|
| Create | `src/pysimplemask/core/reader/beamlines/xpcs_result.py` | `XPCSResultReader` class |
| Modify | `tests/core/reader/conftest.py` | Add `make_xpcs_result` fixture |
| Create | `tests/core/reader/test_xpcs_result_reader.py` | All 8 test cases |
| Modify | `src/pysimplemask/core/file_handler.py` | `/xpcs` peek before dispatch |
| Modify | `tests/core/reader/test_dispatch.py` | 2 new auto-detect tests |
| Modify | `src/pysimplemask/core/model.py` | Restore `saved_partition` in `read_data` |

---

## Task 1: `XPCSResultReader` — reader class + reader-level tests

**Files:**
- Create: `src/pysimplemask/core/reader/beamlines/xpcs_result.py`
- Modify: `tests/core/reader/conftest.py` (add fixture)
- Create: `tests/core/reader/test_xpcs_result_reader.py`

**Interfaces:**
- Produces: `XPCSResultReader(fname)` — subclass of `FileReader` with attributes
  `ftype = "XPCS_Result"`, `stype = "Transmission"`, `saved_partition: dict | None`
- Produces: `get_scattering(**kwargs) -> np.ndarray` shape `(H, W)` dtype `float32`
- Produces: `_get_metadata() -> dict` with keys `beam_center_x`, `beam_center_y`,
  `energy`, `detector_distance`, `pixel_size`
- Produces: `make_xpcs_result` pytest fixture (used by Tasks 1, 2, 3)

- [ ] **Step 1: Add the `make_xpcs_result` fixture to conftest.py**

Append to `tests/core/reader/conftest.py`:

```python
# ---------------------------------------------------------------------------
# XPCS result files
# ---------------------------------------------------------------------------

@pytest.fixture
def make_xpcs_result(tmp_path):
    """Write a minimal XPCS result HDF5 file with /xpcs structure.

    Parameters (all keyword-only via _make):
      with_qmap_scalars  – write beam_center/energy/distance/pixel_size under /xpcs/qmap
      with_partition     – write roi maps and index arrays under /xpcs/qmap
      with_entry_inst    – write /entry/instrument fallback metadata
    """
    H, W = 16, 16

    def _make(
        name="result.hdf",
        with_qmap_scalars=True,
        with_partition=True,
        with_entry_inst=True,
    ):
        path = tmp_path / name
        with h5py.File(path, "w") as f:
            f["/xpcs/temporal_mean/scattering_2d"] = np.ones(
                (1, H, W), dtype=np.float32
            ) * 5.0

            if with_qmap_scalars:
                f["/xpcs/qmap/beam_center_x"] = 100.0
                f["/xpcs/qmap/beam_center_y"] = 80.0
                f["/xpcs/qmap/energy"] = 10.0
                f["/xpcs/qmap/detector_distance"] = 5.0
                f["/xpcs/qmap/pixel_size"] = 7.5e-5

            if with_partition:
                dmap = np.zeros((H, W), dtype=np.int32)
                dmap[4:8, 4:8] = 1
                smap = np.zeros((H, W), dtype=np.int32)
                smap[4:8, 4:8] = 1
                f["/xpcs/qmap/dynamic_roi_map"] = dmap
                f["/xpcs/qmap/static_roi_map"] = smap
                f["/xpcs/qmap/dynamic_index_mapping"] = np.array([1], dtype=np.int32)
                f["/xpcs/qmap/dynamic_num_pts"] = np.array([1, 16], dtype=np.int32)
                f["/xpcs/qmap/dynamic_v_list_dim0"] = np.array([0.1])
                f["/xpcs/qmap/dynamic_v_list_dim1"] = np.array([0.0])
                f["/xpcs/qmap/static_index_mapping"] = np.array([1], dtype=np.int32)
                f["/xpcs/qmap/static_num_pts"] = np.array([1, 16], dtype=np.int32)
                f["/xpcs/qmap/static_v_list_dim0"] = np.array([0.1])
                f["/xpcs/qmap/static_v_list_dim1"] = np.array([0.0])
                f["/xpcs/qmap/mask"] = np.ones((H, W), dtype=np.uint8)
                f["/xpcs/qmap/blemish"] = np.ones((H, W), dtype=np.uint8)
                f["/xpcs/qmap/map_names"] = np.array([b"q", b"phi"])
                f["/xpcs/qmap/map_units"] = np.array([b"1/A", b"degree"])
                f["/xpcs/qmap/source_file"] = b"/path/to/raw.h5"

            if with_entry_inst:
                f["/entry/instrument/detector_1/beam_center_x"] = 100.0
                f["/entry/instrument/detector_1/beam_center_y"] = 80.0
                f["/entry/instrument/detector_1/x_pixel_size"] = 7.5e-5
                f["/entry/instrument/detector_1/distance"] = 5.0
                f["/entry/instrument/incident_beam/incident_energy"] = 10.0

        return str(path)

    return _make
```

- [ ] **Step 2: Write the failing tests**

Create `tests/core/reader/test_xpcs_result_reader.py`:

```python
"""Tests for XPCSResultReader — scattering, metadata, partition loading."""

import numpy as np
import pytest

from pysimplemask.core.reader.beamlines.xpcs_result import XPCSResultReader

H, W = 16, 16

REQUIRED_PARTITION_KEYS = {
    "dynamic_roi_map",
    "static_roi_map",
    "dynamic_index_mapping",
    "dynamic_num_pts",
    "dynamic_v_list_dim0",
    "dynamic_v_list_dim1",
    "static_index_mapping",
    "static_num_pts",
    "static_v_list_dim0",
    "static_v_list_dim1",
    "mask",
    "blemish",
    "map_names",
    "map_units",
    "source_file",
}


def test_scattering_shape(make_xpcs_result):
    reader = XPCSResultReader(make_xpcs_result())
    scat = reader.get_scattering()
    assert scat.shape == (H, W)
    assert scat.dtype == np.float32


def test_metadata_from_xpcs_qmap(make_xpcs_result):
    reader = XPCSResultReader(make_xpcs_result())
    meta = reader.get_metadata()
    assert meta["beam_center_x"] == pytest.approx(100.0)
    assert meta["beam_center_y"] == pytest.approx(80.0)
    assert meta["energy"] == pytest.approx(10.0)
    assert meta["detector_distance"] == pytest.approx(5.0)
    assert meta["pixel_size"] == pytest.approx(7.5e-5)


def test_metadata_fallback_to_entry_instrument(make_xpcs_result):
    path = make_xpcs_result(with_qmap_scalars=False, with_partition=False)
    reader = XPCSResultReader(path)
    meta = reader.get_metadata()
    assert meta["beam_center_x"] == pytest.approx(100.0)
    assert meta["energy"] == pytest.approx(10.0)
    assert meta["pixel_size"] == pytest.approx(7.5e-5)


def test_frame_kwargs_silently_ignored(make_xpcs_result):
    reader = XPCSResultReader(make_xpcs_result())
    scat_default = reader.get_scattering()
    scat_with_kwargs = reader.get_scattering(begin_idx=5, num_frames=100)
    np.testing.assert_array_equal(scat_default, scat_with_kwargs)


def test_saved_partition_contains_required_keys(make_xpcs_result):
    reader = XPCSResultReader(make_xpcs_result())
    assert reader.saved_partition is not None
    assert REQUIRED_PARTITION_KEYS.issubset(reader.saved_partition.keys())


def test_saved_partition_is_none_when_arrays_absent(make_xpcs_result):
    path = make_xpcs_result(with_partition=False)
    reader = XPCSResultReader(path)
    assert reader.saved_partition is None
```

- [ ] **Step 3: Run the failing tests**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/core/reader/test_xpcs_result_reader.py -v
```

Expected: all 6 tests FAIL with `ModuleNotFoundError` — `xpcs_result` module doesn't exist yet.

- [ ] **Step 4: Create `src/pysimplemask/core/reader/beamlines/xpcs_result.py`**

```python
"""Reader for XPCS result HDF5 files (those containing a /xpcs group)."""

import logging

import h5py
import numpy as np

from ..base_reader import FileReader
from ..metadata import read_keymap
from .aps_8idi import DEFAULT_METADATA_WITHUNITS

logger = logging.getLogger(__name__)

# Metadata paths inside /xpcs/qmap (the preferred, already-refined values).
_XPCS_QMAP_KEYMAP = {
    "beam_center_x": "/xpcs/qmap/beam_center_x",
    "beam_center_y": "/xpcs/qmap/beam_center_y",
    "energy": "/xpcs/qmap/energy",
    "detector_distance": "/xpcs/qmap/detector_distance",
    "pixel_size": "/xpcs/qmap/pixel_size",
}

# Fallback paths in /entry/instrument when /xpcs/qmap scalars are absent.
_FALLBACK_KEYMAP = {
    "beam_center_x": "/entry/instrument/detector_1/beam_center_x",
    "beam_center_y": "/entry/instrument/detector_1/beam_center_y",
    "energy": "/entry/instrument/incident_beam/incident_energy",
    "detector_distance": "/entry/instrument/detector_1/distance",
    "pixel_size": "/entry/instrument/detector_1/x_pixel_size",
}

# Array datasets to load from /xpcs/qmap into saved_partition.
_PARTITION_ARRAY_KEYS = [
    "dynamic_roi_map",
    "static_roi_map",
    "dynamic_index_mapping",
    "dynamic_num_pts",
    "dynamic_v_list_dim0",
    "dynamic_v_list_dim1",
    "static_index_mapping",
    "static_num_pts",
    "static_v_list_dim0",
    "static_v_list_dim1",
    "mask",
    "blemish",
    "map_names",
    "map_units",
    "source_file",
]


class XPCSResultReader(FileReader):
    """Reader for XPCS analysis result HDF5 files.

    Auto-detected by ``file_handler.get_handler`` when the file contains a
    top-level ``/xpcs`` group. Reads the temporally-averaged scattering image
    and restores the existing partition from ``/xpcs/qmap``.
    """

    ftype = "XPCS_Result"
    stype = "Transmission"

    def __init__(self, fname):
        super().__init__(fname)
        self.meta_units_fmts = DEFAULT_METADATA_WITHUNITS.copy()
        self.saved_partition = self._load_partition()

    def get_scattering(self, **kwargs):
        """Return the pre-averaged 2-D scattering image.

        All kwargs (``begin_idx``, ``num_frames``, etc.) are accepted and
        silently ignored — the image is already temporally averaged.
        """
        with h5py.File(self.fname, "r") as f:
            return f["/xpcs/temporal_mean/scattering_2d"][0].astype(np.float32)

    def _get_metadata(self):
        """Read instrument metadata.

        Tries ``/xpcs/qmap`` scalars first; falls back to
        ``/entry/instrument`` paths when any required key is absent.
        """
        try:
            return read_keymap(self.fname, _XPCS_QMAP_KEYMAP)
        except (KeyError, OSError):
            logger.info(
                "xpcs/qmap scalars incomplete in %s; falling back to /entry/instrument",
                self.fname,
            )
        return read_keymap(self.fname, _FALLBACK_KEYMAP)

    def _load_partition(self):
        """Load partition arrays from ``/xpcs/qmap``.

        Returns ``None`` if the group is absent or the required roi map
        arrays are missing — the image will still load normally.
        """
        try:
            with h5py.File(self.fname, "r") as f:
                if "/xpcs/qmap" not in f:
                    return None
                grp = f["/xpcs/qmap"]
                if "dynamic_roi_map" not in grp or "static_roi_map" not in grp:
                    return None
                partition = {}
                for key in _PARTITION_ARRAY_KEYS:
                    if key not in grp:
                        continue
                    val = grp[key][()]
                    if isinstance(val, bytes):
                        val = val.decode("utf-8", errors="replace")
                    partition[key] = val
                for key in _XPCS_QMAP_KEYMAP:
                    if key in grp:
                        partition[key] = float(grp[key][()])
                return partition
        except Exception:
            logger.warning(
                "failed to load partition from %s", self.fname, exc_info=True
            )
            return None
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/core/reader/test_xpcs_result_reader.py -v
```

Expected output (all pass):
```
PASSED test_scattering_shape
PASSED test_metadata_from_xpcs_qmap
PASSED test_metadata_fallback_to_entry_instrument
PASSED test_frame_kwargs_silently_ignored
PASSED test_saved_partition_contains_required_keys
PASSED test_saved_partition_is_none_when_arrays_absent
```

- [ ] **Step 6: Verify ruff is clean**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check \
    src/pysimplemask/core/reader/beamlines/xpcs_result.py \
    tests/core/reader/test_xpcs_result_reader.py \
    tests/core/reader/conftest.py
```

Expected: no output (no errors).

- [ ] **Step 7: Commit**

```bash
git add \
    src/pysimplemask/core/reader/beamlines/xpcs_result.py \
    tests/core/reader/test_xpcs_result_reader.py \
    tests/core/reader/conftest.py
git commit -m "feat(reader): add XPCSResultReader for /xpcs result HDF5 files"
```

---

## Task 2: Auto-detect in `file_handler` + dispatch tests

**Files:**
- Modify: `src/pysimplemask/core/file_handler.py` (lines 1–23)
- Modify: `tests/core/reader/test_dispatch.py`

**Interfaces:**
- Consumes: `XPCSResultReader` from Task 1
- Consumes: `make_xpcs_result` fixture from Task 1
- Produces: `get_handler(beamline, fname)` returns `XPCSResultReader` when `/xpcs` found,
  regardless of `beamline` argument

- [ ] **Step 1: Write the two new failing dispatch tests**

Append to `tests/core/reader/test_dispatch.py`:

```python
# ---------------------------------------------------------------------------
# XPCS result auto-detection
# ---------------------------------------------------------------------------


def test_get_handler_xpcs_result_overrides_beamline(make_xpcs_result):
    from pysimplemask.core.reader.beamlines.xpcs_result import XPCSResultReader

    path = make_xpcs_result()
    reader = get_handler("APS_8IDI", path)
    assert isinstance(reader, XPCSResultReader)


def test_get_handler_plain_hdf_not_misdetected_as_xpcs(make_hdf):
    path = make_hdf(np.zeros((2, 4, 5)))
    reader = get_handler("APS_8IDI", path)
    assert isinstance(reader, APS8IDIReader)
```

- [ ] **Step 2: Run the failing tests**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/core/reader/test_dispatch.py::test_get_handler_xpcs_result_overrides_beamline \
    tests/core/reader/test_dispatch.py::test_get_handler_plain_hdf_not_misdetected_as_xpcs \
    -v
```

Expected: both FAIL — `get_handler` returns an `APS8IDIReader`, not `XPCSResultReader`.

- [ ] **Step 3: Replace `src/pysimplemask/core/file_handler.py`**

```python
import logging

import h5py

from .reader import get_reader

logger = logging.getLogger(__name__)


def get_handler(beamline, fname, **kwargs):
    """Return a reader for the given file, or ``None`` on failure.

    If ``fname`` is an HDF5 file containing a top-level ``/xpcs`` group it is
    treated as an XPCS result file regardless of ``beamline``; otherwise the
    ``beamline`` string drives dispatch as usual.
    """
    try:
        if h5py.is_hdf5(fname):
            with h5py.File(fname, "r") as f:
                if "/xpcs" in f:
                    from .reader.beamlines.xpcs_result import XPCSResultReader

                    return XPCSResultReader(fname)
        return get_reader(beamline, fname, **kwargs)
    except Exception:
        logger.error(
            "failed to create a reader for beamline=%s file=%s",
            beamline,
            fname,
            exc_info=True,
        )
        return None
```

- [ ] **Step 4: Run the new dispatch tests**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/core/reader/test_dispatch.py -v
```

Expected: all dispatch tests pass, including the two new ones.

- [ ] **Step 5: Run full reader test suite to catch regressions**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/core/reader/ -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add \
    src/pysimplemask/core/file_handler.py \
    tests/core/reader/test_dispatch.py
git commit -m "feat(file_handler): auto-detect XPCS result files by /xpcs group"
```

---

## Task 3: Model partition restoration + integration test

**Files:**
- Modify: `src/pysimplemask/core/model.py` (inside `read_data`, after line 204)
- Modify: `tests/core/reader/test_xpcs_result_reader.py`

**Interfaces:**
- Consumes: `XPCSResultReader.saved_partition` from Task 1
- Consumes: `get_handler` auto-detect from Task 2
- Consumes: `make_xpcs_result` fixture from Task 1
- Produces: after `model.read_data(result_file)`, `model.new_partition` is a dict with
  `"dynamic_roi_map"` and `"static_roi_map"` keys; `model.dset.data_display[3]`
  (the `dqmap_partition` channel, index 3 in `DISPLAY_FIELD`) is non-zero

- [ ] **Step 1: Write the failing integration test**

Append to `tests/core/reader/test_xpcs_result_reader.py`:

```python
def test_model_read_data_restores_partition(make_xpcs_result):
    from pysimplemask.core.model import SimpleMaskModel

    path = make_xpcs_result()
    model = SimpleMaskModel()
    assert model.read_data(path, beamline="APS_8IDI") is True

    # Partition dict is set on the model.
    assert model.new_partition is not None
    assert "dynamic_roi_map" in model.new_partition
    assert "static_roi_map" in model.new_partition

    # Display channel 3 (dqmap_partition) is non-zero where fixture dmap==1.
    # DISPLAY_FIELD = ["scattering", "scattering * mask", "mask",
    #                  "dqmap_partition", "sqmap_partition", "preview"]
    dqmap_display = model.dset.data_display[3]
    assert dqmap_display[4:8, 4:8].any(), "dqmap partition not written to display"
```

- [ ] **Step 2: Run the failing test**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/core/reader/test_xpcs_result_reader.py::test_model_read_data_restores_partition \
    -v
```

Expected: FAIL — `model.new_partition` is `None` because the restoration block doesn't exist yet.

- [ ] **Step 3: Add the partition restoration block to `model.py`**

In `src/pysimplemask/core/model.py`, find `read_data` (line 187). The last three lines of
the method body currently are:

```python
        self.mask_apply(target="default_blemish")
        self.mask_kernel.update_qmap(self.qmap)
        return True
```

Replace those three lines with:

```python
        self.mask_apply(target="default_blemish")
        self.mask_kernel.update_qmap(self.qmap)

        if getattr(self.dset, "saved_partition", None) is not None:
            p = self.dset.saved_partition
            self.dset.update_partitions(p["dynamic_roi_map"], p["static_roi_map"])
            self.new_partition = p

        return True
```

- [ ] **Step 4: Run the integration test**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest \
    tests/core/reader/test_xpcs_result_reader.py::test_model_read_data_restores_partition \
    -v
```

Expected: PASS.

- [ ] **Step 5: Run the full test suite**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -v
```

Expected: all 111 tests pass (103 existing + 8 new). No regressions.

- [ ] **Step 6: Verify ruff is clean across all changed files**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check \
    src/pysimplemask/core/file_handler.py \
    src/pysimplemask/core/model.py \
    src/pysimplemask/core/reader/beamlines/xpcs_result.py \
    tests/core/reader/conftest.py \
    tests/core/reader/test_dispatch.py \
    tests/core/reader/test_xpcs_result_reader.py
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add \
    src/pysimplemask/core/model.py \
    tests/core/reader/test_xpcs_result_reader.py
git commit -m "feat(model): restore saved_partition from XPCSResultReader after load"
```
