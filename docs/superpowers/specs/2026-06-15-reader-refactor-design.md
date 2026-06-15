# Reader subsystem refactor — design

**Date:** 2026-06-15
**Branch:** refact
**Scope:** `src/pysimplemask/reader/` only (plus the two import lines / delegation in `src/pysimplemask/file_handler.py`). No changes to kernel, qmap, mask, UI, or geometry.

## Goal

Restructure the reader subsystem into clean, consistent, well-tested layers; remove
dead code; fix bugs and vulnerabilities. The kernel/GUI-facing contract is unchanged:
`file_handler.get_handler(beamline, fname)` returns an object exposing the full
`FileReader` API.

## Decisions (settled with the user)

1. **Remove all dead/unwired code** — no WIP scaffolding kept.
2. **Bold layer restructure** — separate format loaders from beamline readers; slim the base.
3. **Unify shared logic** — shared NeXus metadata helpers; 8-ID-I/9-ID-D made symmetric.
4. **Verify with synthetic fixtures + unit tests** — no external data dependency.
5. **Layout B (layered):** `formats/` + `beamlines/`.
6. **Dispatch ownership:** `get_reader()` lives in `reader/__init__.py`; `file_handler.get_handler` becomes a thin delegator.
7. **Frame convention:** all loaders return the **per-pixel MEAN** over the selected frame range (consistent, scale-stable). This changes HDF/Rigaku output by a factor of `1/num_frames`; IMM (already mean) is unchanged.

## Non-goals

- No changes to `qmap.py`, `simplemask_kernel.py`, `area_mask.py`, the UI, or geometry math.
- No new beamlines or file formats; Timepix stays an optional external dependency
  (`timepix_dataset`), lazily imported.
- Not fixing the unrelated stale `tests/test_pysimplemask.py` (optional cleanup only).

## Two concepts, cleanly separated

- **Reader** (`FileReader` + beamline subclasses): the app-facing object *per beamline*.
  Owns metadata, qmap, the `data_display` stack, mask state, center handling. The
  kernel/GUI talk only to this. Public API is preserved exactly.
- **Format loader** (`ScatteringDataset` + subclasses): the low-level thing that turns a
  raw *file format* (HDF/IMM/Rigaku) into a 2-D scattering image. Only `get_scattering()`
  and `det_size` are consumed.

## Target structure

```
reader/
  __init__.py        get_reader(beamline, fname) dispatch + re-export FileReader
  base_reader.py     FileReader (app-facing API unchanged) + DISPLAY_FIELD + param-tree helper
  metadata.py        locate metadata file, has_nexus_fields, keymap read, beam-center helpers
  io_utils.py        average_frames_parallel + process_chunk (raw frame IO)
  formats/
    __init__.py      get_format_loader(fname): extension -> loader instance
    base.py          ScatteringDataset ABC
    hdf.py           HdfDataset
    imm.py           ImmDataset (TOC summing inlined)
    rigaku.py        RigakuDataset, Rigaku3MDataset, convert_sparse (single copy)
  beamlines/
    __init__.py
    aps_8idi.py      APS8IDIReader: defaults + keymap + 8IDI beam-center
    aps_9idd.py      APS9IDDReader: defaults + keymap + 9IDD beam-center/specular
```

**Deleted:** `timepix_reader.py`, `esrf_id02/` (whole dir), `APS_8IDI/rigaku_six_handler.py`,
`APS_8IDI/test_rigaku3m.py`, `APS_8IDI/xpcs_dataset.py` (replaced by `formats/base.py`), the
old `APS_8IDI/` and `APS_9IDD/` packages (migrated into `formats/` + `beamlines/`),
`utils.py` (split into `metadata.py` + `io_utils.py`), and all `test*/read_data/__main__`
debug blocks and hardcoded personal paths.

## The format-loader interface

```python
class ScatteringDataset(ABC):
    fname: str
    det_size: tuple[int, int]

    @abstractmethod
    def get_scattering(self, num_frames=-1, begin_idx=0, num_processes=None) -> np.ndarray:
        """Return the per-pixel MEAN scattering image over the selected frame range
        as a 2-D float array of shape det_size."""
```

- Uniform signature across all loaders (fixes today's per-class signature drift;
  loaders that don't parallelize simply ignore `num_processes`).
- Everything else from the old `XpcsDataset` is dropped (unused by the app): torch
  `device`/`.cpu().numpy()`, batching (`__getbatch__`, `get_raw_index`,
  `update_batch_info`), `get_data`, `__getitem__`, `to_rigaku_bin`, `sparse_to_dense`,
  `get_sparsity`, `describe`, `update_mask_crop`, dataloader hooks.

### Per-format behavior (mean convention)

- **HDF** (`hdf.py`): `average_frames_parallel` (sums chunks in parallel, then divides by the number of frames summed).
- **Rigaku 500k** (`rigaku.py`): `coo[begin:end].sum(axis=0) / (end - begin)`.
- **Rigaku 3M** (`rigaku.py`): per-module mean, then stitched (`patch_data`).
- **IMM** (`imm.py`): mean over the TOC frame range (already mean today; reimplemented
  without the batch abstraction).
- Division guards against `n_frames == 0`.

## Beamline readers (symmetry + shared metadata)

`metadata.py` provides the shared seam:

```python
def find_metadata_file(fname) -> str            # *_metadata.hdf in same folder, explicit raises
def has_nexus_fields(fname, keymap, optional) -> bool
def read_keymap(fname, keymap, optional) -> dict # decodes bytes, squeezes 0-d arrays
def read_nexus_metadata(fname, keymap, optional) -> dict  # locate + read, raw values
```

Each beamline reader:
- owns its `DEFAULT_METADATA_WITHUNITS` and `METADATA_KEYMAPS`,
- calls `read_nexus_metadata(...)`,
- applies its own beam-center derivation (8-ID-I vs 9-ID-D differ; 9-ID-D also computes
  specular), with default-metadata fallback on failure,
- selects its format loader via `get_format_loader(fname)` (9-ID-D now uses the HDF loader
  too — no more direct `sum_frames_parallel` call), and
- sets `stype` ("Transmission" / "Reflection").

`get_reader(beamline, fname)` in `reader/__init__.py` maps `"APS_8IDI"`/`"APS_9IDD"` to the
reader class and raises a clear error on unknown beamlines.
`file_handler.get_handler` delegates to it.

## Bugs & vulnerabilities fixed

| # | Location (old) | Issue | Fix |
|---|----------------|-------|-----|
| B1 | rigaku_handler.read_data | binary file opened in text mode `"r"` | `"rb"` |
| B2 | aps_8idi_reader.__init__ | `return None` on unsupported file → half-built object | raise `ValueError` |
| B3 | base_reader.get_metadata | int→float misses `np.integer`, wrongly catches `bool` | proper numeric check; decode bytes; squeeze 0-d |
| B4 | base_reader.get_scat_with_mask | `np.min` of empty when no positive pixels | if no positive pixels, use floor of 1.0 (log10 → 0) instead of `np.min([])` |
| B5 | APS_8IDI/__init__.py | `__all__` holds classes, not strings | correct (or drop in new layout) |
| B6 | aps_8idi_reader | `getLogger(__file__)` | `getLogger(__name__)` |
| B7 | esrf handler | `logger.warn` deprecated | removed with file; `logger.warning` elsewhere |
| B8 | imm_handler.read_toc | `raise IOError(...)` drops traceback; `err` unused | `raise IOError(...) from err` |
| B9 | find_metadata_same_folder, rigaku3m, rigaku_six | `assert` for input validation (stripped under `-O`) | explicit raises |
| B10 | find_metadata_same_folder | unreachable `elif num_found == 0` after assert | clean control flow |
| B11 | rigaku_3M_handler.get_modules | duplicate unreachable `return` | removed |
| B12 | base_reader.get_center | silent `None` on bad `mode` | raise `ValueError` |
| B13 | get_metadata_from_keymap | raw bytes/0-d arrays propagate | decode/squeeze in `read_keymap` |
| B14 | both readers | redundant unused `scattering_type` key | dropped (`stype` is authoritative) |
| B15 | hdf/imm handlers | leaked file handles via lazy `fhdl`/`__del__` | context-managed reads; explicit close |

## Style normalization

`super().__init__()` everywhere; double quotes; `%`-style logging args; consistent import
grouping; accurate docstrings (fix the endianness and `(value, unit)` tuple-arity comments);
`convert_sparse` defined once and imported.

## Verification — synthetic fixtures + unit tests

New `tests/reader/` (pytest), all fixtures generated in-test (no external data):

- `test_rigaku.py` — synthetic uint64 stream with known (frame, index, count); assert mean
  image vs hand-computed; `convert_sparse` round-trip; 3M module index-offset
  (`>= 1024*1024`) and stitching layout.
- `test_imm.py` — minimal IMM file (header struct + dense and sparse payloads); assert
  mean image vs expected.
- `test_hdf.py` — temp HDF5 with 3-D `/entry/data/data` + NeXus metadata; assert mean,
  metadata read, default fallback when fields absent.
- `test_metadata.py` — `read_keymap`, `has_nexus_fields`, `find_metadata_file` (tmp dir),
  beam-center math for both beamlines.
- `test_dispatch.py` — `get_format_loader`/`get_reader` choose correct types by extension;
  unsupported → `ValueError`.

Equivalence check: for HDF and Rigaku, compute new-vs-old output on the same synthetic
input and assert `new == old / n_frames` (confirms only the intended mean rescaling changed).

Tooling: `pip install -e ".[dev]"` into the conda env first (ruff/mypy are declared but not
currently installed). Run `pytest tests/reader/`, `ruff check src/pysimplemask/reader`,
`mypy` on the reader package.

## Risks & mitigations

- **Mean-vs-sum rescaling** changes HDF/Rigaku absolute intensity. Mitigation: explicit
  user decision (made); only affects log display / threshold presets, not mask geometry.
- **IMM rewrite** must match the old mean output. Mitigation: dedicated synthetic-fixture
  test; compare against the old batch path on the same input.
- **Import churn** from moving modules. Mitigation: changes confined to `reader/` plus the
  delegator in `file_handler.py`; grep confirms no other module imports the moved paths.
- **Timepix** external dependency. Mitigation: lazy import, clear error if missing.
```
