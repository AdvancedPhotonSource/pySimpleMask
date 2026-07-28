# `--metadata-fname` Override — Design Spec

**Date:** 2026-07-28
**Branch:** mc_dev
**Status:** Approved

## Problem

Metadata (beam center, energy, detector distance, …) is currently discovered
automatically: `read_nexus_metadata` checks the data file itself for the required
NeXus fields, then falls back to a sibling `*_metadata.hdf` file in the same
directory, then to hardcoded placeholder defaults
(`core/reader/metadata.py::read_nexus_metadata`). Users sometimes need to point at a
metadata file that lives elsewhere or has a different name than the sibling-glob
convention expects — for example, reusing one experiment's metadata file against a
rescan, or working with datasets whose sibling file was renamed or misplaced.

This spec adds an explicit `metadata_fname` override: when given and valid, it's used
directly; when omitted or invalid, today's discovery chain runs unchanged.

**Scope for this pass:** CLI (`pysimplemask build --metadata-fname`) and
`SimpleMaskModel.read_data(metadata_fname=...)` for headless/scripted use. **GUI
wiring is explicitly deferred** — no `.ui` or `main_window.py` changes in this pass.

**Readers affected:** APS_8IDI, APS_9IDD, NativeFiles. `XPCSResultReader` reads
metadata from `/xpcs/qmap` inside the result file itself and has no sidecar-file
concept, so it does not support an override — it only needs a signature change to stay
compatible with the shared `prepare_data` call path (see below), with no behavior
change.

## Architecture

### Data flow

`metadata_fname` reaches every reader through the **existing** kwargs pass-through
that already carries `begin_idx`/`num_frames` from the CLI down to the format loader —
no changes to `model.read_data`, `get_handler`, `get_reader`, or any reader
`__init__`:

```
model.read_data(fname, beamline, metadata_fname=..., **kwargs)
  -> self.dset.prepare_data(**kwargs)          # kwargs already forwarded verbatim today
       -> prepare_data pops metadata_fname, calls self.get_metadata(metadata_fname=...)
            -> reader's _get_metadata(metadata_fname=...)
```

### Validation/fallback semantics (`core/reader/metadata.py`)

`read_nexus_metadata` gains a `metadata_fname=None` parameter:

```python
def read_nexus_metadata(fname, keymap, optional_fields=None, metadata_fname=None):
    if metadata_fname and has_nexus_fields(metadata_fname, keymap, optional_fields):
        meta_fname = metadata_fname
    elif has_nexus_fields(fname, keymap, optional_fields):
        meta_fname = fname
    else:
        if metadata_fname:
            logger.warning(
                "metadata_fname %s is missing required fields; "
                "falling back to automatic discovery", metadata_fname,
            )
        meta_fname = find_metadata_file(fname)
        if not has_nexus_fields(meta_fname, keymap, optional_fields):
            raise FileNotFoundError(f"No valid metadata found in {meta_fname}")

    logger.info("using metadata file: %s", meta_fname)
    metadata = read_keymap(meta_fname, keymap, optional_fields)
    metadata["meta_fname"] = meta_fname
    return metadata, meta_fname
```

- Valid override → used directly, self-check and sibling-glob are skipped entirely.
- Invalid override (missing fields, not HDF5, doesn't exist) → warning logged, then
  today's self-then-sibling logic runs unchanged, including its own eventual
  `FileNotFoundError` if nothing valid is found anywhere.
- `metadata_fname=None` → byte-for-byte identical to today's behavior.

`has_nexus_fields` already returns `False` (not an exception) for a nonexistent or
non-HDF5 path, so no extra existence check is needed before calling it.

### Per-file changes

**`aps_8idi.py` / `aps_9idd.py`** — thread `metadata_fname=None` through the three call
sites, symmetric for both modules:

```python
def get_nexus_metadata(fname, metadata_fname=None):
    meta, _meta_fname = read_nexus_metadata(
        fname, METADATA_KEYMAPS, OPTIONAL_FIELDS, metadata_fname=metadata_fname
    )
    ...  # beam-center derivation unchanged

def get_metadata(fname, metadata_fname=None):
    try:
        return get_nexus_metadata(fname, metadata_fname=metadata_fname)
    except Exception:
        ...  # unchanged DEFAULT_METADATA fallback

class APS8IDIReader(FileReader):
    def _get_metadata(self, metadata_fname=None):
        return get_metadata(self.fname, metadata_fname=metadata_fname)
```

(`APS9IDDReader` mirrors this exactly against its own keymap/derivation.)

**`native_files.py`** — validates the override against APS_8IDI's keymap (per prior
decision to reuse the full NeXus keymap rather than a new minimal one) and reuses
`aps_8idi.get_nexus_metadata` directly when valid:

```python
def _get_metadata(self, metadata_fname=None) -> dict:
    if metadata_fname:
        from .aps_8idi import METADATA_KEYMAPS, OPTIONAL_FIELDS, get_nexus_metadata
        from ..metadata import has_nexus_fields
        if has_nexus_fields(metadata_fname, METADATA_KEYMAPS, OPTIONAL_FIELDS):
            return get_nexus_metadata(metadata_fname)
        logger.warning(
            "metadata_fname %s is missing required 8-ID-I NeXus fields; "
            "using placeholder metadata", metadata_fname,
        )
    return get_fake_metadata()
```

Calling `get_nexus_metadata(metadata_fname)` (with no `metadata_fname=` kwarg) is
deliberate: `has_nexus_fields` was already checked above, so `read_nexus_metadata`
takes its "file already has fields" branch immediately — no sibling-glob search
against `metadata_fname`'s directory is ever triggered.

**`xpcs_result.py`** — signature-only change, no behavior change:

```python
def _get_metadata(self, metadata_fname=None):
    ...  # body unchanged; parameter accepted and ignored
```

Required because `prepare_data` (below) now always passes `metadata_fname` to every
reader's `_get_metadata`, including this one.

**`base_reader.py`** — `FileReader.prepare_data` pops `metadata_fname` as a
keyword-only argument and forwards it to `get_metadata`; everything else (frame-range
kwargs to `get_scattering`) is unchanged:

```python
def prepare_data(self, *args, metadata_fname=None, **kwargs):
    self.metadata = self.get_metadata(metadata_fname=metadata_fname)
    self.scat = self.get_scattering(*args, **kwargs).astype(np.float32)
    ...  # unchanged
```

**`cli.py`** — new argument in `_add_build_args`'s "data loading" group, and threaded
into the pipeline call plus the report for provenance:

```python
grp_load.add_argument(
    "--metadata-fname", default=None, metavar="FILE",
    help="Explicit metadata HDF5 file. Falls back to automatic discovery if omitted or invalid.",
)
```

```python
ok = m.read_data(
    args.dataset,
    beamline=args.beamline,
    begin_idx=args.begin_idx,
    num_frames=args.num_frames,
    metadata_fname=args.metadata_fname,
)
```

`report_params` in `_run_build_qmap` gains a `"metadata_fname": args.metadata_fname`
entry alongside the other loading knobs already recorded there.

Note: the `build` subcommand's `--beamline` choices are currently `["APS_8IDI",
"APS_9IDD"]` (NativeFiles isn't a CLI-exposed beamline). `--metadata-fname` is still
useful for those two beamlines from the CLI; NativeFiles' override support is reachable
today only via `SimpleMaskModel.read_data(beamline="NativeFiles", metadata_fname=...)`
in a script, consistent with NativeFiles not being CLI-selectable at all currently.

## GUI Impact

None in this pass. No `.ui` changes, no `main_window.py` changes. Deferred to a future
spec if/when a GUI entry point is wanted.

## Testing

`tests/core/reader/test_metadata.py`:

| Test | What it checks |
|------|----------------|
| Valid override used directly | `read_nexus_metadata(fname, keymap, metadata_fname=valid_path)` returns `meta_fname == valid_path`, no sibling glob triggered even if none exists |
| Invalid override falls back | `metadata_fname` missing required fields → warning logged, falls through to self/sibling discovery (existing behavior) |
| `metadata_fname=None` unchanged | Existing self-then-sibling tests continue to pass with the new parameter defaulted |

Beamline-level (`aps_8idi.py`/`aps_9idd.py`):

| Test | What it checks |
|------|----------------|
| `get_metadata`/`get_nexus_metadata` thread `metadata_fname` | Override reaches `read_nexus_metadata`; derived fields (`beam_center_x/y`, `pixel_size`) computed from the override file's raw values |

`native_files.py` (new tests):

| Test | What it checks |
|------|----------------|
| Valid 8-ID-I-shaped override | `NativeFilesReader(fname)._get_metadata(metadata_fname=valid)` returns real (non-placeholder) values |
| Invalid override | Missing required fields → warning logged, `get_fake_metadata()` returned |
| `metadata_fname=None` | Unchanged — `get_fake_metadata()` |

`xpcs_result.py` (regression check):

| Test | What it checks |
|------|----------------|
| `prepare_data()` still works | No crash now that `prepare_data` always passes `metadata_fname=None` through to `_get_metadata` |

`tests/cli/test_subcommands.py` (new test, mirrors the existing `--no-find-center`
pattern):

| Test | What it checks |
|------|----------------|
| `--metadata-fname` flows through | `pysimplemask build FILE --metadata-fname META` → `_run_build_qmap`'s `read_data` call receives `metadata_fname="META"` |

## Files Changed

| File | Change |
|------|--------|
| `src/pysimplemask/core/reader/metadata.py` | `read_nexus_metadata` gains `metadata_fname=None` override-then-fallback logic |
| `src/pysimplemask/core/reader/beamlines/aps_8idi.py` | Thread `metadata_fname` through `get_nexus_metadata`, `get_metadata`, `APS8IDIReader._get_metadata` |
| `src/pysimplemask/core/reader/beamlines/aps_9idd.py` | Same threading, mirrored |
| `src/pysimplemask/core/reader/beamlines/native_files.py` | `_get_metadata` validates override against APS_8IDI's keymap, reuses `get_nexus_metadata` |
| `src/pysimplemask/core/reader/beamlines/xpcs_result.py` | `_get_metadata(self, metadata_fname=None)` — signature-only, no behavior change |
| `src/pysimplemask/core/reader/base_reader.py` | `prepare_data` accepts and forwards `metadata_fname` |
| `src/pysimplemask/cli.py` | New `--metadata-fname` arg; forwarded to `read_data`; recorded in `report_params` |
| `tests/core/reader/test_metadata.py` | New override/fallback/none-regression tests |
| `tests/cli/test_subcommands.py` | New `--metadata-fname` pass-through test |

No changes to `model.py`, `file_handler.py`, `reader/__init__.py`, `get_handler`,
`get_reader`, or any reader `__init__`. No GUI or `.ui` changes.
