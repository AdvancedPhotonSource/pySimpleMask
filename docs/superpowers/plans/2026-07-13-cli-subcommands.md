# CLI Subcommands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `pysimplemask` a unified entry point with four subcommands — `gui` (default), `web`, `build`, `combine` — while preserving all existing `pysimplemask-*` scripts unchanged.

**Architecture:** Two files change. `server.py` gains a `run_web(host, port, path, debug)` helper so `cli.py` can start the web server without duplicating startup logic. `cli.py` adds `_add_build_args(parser)` (shared argument definitions) and a new `main()` dispatcher with argparse subparsers; the old standalone functions (`build_qmap`, `combine_qmaps`) stay untouched for backward compatibility.

**Tech Stack:** Python stdlib `argparse`, existing `pysimplemask.cli`, `pysimplemask.web.server`

## Global Constraints

- Environment: `/local/MQICHU/envs/l2606_simplemask_refact/bin/`
- All existing `pysimplemask-*` console scripts must keep working unchanged
- `pysimplemask` with no subcommand must launch the GUI (same as before)
- `pysimplemask --path /some/dir` (top-level `--path`) must still launch the GUI at that path (backward compat)
- `_build_qmap_args(argv)` public signature must not change — existing tests call it directly
- `_run_build_qmap(args)` public signature must not change — existing tests call it directly
- Ruff must pass clean

---

## File Map

| Status | File | Change |
|--------|------|--------|
| Modify | `src/pysimplemask/web/server.py` | Extract `run_web(host, port, path, debug)` from `main_web()` |
| Modify | `src/pysimplemask/cli.py` | Add `_add_build_args(parser)` + new `main()` dispatcher |
| Create | `tests/cli/test_subcommands.py` | New tests for subcommand dispatch |

---

## Task 1: Extract `run_web()` from `server.py`

**Files:**
- Modify: `src/pysimplemask/web/server.py`

**Interfaces:**
- Produces: `run_web(host: str = "127.0.0.1", port: int = 8050, path: str | None = None, debug: bool = False) -> None`
  — starts the Dash app; called by both `main_web()` and the `cli.py` web subcommand
- `main_web()` signature unchanged (still the `pysimplemask-web` entry point)

- [ ] **Step 1: Write the failing test**

Create `tests/cli/test_subcommands.py` (just the import test for now):

```python
"""Tests for pysimplemask subcommand dispatcher."""


def test_run_web_importable():
    from pysimplemask.web.server import run_web  # noqa: F401
    assert callable(run_web)
```

- [ ] **Step 2: Run failing test**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/cli/test_subcommands.py::test_run_web_importable -v
```

Expected: FAIL — `ImportError: cannot import name 'run_web'`.

- [ ] **Step 3: Refactor `server.py`**

Replace the existing `main_web()` function with:

```python
def run_web(
    host: str = "127.0.0.1",
    port: int = 8050,
    path: str | None = None,
    debug: bool = False,
) -> None:
    """Start the Dash web server.

    Called by ``main_web()`` and by the ``pysimplemask web`` subcommand.
    """
    from pysimplemask.web import layout as _layout

    try:
        from pysimplemask.web import callbacks as _callbacks  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pysimplemask.web.callbacks not found. "
            "Ensure the web package is fully installed."
        ) from exc
    try:
        from pysimplemask.web import mask_callbacks as _mask_callbacks  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pysimplemask.web.mask_callbacks not found. "
            "Ensure the web package is fully installed."
        ) from exc
    try:
        from pysimplemask.web import partition_callbacks as _partition_callbacks  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "pysimplemask.web.partition_callbacks not found. "
            "Ensure the web package is fully installed."
        ) from exc

    app.layout = _layout.build_layout(initial_path=path or "")
    print(f"pySimpleMask web interface running at http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)


def main_web() -> None:
    """Console-script entry point: ``pysimplemask-web``."""
    parser = argparse.ArgumentParser(
        prog="pysimplemask-web",
        description="Launch the pySimpleMask web interface.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="Port number (default: 8050)",
    )
    parser.add_argument(
        "--path", default=None,
        help="Pre-populate the file-path field with this path",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Dash debug/hot-reload mode",
    )
    args = parser.parse_args()
    run_web(host=args.host, port=args.port, path=args.path, debug=args.debug)
```

- [ ] **Step 4: Run the test**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/cli/test_subcommands.py::test_run_web_importable -v
```

Expected: PASS.

- [ ] **Step 5: Run full suite + ruff**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check src/pysimplemask/web/server.py
```

Expected: all tests pass; ruff clean.

- [ ] **Step 6: Commit**

```bash
git add src/pysimplemask/web/server.py tests/cli/test_subcommands.py
git commit -m "refactor(web): extract run_web() helper from main_web()"
```

---

## Task 2: Add subcommand dispatcher to `cli.py`

**Files:**
- Modify: `src/pysimplemask/cli.py`
- Modify: `tests/cli/test_subcommands.py`

**Interfaces:**
- Consumes: `run_web(host, port, path, debug)` from Task 1
- Produces: `_add_build_args(parser) -> None` — adds all build-related arguments to any ArgumentParser
- `main()` — unified entry point; replaces the current GUI-only `main()`; dispatches on subcommand
- `_build_qmap_args(argv=None)` — signature unchanged; now calls `_add_build_args()` internally

**Key constraint:** `_build_qmap_args(argv)` and `_run_build_qmap(args)` must keep their current signatures exactly — existing tests depend on them.

- [ ] **Step 1: Write the failing tests**

Append to `tests/cli/test_subcommands.py`:

```python
import argparse
import os
import sys
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# _add_build_args
# ---------------------------------------------------------------------------


def test_add_build_args_produces_same_defaults_as_build_qmap_args():
    """_add_build_args adds identical arguments to any parser."""
    from pysimplemask.cli import _add_build_args, _build_qmap_args
    import h5py, numpy as np, tempfile, pathlib

    # Create a minimal HDF5 so _build_qmap_args doesn't fail on the file check
    with tempfile.TemporaryDirectory() as d:
        p = str(pathlib.Path(d) / "scan.h5")
        with h5py.File(p, "w") as h:
            h["/entry/data/data"] = np.zeros((2, 4, 4), dtype=np.uint16)

        # Reference: old interface
        ref = _build_qmap_args([p])

        # New interface: add args to a fresh parser
        new_parser = argparse.ArgumentParser()
        _add_build_args(new_parser)
        new_args = new_parser.parse_args([p])

    assert new_args.beamline == ref.beamline
    assert new_args.dq_num == ref.dq_num
    assert new_args.mode == ref.mode
    assert new_args.output_qmap == ref.output_qmap


# ---------------------------------------------------------------------------
# main() subcommand dispatch
# ---------------------------------------------------------------------------


def test_main_no_subcommand_launches_gui(monkeypatch):
    """pysimplemask with no subcommand calls main_gui."""
    monkeypatch.setattr(sys, "argv", ["pysimplemask"])
    mock_gui = MagicMock(return_value=0)
    with patch("pysimplemask.gui.app.main_gui", mock_gui):
        from pysimplemask import cli
        try:
            cli.main()
        except SystemExit:
            pass
    mock_gui.assert_called_once()


def test_main_gui_subcommand_passes_path(monkeypatch, tmp_path):
    """pysimplemask gui --path /tmp calls main_gui with that path."""
    monkeypatch.setattr(sys, "argv", ["pysimplemask", "gui", "--path", str(tmp_path)])
    mock_gui = MagicMock(return_value=0)
    with patch("pysimplemask.gui.app.main_gui", mock_gui):
        from pysimplemask import cli
        try:
            cli.main()
        except SystemExit:
            pass
    mock_gui.assert_called_once_with(str(tmp_path))


def test_main_web_subcommand_calls_run_web(monkeypatch):
    """pysimplemask web --host 0.0.0.0 --port 9000 calls run_web with those args."""
    monkeypatch.setattr(
        sys, "argv",
        ["pysimplemask", "web", "--host", "0.0.0.0", "--port", "9000"],
    )
    mock_run = MagicMock()
    with patch("pysimplemask.web.server.run_web", mock_run):
        from pysimplemask import cli
        cli.main()
    mock_run.assert_called_once_with(
        host="0.0.0.0", port=9000, path=None, debug=False
    )


def test_main_combine_subcommand_calls_combine(monkeypatch, tmp_path):
    """pysimplemask combine a.h5 b.h5 out.h5 calls combine_qmap_files."""
    a = str(tmp_path / "a.h5")
    b = str(tmp_path / "b.h5")
    o = str(tmp_path / "out.h5")
    monkeypatch.setattr(sys, "argv", ["pysimplemask", "combine", a, b, o])
    mock_combine = MagicMock()
    with patch("pysimplemask.core.partition.combine_qmap_files", mock_combine):
        from pysimplemask import cli
        cli.main()
    mock_combine.assert_called_once_with(a, b, o)


def test_main_build_subcommand_calls_run_build(monkeypatch, tmp_path):
    """pysimplemask build FILE --no-find-center calls _run_build_qmap."""
    import h5py, numpy as np
    p = str(tmp_path / "scan.h5")
    with h5py.File(p, "w") as h:
        h["/entry/data/data"] = np.zeros((2, 4, 4), dtype=np.uint16)
    monkeypatch.setattr(
        sys, "argv", ["pysimplemask", "build", p, "--no-find-center"]
    )
    mock_run = MagicMock()
    with patch("pysimplemask.cli._run_build_qmap", mock_run):
        from pysimplemask import cli
        cli.main()
    mock_run.assert_called_once()
    call_args = mock_run.call_args[0][0]
    assert call_args.dataset == p
    assert call_args.no_find_center is True


def test_build_qmap_args_still_works_directly():
    """_build_qmap_args([...]) still works — no regression on existing interface."""
    import h5py, numpy as np, tempfile, pathlib
    with tempfile.TemporaryDirectory() as d:
        p = str(pathlib.Path(d) / "scan.h5")
        with h5py.File(p, "w") as h:
            h["/entry/data/data"] = np.zeros((2, 4, 4), dtype=np.uint16)
        from pysimplemask.cli import _build_qmap_args
        args = _build_qmap_args([p])
    assert args.dataset == p
    assert args.mode == "q-phi"
```

- [ ] **Step 2: Run failing tests**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/cli/test_subcommands.py -v
```

Expected: `test_run_web_importable` PASS (from Task 1); all new tests FAIL — `_add_build_args` not found, `main()` is still the old GUI launcher.

- [ ] **Step 3: Rewrite `cli.py`**

Replace the contents of `src/pysimplemask/cli.py` with:

```python
"""Console scripts for pysimplemask."""

import argparse
import logging
import os
import sys

from pysimplemask import __version__
from pysimplemask.core.partition import combine_qmap_files


# ---------------------------------------------------------------------------
# Shared build-qmap argument definitions
# ---------------------------------------------------------------------------


def _add_build_args(parser: argparse.ArgumentParser) -> None:
    """Add all build-qmap arguments to *parser* (standalone or subparser)."""

    # positional
    parser.add_argument(
        "dataset",
        help="Path to the raw scattering file (.hdf, .h5, .imm, .bin, …)",
    )

    # data loading
    grp_load = parser.add_argument_group("data loading")
    grp_load.add_argument(
        "--beamline",
        default="APS_8IDI",
        choices=["APS_8IDI", "APS_9IDD"],
        help="Beamline reader.",
    )
    grp_load.add_argument(
        "--begin-idx", type=int, default=0, metavar="N",
        help="First frame index to include.",
    )
    grp_load.add_argument(
        "--num-frames", type=int, default=-1, metavar="N",
        help="Frames to average. 0=all, -1=representative subset.",
    )

    # beam center
    grp_cen = parser.add_argument_group("beam center")
    grp_cen.add_argument(
        "--no-find-center", action="store_true",
        help="Skip goto_max + find_center; use metadata center as-is.",
    )
    grp_cen.add_argument(
        "--max-radius", type=int, default=384, metavar="N",
        help="Crop half-size (px) passed to find_center.",
    )
    grp_cen.add_argument(
        "--beamstop-diameter", type=int, default=30, metavar="N",
        help="Diameter (px) of the circular beamstop mask after find_center. 0 to disable.",
    )

    # mask
    grp_mask = parser.add_argument_group("mask")
    grp_mask.add_argument(
        "--blemish", default=None, metavar="FILE",
        help="Blemish/bad-pixel file (.tif or .h5).",
    )
    grp_mask.add_argument(
        "--blemish-key", default="/qmap/mask", metavar="KEY",
        help="HDF5 dataset path inside the blemish file.",
    )
    grp_mask.add_argument(
        "--threshold-high", type=float, default=None, metavar="VAL",
        help="Mask pixels with intensity >= VAL (raw counts).",
    )
    grp_mask.add_argument(
        "--param-constraint",
        action="append", default=[], metavar="MAPNAME:LOGIC:VBEG:VEND",
        dest="param_constraints",
        help=(
            "Mask pixels by geometry map range. Repeatable. "
            "Format: MAPNAME:LOGIC:VBEG:VEND, e.g. q:AND:0.01:0.1. "
            "LOGIC is AND or OR."
        ),
    )

    # partition
    grp_part = parser.add_argument_group("partition")
    grp_part.add_argument(
        "--mode", default="q-phi", choices=["q-phi", "x-y", "eq-ephi"],
        help="Partition axes.",
    )
    grp_part.add_argument("--dq-num", type=int, default=36, metavar="N",
                          help="Dynamic q bins.")
    grp_part.add_argument("--sq-num", type=int, default=360, metavar="N",
                          help="Static q bins.")
    grp_part.add_argument("--dp-num", type=int, default=1, metavar="N",
                          help="Dynamic phi bins.")
    grp_part.add_argument("--sp-num", type=int, default=1, metavar="N",
                          help="Static phi bins.")
    grp_part.add_argument("--phi-offset", type=float, default=0.0, metavar="DEG",
                          help="Phi axis offset in degrees.")
    grp_part.add_argument("--symmetry-fold", type=int, default=1, metavar="N",
                          help="Rotational symmetry fold.")
    grp_part.add_argument(
        "--style", default="linear", choices=["linear", "logarithmic"],
        help="Bin spacing style.",
    )

    # output
    grp_out = parser.add_argument_group("output")
    grp_out.add_argument(
        "--output-qmap", default="qmap.hdf", metavar="FILE",
        help="Output qmap HDF5 path.",
    )
    grp_out.add_argument(
        "--output-mask", default="mask.tif", metavar="FILE",
        help="Output mask TIFF path. Pass empty string to skip.",
    )
    grp_out.add_argument(
        "--report", default=None, metavar="FILE",
        help=(
            "Write a one-page PDF summary to FILE. "
            "Default: same stem as --output-qmap with .pdf extension."
        ),
    )

    # logging (shared by both standalone and subcommand)
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable DEBUG-level logging.",
    )


# ---------------------------------------------------------------------------
# Unified entry point  (pysimplemask)
# ---------------------------------------------------------------------------


def main() -> None:
    """Unified pysimplemask entry point.

    Subcommands: gui (default), web, build, combine.
    Running ``pysimplemask`` with no subcommand launches the GUI.
    ``pysimplemask --path DIR`` is a backward-compatible shortcut for
    ``pysimplemask gui --path DIR``.
    """
    parser = argparse.ArgumentParser(
        prog="pysimplemask",
        description=(
            "pySimpleMask: mask and q-partition maps for SAXS/WAXS/XPCS data reduction."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"pySimpleMask {__version__}"
    )
    # Top-level --path for backward compatibility (pysimplemask --path DIR)
    parser.add_argument(
        "--path", "-p", default=None, metavar="DIR",
        help="GUI working directory (shortcut for 'gui --path DIR').",
    )

    subparsers = parser.add_subparsers(dest="subcommand")

    # ── gui ──────────────────────────────────────────────────────────────────
    p_gui = subparsers.add_parser(
        "gui", help="Launch the PySide6 GUI (default when no subcommand given)."
    )
    p_gui.add_argument(
        "--path", "-p", default=None, metavar="DIR",
        help="Starting directory (defaults to cwd).",
    )

    # ── web ──────────────────────────────────────────────────────────────────
    p_web = subparsers.add_parser(
        "web", help="Launch the Dash web interface."
    )
    p_web.add_argument("--host", default="127.0.0.1",
                       help="Bind address (default: 127.0.0.1).")
    p_web.add_argument("--port", type=int, default=8050,
                       help="Port number (default: 8050).")
    p_web.add_argument("--path", default=None,
                       help="Pre-populate the file-path field.")
    p_web.add_argument("--debug", action="store_true",
                       help="Enable Dash debug/hot-reload mode.")

    # ── build ─────────────────────────────────────────────────────────────────
    p_build = subparsers.add_parser(
        "build",
        help="Headless build-qmap pipeline (load → center → mask → partition → save).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_build_args(p_build)

    # ── combine ───────────────────────────────────────────────────────────────
    p_combine = subparsers.add_parser(
        "combine", help="Combine two qmap HDF5 files into one."
    )
    p_combine.add_argument("qmap_file1", help="Path to the first qmap HDF5 file.")
    p_combine.add_argument("qmap_file2", help="Path to the second qmap HDF5 file.")
    p_combine.add_argument("output_file", help="Path for the combined output file.")
    p_combine.add_argument("-v", "--verbose", action="store_true",
                           help="Enable DEBUG-level logging.")

    args = parser.parse_args()

    # ── dispatch ──────────────────────────────────────────────────────────────
    if args.subcommand is None or args.subcommand == "gui":
        path = getattr(args, "path", None) or os.getcwd()
        from pysimplemask.gui.app import main_gui
        sys.exit(main_gui(path))

    elif args.subcommand == "web":
        from pysimplemask.web.server import run_web
        run_web(host=args.host, port=args.port, path=args.path, debug=args.debug)

    elif args.subcommand == "build":
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        try:
            _run_build_qmap(args)
        except RuntimeError as exc:
            logging.error("%s", exc)
            sys.exit(1)

    elif args.subcommand == "combine":
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        combine_qmap_files(args.qmap_file1, args.qmap_file2, args.output_file)


# ---------------------------------------------------------------------------
# Standalone entry points (pysimplemask-* scripts — backward compat)
# ---------------------------------------------------------------------------


def combine_qmaps() -> None:
    """CLI entry point: combine two qmap HDF5 files into one."""
    parser = argparse.ArgumentParser(
        prog="pysimplemask-combine-qmaps",
        description="Combine two pySimpleMask qmap HDF5 files into a single output file.",
    )
    parser.add_argument("qmap_file1", help="Path to the first qmap HDF5 file.")
    parser.add_argument("qmap_file2", help="Path to the second qmap HDF5 file.")
    parser.add_argument("output_file", help="Path for the combined output qmap HDF5 file.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable DEBUG-level logging.")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    combine_qmap_files(args.qmap_file1, args.qmap_file2, args.output_file)


def _build_qmap_args(argv=None) -> argparse.Namespace:
    """Parse arguments for build-qmap; returns a Namespace. Testable without sys.argv."""
    parser = argparse.ArgumentParser(
        prog="pysimplemask-build-qmap",
        description=(
            "Build a q-partition map (qmap) from a raw scattering file. "
            "Replicates the GUI workflow: load → center → mask → partition → save."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_build_args(parser)
    return parser.parse_args(argv)


_ANGLE_MAPS = {"phi", "chi", "alpha"}


def _parse_param_constraints(raw):
    """Parse repeated --param-constraint strings into constraint tuples."""
    constraints = []
    for token in raw:
        parts = token.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid --param-constraint {token!r}. "
                "Expected MAPNAME:LOGIC:VBEG:VEND (e.g. q:AND:0.01:0.1)"
            )
        xmap_name, logic, vbeg_s, vend_s = parts
        logic = logic.upper()
        if logic not in ("AND", "OR"):
            raise ValueError(
                f"Invalid logic {logic!r} in --param-constraint {token!r}. Use AND or OR."
            )
        try:
            vbeg, vend = float(vbeg_s), float(vend_s)
        except ValueError:
            raise ValueError(
                f"VBEG and VEND must be numbers in --param-constraint {token!r}."
            )
        unit = "deg" if xmap_name in _ANGLE_MAPS else xmap_name
        constraints.append((xmap_name, logic, unit, vbeg, vend))
    return constraints


def _run_build_qmap(args) -> None:
    """Execute the qmap-build pipeline. Separated from argument parsing for testability."""
    from pysimplemask.core import SimpleMaskModel

    m = SimpleMaskModel()

    # 1. Load
    ok = m.read_data(
        args.dataset,
        beamline=args.beamline,
        begin_idx=args.begin_idx,
        num_frames=args.num_frames,
    )
    if not ok:
        raise RuntimeError(f"Failed to load dataset: {args.dataset}")
    logging.info("Loaded %s  shape=%s", args.dataset, m.shape)

    # 2. Beam center
    if not args.no_find_center:
        center_vh = m.goto_max()
        logging.info("goto_max center: row=%.1f col=%.1f", center_vh[0], center_vh[1])
        refined_vh = m.find_center(
            max_radius=args.max_radius,
            beamstop_diameter=args.beamstop_diameter,
        )
        logging.info(
            "find_center result: row=%.1f col=%.1f", refined_vh[0], refined_vh[1]
        )
        m.dset.set_center_vh(refined_vh)
        m.update_parameters()

    # 3. Mask
    if args.blemish:
        m.mask_evaluate("mask_blemish", fname=args.blemish, key=args.blemish_key)
        m.mask_apply("mask_blemish")
        logging.info("Applied blemish: %s", args.blemish)

    if args.threshold_high is not None:
        m.mask_evaluate(
            "mask_threshold",
            low=0, high=args.threshold_high,
            low_enable=False, high_enable=True,
        )
        m.mask_apply("mask_threshold")
        logging.info("Applied threshold-high: %s", args.threshold_high)

    if args.param_constraints:
        constraints = _parse_param_constraints(args.param_constraints)
        m.mask_evaluate("mask_parameter", constraints=constraints)
        m.mask_apply("mask_parameter")
        logging.info("Applied param constraints: %s", args.param_constraints)

    bad = int(m.mask.size - m.mask.sum())
    logging.info("Final mask: %d pixels masked (%.2f%%)", bad, bad / m.mask.size * 100)

    # 4. Partition
    m.compute_partition(
        mode=args.mode,
        dq_num=args.dq_num,
        sq_num=args.sq_num,
        dp_num=args.dp_num,
        sp_num=args.sp_num,
        phi_offset=args.phi_offset,
        symmetry_fold=args.symmetry_fold,
        style=args.style,
    )
    logging.info("Partition computed (mode=%s)", args.mode)

    # 5. Save
    m.save_partition(args.output_qmap)
    logging.info("Saved qmap: %s", args.output_qmap)

    if args.output_mask:
        m.save_mask(args.output_mask)
        logging.info("Saved mask: %s", args.output_mask)

    # 6. Report
    report_path = args.report
    if report_path is None:
        report_path = os.path.splitext(args.output_qmap)[0] + ".pdf"
    if report_path:
        from pysimplemask.core.report import generate_report
        report_params = {
            "beamline": args.beamline,
            "begin_idx": args.begin_idx,
            "num_frames": args.num_frames,
            "find_center": not args.no_find_center,
            "max_radius": args.max_radius,
            "beamstop_diameter": args.beamstop_diameter,
            "blemish": args.blemish,
            "threshold_high": args.threshold_high,
            "param_constraints": args.param_constraints or None,
            "mode": args.mode,
            "dq_num": args.dq_num,
            "sq_num": args.sq_num,
            "dp_num": args.dp_num,
            "sp_num": args.sp_num,
            "phi_offset": args.phi_offset,
            "symmetry_fold": args.symmetry_fold,
            "style": args.style,
        }
        generate_report(m, report_path, params=report_params)


def build_qmap() -> None:
    """CLI entry point: build a qmap from a raw scattering file."""
    args = _build_qmap_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        _run_build_qmap(args)
    except RuntimeError as exc:
        logging.error("%s", exc)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
```

- [ ] **Step 4: Run the new tests**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/cli/test_subcommands.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 5: Run existing build-qmap tests to confirm no regression**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/cli/test_build_qmap.py -v
```

Expected: all pass (they call `_build_qmap_args` and `_run_build_qmap` directly, which are unchanged).

- [ ] **Step 6: Run full suite + ruff**

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/pytest tests/ -q
/local/MQICHU/envs/l2606_simplemask_refact/bin/ruff check src/pysimplemask/cli.py tests/cli/test_subcommands.py
```

Expected: all tests pass; ruff clean.

- [ ] **Step 7: Commit**

```bash
git add src/pysimplemask/cli.py tests/cli/test_subcommands.py
git commit -m "feat(cli): add subcommand dispatcher — gui (default), web, build, combine"
```
