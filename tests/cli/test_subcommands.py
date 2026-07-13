"""Tests for pysimplemask subcommand dispatcher."""

import argparse
import sys
from unittest.mock import patch, MagicMock


def test_run_web_importable():
    from pysimplemask.web.server import run_web  # noqa: F401
    assert callable(run_web)


# ---------------------------------------------------------------------------
# _add_build_args
# ---------------------------------------------------------------------------


def test_add_build_args_produces_same_defaults_as_build_qmap_args():
    """_add_build_args adds identical arguments to any parser."""
    from pysimplemask.cli import _add_build_args, _build_qmap_args
    import h5py
    import numpy as np
    import tempfile
    import pathlib

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
    import h5py
    import numpy as np
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
    import h5py
    import numpy as np
    import tempfile
    import pathlib
    with tempfile.TemporaryDirectory() as d:
        p = str(pathlib.Path(d) / "scan.h5")
        with h5py.File(p, "w") as h:
            h["/entry/data/data"] = np.zeros((2, 4, 4), dtype=np.uint16)
        from pysimplemask.cli import _build_qmap_args
        args = _build_qmap_args([p])
    assert args.dataset == p
    assert args.mode == "q-phi"
