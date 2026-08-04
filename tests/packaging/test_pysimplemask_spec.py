"""Exercises pysimplemask.spec's per-platform branches without needing a real
PyInstaller install or a macOS/Windows machine.

PyInstaller runs .spec files with exec(), injecting SPECPATH and the
Analysis/PYZ/EXE/BUNDLE/COLLECT builder names into the execution namespace.
We reproduce that here with recording stand-ins so the branch logic (which
builder gets which name/icon/bundle id) has real test coverage instead of
only being checkable by actually building on each target OS.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_SOURCE = (REPO_ROOT / "pysimplemask.spec").read_text()


class _Recorder:
    """Stand-in for a PyInstaller builder class; records every call."""

    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return object()


class _AnalysisStub:
    """Stand-in for PyInstaller's Analysis result.

    The spec accesses a.pure (for PYZ) and a.scripts/a.binaries/a.datas (for
    EXE/COLLECT) across its three platform branches, so all four must exist.
    """

    def __init__(self, *args, **kwargs):
        self.pure = []
        self.scripts = []
        self.binaries = []
        self.datas = []


def _run_spec(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    exe_recorder = _Recorder()
    bundle_recorder = _Recorder()
    collect_recorder = _Recorder()
    namespace = {
        "SPECPATH": str(REPO_ROOT),
        "Analysis": _AnalysisStub,
        "PYZ": lambda *a, **k: object(),
        "EXE": exe_recorder,
        "BUNDLE": bundle_recorder,
        "COLLECT": collect_recorder,
    }
    exec(compile(SPEC_SOURCE, "pysimplemask.spec", "exec"), namespace)
    return exe_recorder, bundle_recorder, collect_recorder


def test_macos_branch_wraps_exe_in_app_bundle(monkeypatch):
    exe_rec, bundle_rec, collect_rec = _run_spec(monkeypatch, "darwin")

    assert len(exe_rec.calls) == 1
    assert len(bundle_rec.calls) == 1
    assert len(collect_rec.calls) == 0

    _, exe_kwargs = exe_rec.calls[0]
    assert exe_kwargs["codesign_identity"] is None

    _, bundle_kwargs = bundle_rec.calls[0]
    assert bundle_kwargs["name"] == "pySimpleMask.app"
    assert bundle_kwargs["bundle_identifier"] == "gov.anl.aps.pysimplemask"
    assert bundle_kwargs["icon"].endswith("packaging/macos/icon.icns")


def test_windows_branch_unchanged_no_bundle(monkeypatch):
    exe_rec, bundle_rec, collect_rec = _run_spec(monkeypatch, "win32")

    assert len(exe_rec.calls) == 1
    assert len(bundle_rec.calls) == 0
    _, exe_kwargs = exe_rec.calls[0]
    assert exe_kwargs["icon"].endswith("packaging/icon.ico")


def test_linux_branch_unchanged_uses_collect(monkeypatch):
    exe_rec, bundle_rec, collect_rec = _run_spec(monkeypatch, "linux")

    assert len(exe_rec.calls) == 1
    assert len(collect_rec.calls) == 1
    assert len(bundle_rec.calls) == 0
