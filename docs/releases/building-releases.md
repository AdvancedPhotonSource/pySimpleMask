# Building GUI Releases

This document describes how to build standalone GUI binaries for pySimpleMask:
a **Windows `.exe`** and a **Linux AppImage**.

## Quick Start

### Local build (Linux)

```bash
# Activate the conda env (or any env with the package installed)
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m pip install pyinstaller

# Build the one-dir bundle
bash packaging/build_gui.sh

# Smoke test
QT_QPA_PLATFORM=offscreen timeout 5 dist/pySimpleMask/pySimpleMask || true
```

### Local build (Windows)

```powershell
pip install pyinstaller
powershell -ExecutionPolicy Bypass -File packaging/build_gui.ps1
.\dist\pySimpleMask.exe
```

### Build an AppImage (Linux)

```bash
/local/MQICHU/envs/l2606_simplemask_refact/bin/python -m pip install pyinstaller
bash packaging/linux/build-appimage.sh v1.0.0
```

This produces `dist/pySimpleMask-v1.0.0-x86_64.AppImage`.

---

## How It Works

The build uses [PyInstaller](https://pyinstaller.org/) to package the GUI into a
standalone binary. The spec file (`pysimplemask.spec`) defines the entry point,
hidden imports, and data files.

**Entry point:** `packaging/entrypoint.py` — a thin wrapper that calls
`pysimplemask.gui.app.main_gui()`. This decouples the build from the internal
module structure.

**Spec file:** `pysimplemask.spec` — the master config. It is platform-aware:
- **Windows:** single-file `.exe` (`onefile`-style, all binaries in the exe)
- **Linux:** one-dir bundle (required for AppImage assembly)

### What gets packaged

- All `pysimplemask.core.*` modules (Qt-free engine)
- All `pysimplemask.gui.*` modules (PySide6 + pyqtgraph)
- `mask.ui` (Qt Designer source) and `logo.svg` (branding)
- Implicit dependencies: numpy, scipy, h5py, scikit-image, matplotlib,
  pyqtgraph, PySide6, tifffile, astropy, imagecodecs

---

## GitHub Actions CI/CD

Pushing a tag triggers the release build automatically:

```bash
git tag v1.2.3
git push origin v1.2.3
```

The workflow (`.github/workflows/build-releases.yml`) does:

1. **Windows:** builds `.exe` on a self-hosted runner, smoke tests with
   `QT_QPA_PLATFORM=offscreen`, uploads as an artifact.
2. **Linux:** builds in a Rocky Linux 9 container, assembles an AppImage,
   smoke tests, uploads as an artifact.
3. **GitHub Release:** creates/updates a GitHub release with both binaries
   as attachments (only on tag pushes, not manual dispatches).

### Manual trigger

You can also run the workflow manually from the GitHub Actions tab
(`workflow_dispatch`). A version tag input lets you name the artifact.

### Self-hosted Windows runner

The Windows job uses `runs-on: [self-hosted, Windows, X64]` because it needs
a machine with a display server for the smoke test. To use GitHub-hosted
runners instead, change this to `windows-latest` — the smoke test still
passes with `QT_QPA_PLATFORM=offscreen`.

---

## Project Structure (Build-Related)

```
pysimplemask.spec              # PyInstaller master spec (platform-aware)
packaging/
├── entrypoint.py              # Thin GUI entry point for PyInstaller
├── icon.ico                   # Windows executable icon (derived from logo.svg)
├── build_gui.sh               # Local Linux build script
├── build_gui.ps1              # Local Windows build script
└── linux/
    ├── build-appimage.sh      # AppImage assembly script
    ├── AppRun                 # AppImage entry point
    ├── pySimpleMask.desktop   # Desktop integration file
```

---

## Troubleshooting

### `ImportError: libGL.so.1: cannot open shared object file`

Install the missing system library. On Rocky/EL:

```bash
sudo dnf install mesa-libGL mesa-libEGL libxkbcommon
```

### `Qt QPA platform plugin "xcb" could not be loaded`

The binary needs Qt's xcb platform plugin. In the AppImage this is bundled;
on a local build, ensure `libxcb*` packages are installed:

```bash
sudo dnf install xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil
```

### PyInstaller can't find `hdf5plugin` cextensions

`hdf5plugin` ships compiled `.so`/`.cpython*.so` files that PyInstaller
should pick up automatically via its hooks. If you see missing-symbol
errors at runtime, ensure the package is installed in the same environment
used for the build:

```bash
pip install hdf5plugin
pyinstaller pysimplemask.spec
```

### The `.exe` is too large (> 500 MB)

This is expected — the binary bundles numpy, scipy, h5py, scikit-image,
matplotlib, PySide6, and pyqtgraph. On first run, PyInstaller extracts
to a temp directory, which adds a few seconds to the startup time.

### Smoke test fails with `QApplication: No such file or directory`

Set the offscreen platform:

```bash
export QT_QPA_PLATFORM=offscreen
dist/pySimpleMask/pySimpleMask
```

---

## Adding Dependencies

If you add a new Python package or data directory:

1. Add the import to `hiddenimports` in `pysimplemask.spec` (PyInstaller
   often misses dynamically imported modules).
2. Add data files to `datas` in the spec.
3. If the dependency has C extensions, test the build — PyInstaller's
   hook database usually handles them, but some (like `hdf5plugin`) may
   need manual `binaries` entries.