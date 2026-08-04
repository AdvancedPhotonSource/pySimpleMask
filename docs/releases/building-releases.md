# Building GUI Releases

This document describes how to build standalone GUI binaries for pySimpleMask:
a **Windows `.exe`**, a **Linux AppImage**, and a signed/notarized **macOS `.dmg`**.

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

### Local build (macOS, unsigned)

```bash
pip install pyinstaller

# Generate the .icns app icon (macOS-only tooling; must run before pyinstaller)
bash packaging/macos/make_icns.sh

pyinstaller pysimplemask.spec
open dist/pySimpleMask.app
```

This produces an **unsigned** `.app` — Gatekeeper will refuse to open it without
right-click → Open. Signing and notarization only happen in CI (see below), where the
Apple Developer ID certificate and notarization credentials live as GitHub secrets in
the `macos-signing` environment.

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
- **macOS:** one-dir bundle (same shape as Linux) wrapped in an `.app` via `BUNDLE`,
  signed + notarized in CI

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
3. **macOS:** imports the Apple Developer ID certificate, builds a one-dir `.app`
   bundle, deep-codesigns it with hardened runtime, notarizes and staples both the
   `.app` and the final `.dmg`, verifies with Gatekeeper, uploads as an artifact.
4. **GitHub Release:** creates/updates a GitHub release with all three binaries
   as attachments (only on tag pushes, not manual dispatches).

### Manual trigger

You can also run the workflow manually from the GitHub Actions tab
(`workflow_dispatch`). A version tag input lets you name the artifact.

### Windows runner note

The Windows job currently runs on `windows-latest` (GitHub-hosted) rather than a
self-hosted runner — see the `TODO` comment above the `windows-build` job in
`build-releases.yml` for context. This gives a headless smoke test only (no
GPU/display); switch back to `[self-hosted, Windows, X64]` once a self-hosted Windows
runner is registered for this repo.

---

## Project Structure (Build-Related)

```
pysimplemask.spec              # PyInstaller master spec (platform-aware)
packaging/
├── entrypoint.py              # Thin GUI entry point for PyInstaller
├── icon.ico                   # Windows executable icon (derived from logo.svg)
├── icon.png                   # Source icon (256x256), used to generate icon.icns
├── build_gui.sh               # Local Linux build script
├── build_gui.ps1              # Local Windows build script
├── macos/
│   ├── make_icns.sh           # Generates icon.icns from icon.png (macOS-only tools)
│   └── entitlements.plist     # Hardened-runtime entitlements for codesign
└── linux/
    ├── build-appimage.sh      # AppImage assembly script
    ├── AppRun                 # AppImage entry point
    └── pySimpleMask.desktop   # Desktop integration file
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