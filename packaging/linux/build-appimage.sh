#!/bin/bash
# packaging/linux/build-appimage.sh
# Build a Linux AppImage for pySimpleMask.
#
# Usage (from repo root):
#   bash packaging/linux/build-appimage.sh <version-tag>
#
# Prerequisites:
#   - PyInstaller installed in the active Python environment
#   - curl (to download appimagetool)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
REPO_ROOT="$(pwd)"
VERSION="${1:?Usage: build-appimage.sh <version-tag>}"

rm -rf build dist/pySimpleMask dist/AppDir

# Prefer whatever "python" resolves to in the active environment (local dev:
# the activated venv/conda env with PyInstaller installed). Fall back to
# python3.11 before the bare "python3" system interpreter: rockylinux:9 ships
# its own /usr/bin/python3 (3.9) for OS tooling, which does NOT have
# PyInstaller installed by the CI workflow (that targets python3.11
# explicitly), so picking it up here would silently build with the wrong
# interpreter.
PYTHON="${PYTHON:-$(command -v python || command -v python3.11 || command -v python3)}"

# Build the one-dir bundle with PyInstaller
"$PYTHON" -m PyInstaller pysimplemask.spec

# --- Assemble AppDir ---
mkdir -p dist/AppDir/usr/bin
cp -r dist/pySimpleMask dist/AppDir/usr/bin/pySimpleMask
install -m 755 packaging/linux/AppRun dist/AppDir/AppRun
cp packaging/linux/pySimpleMask.desktop dist/AppDir/pySimpleMask.desktop

# Use the SVG logo as a PNG fallback, or provide a real PNG later.
# For now, copy the SVG renamed (some desktop environments handle it).
if [ -f "src/pysimplemask/resources/logo.svg" ]; then
    cp src/pysimplemask/resources/logo.svg dist/AppDir/pySimpleMask.svg
fi

# --- Download appimagetool ---
curl -sL -o dist/appimagetool \
  https://github.com/AppImage/appimagetool/releases/download/continuous/appimagetool-x86_64.AppImage
chmod +x dist/appimagetool

# --- Build AppImage ---
# appimagetool's --appimage-extract-and-run chdirs during self-extraction, so
# relative path arguments resolve against the wrong directory. Use absolute
# paths.
dist/appimagetool --appimage-extract-and-run \
  "$REPO_ROOT/dist/AppDir" \
  "$REPO_ROOT/dist/pySimpleMask-${VERSION}-x86_64.AppImage"

echo "AppImage built: dist/pySimpleMask-${VERSION}-x86_64.AppImage"