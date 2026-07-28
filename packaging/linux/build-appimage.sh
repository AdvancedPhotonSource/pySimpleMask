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
VERSION="${1:?Usage: build-appimage.sh <version-tag>}"

rm -rf build dist/pySimpleMask dist/AppDir

# Build the one-dir bundle with PyInstaller
python -m PyInstaller pysimplemask.spec

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
dist/appimagetool --appimage-extract-and-run \
  dist/AppDir \
  "dist/pySimpleMask-${VERSION}-x86_64.AppImage"

echo "AppImage built: dist/pySimpleMask-${VERSION}-x86_64.AppImage"