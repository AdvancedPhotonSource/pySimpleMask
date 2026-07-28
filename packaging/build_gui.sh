#!/bin/bash
# packaging/build_gui.sh
# Local build script for Linux — produces a PyInstaller one-dir bundle.
#
# Usage (from repo root):
#   bash packaging/build_gui.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== pySimpleMask: Local GUI build (Linux) ==="
echo ""

if ! command -v pyinstaller &> /dev/null; then
    echo "Error: PyInstaller not found. Install it first:"
    echo "  pip install pyinstaller"
    exit 1
fi

rm -rf build dist/pySimpleMask

echo "Running PyInstaller..."
pyinstaller pysimplemask.spec

BUNDLE="dist/pySimpleMask"
if [ -d "$BUNDLE" ]; then
    echo ""
    echo "Build succeeded: $BUNDLE"
    echo "Launch with:"
    echo "  $BUNDLE/pySimpleMask"
    echo ""
    echo "Smoke test (offscreen):"
    echo "  QT_QPA_PLATFORM=offscreen timeout 5 $BUNDLE/pySimpleMask || true"
else
    echo "Build FAILED — output directory not found."
    exit 1
fi