#!/usr/bin/env bash
# Generates packaging/macos/icon.icns from packaging/icon.png.
#
# Uses sips/iconutil, which only exist on macOS, so this runs in CI
# (macos-14 runner) rather than being committed as a prebuilt binary asset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SOURCE_PNG="$REPO_ROOT/packaging/icon.png"
ICONSET_DIR="$REPO_ROOT/packaging/macos/icon.iconset"
OUTPUT_ICNS="$REPO_ROOT/packaging/macos/icon.icns"

rm -rf "$ICONSET_DIR"
mkdir -p "$ICONSET_DIR"

for size in 16 32 128 256 512; do
  sips -z "$size" "$size" "$SOURCE_PNG" \
    --out "$ICONSET_DIR/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z "$double" "$double" "$SOURCE_PNG" \
    --out "$ICONSET_DIR/icon_${size}x${size}@2x.png" >/dev/null
done

iconutil -c icns "$ICONSET_DIR" -o "$OUTPUT_ICNS"
rm -rf "$ICONSET_DIR"

echo "Wrote $OUTPUT_ICNS"
