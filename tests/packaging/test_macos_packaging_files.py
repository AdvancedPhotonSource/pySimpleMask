"""Static checks for the macOS packaging assets.

Neither `sips`/`iconutil` (used by make_icns.sh) nor real icon generation can
run on this dev machine's Linux/pytest environment, so these tests only check
the two things that ARE verifiable without a macOS runner: the shell script
is syntactically valid, and the entitlements file is a well-formed plist with
the two keys notarization needs.
"""

import plistlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAKE_ICNS = REPO_ROOT / "packaging" / "macos" / "make_icns.sh"
ENTITLEMENTS = REPO_ROOT / "packaging" / "macos" / "entitlements.plist"


def test_make_icns_script_exists_and_is_executable_bash():
    assert MAKE_ICNS.exists()
    result = subprocess.run(
        ["bash", "-n", str(MAKE_ICNS)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_entitlements_plist_is_well_formed():
    with open(ENTITLEMENTS, "rb") as f:
        data = plistlib.load(f)
    assert data["com.apple.security.cs.allow-unsigned-executable-memory"] is True
    assert data["com.apple.security.cs.disable-library-validation"] is True
