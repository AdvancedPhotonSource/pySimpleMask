#!/usr/bin/env python3
# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Convert mask.ui to ui_mask.py using pyside6-uic.

Run from anywhere:
    python src/pysimplemask/gui/view/compile_ui.py
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
UI_SRC = HERE / "mask.ui"
UI_OUT = HERE / "ui_mask.py"
LICENSE_HEADER = "# Copyright © UChicago Argonne LLC\n# See LICENSE file for details\n"


def add_license_header(path):
    """Insert the Argonne license header after the uic encoding line."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if LICENSE_HEADER in "".join(lines[:4]):
        return
    pos = 1 if lines and lines[0].startswith("# -*- coding") else 0
    lines.insert(pos, LICENSE_HEADER)
    path.write_text("".join(lines), encoding="utf-8")


def main():
    cmd = [sys.executable, "-m", "PySide6.scripts.uic", str(UI_SRC), "-o", str(UI_OUT)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # fall back to the pyside6-uic binary beside this interpreter, then on PATH
        import shutil
        uic = shutil.which("pyside6-uic", path=str(Path(sys.executable).parent))
        uic = uic or shutil.which("pyside6-uic")
        if uic is None:
            print("ERROR: pyside6-uic not found. Install PySide6 or activate the project env.")
            sys.exit(1)
        cmd = [uic, str(UI_SRC), "-o", str(UI_OUT)]
        print(f"Retrying: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        add_license_header(UI_OUT)
        print(f"OK  {UI_OUT}")
    else:
        print(result.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
