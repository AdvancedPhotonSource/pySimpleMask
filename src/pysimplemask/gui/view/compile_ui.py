#!/usr/bin/env python3
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


def main():
    cmd = [sys.executable, "-m", "PySide6.scripts.uic", str(UI_SRC), "-o", str(UI_OUT)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # fall back to pyside6-uic binary on PATH
        import shutil
        uic = shutil.which("pyside6-uic")
        if uic is None:
            print("ERROR: pyside6-uic not found. Install PySide6 or activate the project env.")
            sys.exit(1)
        cmd = [uic, str(UI_SRC), "-o", str(UI_OUT)]
        print(f"Retrying: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(f"OK  {UI_OUT}")
    else:
        print(result.stderr)
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
