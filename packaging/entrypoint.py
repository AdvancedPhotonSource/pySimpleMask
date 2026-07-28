# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""PyInstaller entry point for the pySimpleMask GUI."""

import sys

from pysimplemask.gui.app import main_gui


def run():
    # On Windows, PyInstaller sets sys.argv[0] to the bootloader path;
    # pass nothing (GUI defaults to cwd) unless --path is given.
    path = None
    if len(sys.argv) > 1:
        # Accept --path DIR or -p DIR for consistency with the CLI
        if sys.argv[1] in ('--path', '-p') and len(sys.argv) > 2:
            path = sys.argv[2]
        elif len(sys.argv) > 1:
            path = sys.argv[1]
    main_gui(path)


if __name__ == "__main__":
    run()