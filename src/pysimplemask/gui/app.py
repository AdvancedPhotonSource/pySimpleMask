# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""GUI application entry point."""

import sys

from PySide6.QtWidgets import QApplication

from pysimplemask.gui.control.main_window import SimpleMaskGUI


def main_gui(path=None):
    app = QApplication(sys.argv)
    window = SimpleMaskGUI(path)  # noqa: F841 (kept alive by the event loop)
    app.exec()


if __name__ == "__main__":
    main_gui()
