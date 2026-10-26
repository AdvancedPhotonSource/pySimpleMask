# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""The GUI tests need PySide6, the ``gui`` extra; without it this folder is left out at collection."""

import importlib.util

if importlib.util.find_spec("PySide6") is None:
    collect_ignore_glob = ["test_*.py"]
