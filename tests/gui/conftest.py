# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""The GUI tests need PySide6, the ``gui`` extra; without it this folder is left out at collection."""

import gc
import importlib.util

import pytest

if importlib.util.find_spec("PySide6") is None:
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(autouse=True)
def _collect_garbage_between_tests():
    """Keep the garbage collector from running in the middle of a Qt call.

    Each test builds its own main window and drops it. When the collector
    later destroys that window's pyqtgraph items while Qt is constructing a
    new graphics item, pyqtgraph's ``itemChange`` is called on an item that is
    being destroyed and the interpreter segfaults. Where a collection lands
    depends only on how much was allocated before, so any change to the tests
    moved the crash to another test or made it come and go. Collect between
    tests instead, where it is safe.
    """
    gc.disable()
    yield
    gc.collect()
    gc.enable()
