"""Top-level package for pySimpleMask."""

from importlib.metadata import PackageNotFoundError, version

__author__ = """Miaoqi Chu"""
__email__ = "mqichu@anl.gov"

try:
    __version__ = version("pysimplemask")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback if package is not installed

# NOTE: keep this module Qt-free. The scriptable core model is re-exported once
# core/model.py exists (added during the MVC refactor); the GUI entry point
# lives in pysimplemask.gui.app and is imported lazily by the CLI.
