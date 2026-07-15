# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Top-level package for pySimpleMask."""

from importlib.metadata import PackageNotFoundError, version

__author__ = """Miaoqi Chu"""
__email__ = "mqichu@anl.gov"

try:
    __version__ = version("pysimplemask")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback if package is not installed

# NOTE: keep this module Qt-free. The GUI entry point lives in pysimplemask.gui.app
# and is imported lazily by the CLI.
from .core.model import SimpleMaskModel  # noqa: E402 (after __version__ is defined)

__all__ = ["SimpleMaskModel", "__version__"]
