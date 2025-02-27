"""Top-level package for pySimpleMask."""
__author__ = """Miaoqi Chu"""
__email__ = 'mqichu@anl.gov'

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("pysimplemask")
except PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback if package is not installed

from .simplemask_main import main_gui
