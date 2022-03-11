"""Top-level package for pySimpleMask."""

__author__ = """Miaoqi Chu"""
__email__ = 'mqichu@anl.gov'


from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    __version__ = "0.0.0"

from .mask_gui import run 
__all__ = (run, )
