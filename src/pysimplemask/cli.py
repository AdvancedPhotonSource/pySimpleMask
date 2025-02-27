"""Console script for pysimplemask."""
import argparse
import sys
import os
from pysimplemask import main_gui, __version__


def main():
    """Console script for pysimplemask."""
    parser = argparse.ArgumentParser('pySimpleMask: A GUI for creating mask and q-partition maps for scattering patterns in preparation for SAXS/WAXS/XPCS data reduction')
    parser.add_argument('--path', '-p', required=False, default=os.getcwd())
    parser.add_argument("--version", action="version",
                        version=f"pySimpleMask {__version__}")
    args = parser.parse_args()
    sys.exit(main_gui(args.path))


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
