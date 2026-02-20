"""Console script for pysimplemask."""
import argparse
import logging
import sys
import os
from pysimplemask import main_gui, __version__
from pysimplemask.utils import combine_qmap_files


def main():
    """Console script for pysimplemask."""
    parser = argparse.ArgumentParser('pySimpleMask: A GUI for creating mask and q-partition maps for scattering patterns in preparation for SAXS/WAXS/XPCS data reduction')
    parser.add_argument('--path', '-p', required=False, default=os.getcwd())
    parser.add_argument("--version", action="version",
                        version=f"pySimpleMask {__version__}")
    args = parser.parse_args()
    sys.exit(main_gui(args.path))


def combine_qmaps():
    """CLI entry point: combine two qmap HDF5 files into one."""
    parser = argparse.ArgumentParser(
        prog="pysimplemask-combine-qmaps",
        description="Combine two pySimpleMask qmap HDF5 files into a single output file.",
    )
    parser.add_argument("qmap_file1", help="Path to the first qmap HDF5 file.")
    parser.add_argument("qmap_file2", help="Path to the second qmap HDF5 file.")
    parser.add_argument("output_file", help="Path for the combined output qmap HDF5 file.")
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    combine_qmap_files(args.qmap_file1, args.qmap_file2, args.output_file)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
