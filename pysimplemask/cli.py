"""Console script for pysimplemask."""
import argparse
import sys
import os


def main():
    """Console script for pysimplemask."""
    parser = argparse.ArgumentParser()
    parser.add_argument('path', default=os.getcwd())
    args = parser.parse_args()
    kwargs = vars(args)
    from .mask_gui import run
    run(**kwargs)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
