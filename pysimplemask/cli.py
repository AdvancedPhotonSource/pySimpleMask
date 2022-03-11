"""Console script for pysimplemask."""
import argparse
import sys


def main():
    """Console script for pysimplemask."""
    parser = argparse.ArgumentParser()
    parser.add_argument('_', nargs='*')
    args = parser.parse_args()
    kwargs = vars(args)
    from .mask_gui import run
    run(**kwargs)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
