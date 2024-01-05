"""Console script for pysimplemask."""
import argparse
import sys
import os


def main():
    """Console script for pysimplemask."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', required=False, default=os.getcwd())
    args = parser.parse_args()
    kwargs = vars(args)
    from .pysimplemask import run
    run(**kwargs)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
