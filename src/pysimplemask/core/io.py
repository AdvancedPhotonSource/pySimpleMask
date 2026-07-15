# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Parsing helpers for pixel-coordinate input (text and files)."""

import json

import numpy as np


def text_to_array(pts, dtype=np.int64):
    """Parse a free-form string of numbers (brackets/commas ignored) into a 1-D array."""
    for symbol in "[](),":
        pts = pts.replace(symbol, " ")
    tokens = [tok for tok in pts.split(" ") if tok != ""]
    if dtype == np.int64:
        values = [int(tok) for tok in tokens]
    elif dtype == np.float64:
        values = [float(tok) for tok in tokens]
    else:
        values = [dtype(tok) for tok in tokens]
    return np.array(values).astype(dtype)


def load_pixel_list(fname):
    """Load an (N, 2) array of [x, y] pixel coordinates from .json/.txt/.csv.

    JSON format: ``{"Bad pixels": [{"Pixel": [x, y]}, ...]}``.
    Text/CSV: two columns, comma- or whitespace-separated.
    """
    if fname.endswith(".json"):
        with open(fname, "r") as f:
            entries = json.load(f)["Bad pixels"]
        xy = np.array([entry["Pixel"] for entry in entries])
    elif fname.endswith(".txt") or fname.endswith(".csv"):
        try:
            xy = np.loadtxt(fname, delimiter=",")
        except ValueError:
            xy = np.loadtxt(fname)
    else:
        raise ValueError(f"unsupported pixel-list file: {fname}")
    return xy.astype(np.int64).reshape(-1, 2)
