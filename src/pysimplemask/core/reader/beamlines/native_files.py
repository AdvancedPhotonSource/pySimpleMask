# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""NativeFiles reader — TIFF/TIFF scattering images with placeholder metadata."""

from __future__ import annotations

import logging

import numpy as np
import tifffile

from ..base_reader import FileReader, get_fake_metadata

logger = logging.getLogger(__name__)


class NativeFilesReader(FileReader):
    """Reader for native TIFF/TIFF scattering images.

    All instrument parameters (beam center, energy, …) are populated with
    placeholder values from :func:`get_fake_metadata`.  Users can edit them
    via the metadata panel after loading.
    """

    ftype = "NativeFiles"
    stype = "Transmission"

    def __init__(self, fname: str) -> None:
        super().__init__(fname)
        self.meta_units_fmts = None  # no units/formatting info

    def get_scattering(self, **kwargs) -> np.ndarray:
        """Read a TIFF/TIFF file and return a 2-D float32 mean image.

        A 3-D array ``(frames, H, W)`` is averaged over the frame axis.
        """
        data = tifffile.imread(self.fname).astype(np.float32)
        if data.ndim == 3:
            data = data.mean(axis=0)
        if data.ndim != 2:
            raise ValueError(
                f"Expected a 2-D or 3-D TIFF array, got shape {data.shape}"
            )
        return data

    def _get_metadata(self) -> dict:
        """Return placeholder metadata; edited by the user after loading."""
        return get_fake_metadata()
