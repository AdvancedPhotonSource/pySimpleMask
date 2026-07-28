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

    def _get_metadata(self, metadata_fname: str | None = None) -> dict:
        """Return placeholder metadata, or real metadata from an override file.

        When ``metadata_fname`` is provided and contains valid APS 8-ID-I NeXus
        fields, those values are returned.  Otherwise a warning is logged and
        placeholder metadata is used.
        """
        if metadata_fname:
            from ..metadata import has_nexus_fields
            from .aps_8idi import METADATA_KEYMAPS, OPTIONAL_FIELDS, get_nexus_metadata

            if has_nexus_fields(metadata_fname, METADATA_KEYMAPS, OPTIONAL_FIELDS):
                return get_nexus_metadata(metadata_fname)
            logger.warning(
                "metadata_fname %s is missing required 8-ID-I NeXus fields; "
                "using placeholder metadata", metadata_fname,
            )
        return get_fake_metadata()
