# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Abstract base class for low-level scattering-format loaders.

A format loader turns a raw detector file (HDF5, IMM, Rigaku binary, ...) into a
2-D mean scattering image. It deliberately exposes only what pySimpleMask needs:
``det_size`` and :meth:`get_scattering`, plus, for the loaders that can read a
few frames quickly, :meth:`get_frames_mean` to browse the frames.
"""

import abc
import os


class ScatteringDataset(abc.ABC):
    """Minimal interface every format loader implements."""

    # Frames averaged into one image, by default, when browsing the frames.
    # Loaders of sparse frames, each too faint to show on its own, set more.
    default_frame_average = 1

    def __init__(self, fname):
        self.fname = fname
        # Subclasses set the real detector shape during construction.
        self.det_size = (0, 0)
        # Number of frames that get_frames_mean can read. Loaders that support
        # it set this during construction; it stays 0 for the others.
        self.num_browsable_frames = 0

    @property
    def file_size_mb(self):
        """Size of the backing file in MiB (0 if it cannot be determined)."""
        try:
            return os.path.getsize(self.fname) / (1024**2)
        except OSError:
            return 0.0

    @abc.abstractmethod
    def get_scattering(self, num_frames=-1, begin_idx=0, num_processes=None):
        """Return the per-pixel mean scattering image over a frame range.

        Args:
            num_frames: Frames to average, starting at ``begin_idx``. ``>= 1`` is
                that many frames; ``0``/``None`` is all remaining frames; a
                fraction between 0 and 1, or ``< 0``, is a representative
                subset (see ``resolve_frame_range``).
            begin_idx: First frame index to include.
            num_processes: Optional worker count for loaders that parallelize;
                ignored by loaders that do not.

        Returns:
            np.ndarray: 2-D image of shape :attr:`det_size`.
        """
        raise NotImplementedError

    def get_frames_mean(self, begin_idx, end_idx):
        """Return the per-pixel mean of the frames ``[begin_idx, end_idx)``.

        This is for browsing the frames a few at a time, so it has to be quick
        for a short range, where :meth:`get_scattering` is built to average a
        large share of the dataset. Only the loaders that set
        ``num_browsable_frames`` implement it.

        Returns:
            np.ndarray: 2-D float32 image of shape :attr:`det_size`.

        Raises:
            ValueError: If the range is empty or reaches outside the frames.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot read a short range of frames"
        )

    def _check_frame_range(self, begin_idx, end_idx):
        """Raise ``ValueError`` unless ``[begin_idx, end_idx)`` is a non-empty
        range of the browsable frames."""
        if not 0 <= begin_idx < end_idx <= self.num_browsable_frames:
            raise ValueError(
                f"frames [{begin_idx}, {end_idx}) are not within the "
                f"{self.num_browsable_frames} frames of {self.fname}"
            )
