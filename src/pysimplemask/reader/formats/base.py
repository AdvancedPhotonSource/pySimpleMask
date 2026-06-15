"""Abstract base class for low-level scattering-format loaders.

A format loader turns a raw detector file (HDF5, IMM, Rigaku binary, ...) into a
2-D mean scattering image. It deliberately exposes only what pySimpleMask needs:
``det_size`` and :meth:`get_scattering`.
"""

import abc
import os


class ScatteringDataset(abc.ABC):
    """Minimal interface every format loader implements."""

    def __init__(self, fname):
        self.fname = fname
        # Subclasses set the real detector shape during construction.
        self.det_size = (0, 0)

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
            num_frames: Number of frames to average. ``<= 0`` means all
                available frames from ``begin_idx``.
            begin_idx: First frame index to include.
            num_processes: Optional worker count for loaders that parallelize;
                ignored by loaders that do not.

        Returns:
            np.ndarray: 2-D image of shape :attr:`det_size`.
        """
        raise NotImplementedError
