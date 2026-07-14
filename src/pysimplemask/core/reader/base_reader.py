"""App-facing reader base class shared by all beamline readers.

``FileReader`` owns the metadata, q-map, display-channel stack, mask state, and
beam-center handling that the kernel and GUI consume. Beamline subclasses supply
the raw scattering image (:meth:`get_scattering`) and metadata
(:meth:`_get_metadata`).
"""

import logging
import re

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from ..qmap import compute_display_center, compute_qmap

logger = logging.getLogger(__name__)


DISPLAY_FIELD = [
    "scattering",
    "scattering * mask",
    "mask",
    "dqmap_partition",
    "sqmap_partition",
    "preview",
]


def dict_to_params(name, data_dict, meta_units_formats=None):
    """Build a pyqtgraph ParameterTree group definition from a metadata dict."""

    def get_param_type(value):
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        return "str"

    children = []
    for key, value in data_dict.items():
        line = {"name": key, "type": get_param_type(value), "value": value}
        if meta_units_formats and key in meta_units_formats:
            _, unit, fmt_str = meta_units_formats[key]
            line["suffix"] = f" {unit}"
            line["siPrefix"] = False
            match = re.search(r"%\.(\d+)f", fmt_str)
            if match:
                line["decimals"] = int(match.group(1))
        children.append(line)

    return {"name": name, "type": "group", "children": children}


def get_fake_metadata():
    """Return placeholder metadata used when real metadata cannot be read.

    Includes every field required by ``compute_transmission_qmap`` so that
    any reader that falls back to this dict can complete qmap computation.
    ``detector_shape_x/y`` are omitted here because ``prepare_data`` sets
    them from the actual image shape after ``get_scattering()`` returns.
    """
    return {
        "energy": 12.3,               # keV
        "detector_distance": 12.3456, # meter
        "pixel_size": 75e-6,          # meter
        "beam_center_x": 512.0,       # pixel
        "beam_center_y": 256.0,       # pixel
        "swing_angle_horizontal": 0.0,  # degree
        "swing_angle_vertical": 0.0,    # degree
    }


def smart_float(x, precision=4):
    """Format a float as a clean fixed-point or scientific string."""
    if x == 0 or (1e-2 <= abs(x) < 1e2):
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return f"{x:.{precision}e}"


def _coerce_float(value):
    """Cast numeric scalars (incl. NumPy types) to float, leaving others as-is."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray) and value.size == 1:
        return float(value.reshape(-1)[0])
    return value


class FileReader(object):
    def __init__(self, fname) -> None:
        self.fname = fname
        self.ftype = "Base Class"
        self.stype = "Transmission"
        self.metadata = None
        self.shape = None
        self.qmap = None
        self.qmap_unit = None
        self.meta_units_fmts = None
        self.data_display = None
        self.scat = None
        self.scat_log = None

    def prepare_data(self, *args, **kwargs):
        self.metadata = self.get_metadata()
        self.scat = self.get_scattering(*args, **kwargs).astype(np.float32)
        self.shape = self.scat.shape
        # update metadata shape with the real values
        self.metadata["detector_shape_x"] = self.shape[1]
        self.metadata["detector_shape_y"] = self.shape[0]
        len_qmap = 7 if self.stype == "Transmission" else 11
        self.data_display = np.zeros((len(DISPLAY_FIELD) + len_qmap, *self.shape))
        self.scat_log = self.get_scat_with_mask(mask=None, mode="log")
        self.data_display[DISPLAY_FIELD.index("scattering")] = self.scat_log
        self.update_mask()

    def get_scat_with_mask(self, mask=None, mode="log"):
        if mask is None:
            mask = np.ones(self.shape, dtype=bool)
        temp = self.scat * mask
        positive = temp[temp > 0]
        # Avoid np.min of an empty array when a frame has no positive pixels.
        floor = positive.min() if positive.size else 1.0
        temp[temp <= 0] = floor
        if mode == "log":
            return np.log10(temp)
        return temp

    def update_mask(self, mask=None):
        if mask is None:
            mask = np.ones(self.shape, dtype=bool)
        self.data_display[DISPLAY_FIELD.index("mask")] = mask
        self.data_display[DISPLAY_FIELD.index("scattering * mask")] = (
            self.get_scat_with_mask(mask)
        )

    def update_partitions(self, dqmap, sqmap):
        self.data_display[DISPLAY_FIELD.index("dqmap_partition")] = dqmap
        self.data_display[DISPLAY_FIELD.index("sqmap_partition")] = sqmap

    def get_pts_with_similar_intensity(self, cen=None, radius=50, variation=50):
        vmin = max(0, int(cen[0] - radius))
        vmax = min(self.shape[0], int(cen[0] + radius))
        hmin = max(0, int(cen[1] - radius))
        hmax = min(self.shape[1], int(cen[1] + radius))
        crop = self.scat[vmin:vmax, hmin:hmax]
        val = self.scat[cen]
        idx = np.abs(crop - val) <= variation / 100.0 * val
        pos = np.array(np.nonzero(idx))
        pos[0] += vmin
        pos[1] += hmin
        pos = np.roll(pos, shift=1, axis=0)
        return pos.T

    def get_scattering(self, *args, **kwargs):
        raise NotImplementedError

    def get_metadata(self, *args, **kwargs):
        try:
            metadata = self._get_metadata(*args, **kwargs)
        except Exception:
            logger.warning(
                "failed to get the real metadata, using default values instead",
                exc_info=True,
            )
            return get_fake_metadata()
        # convert numerics to float for consistent downstream processing
        return {key: _coerce_float(value) for key, value in metadata.items()}

    def find_maximal_intensity_center(
        self, median_size: int = 3, gaussian_sigma: float = 1.0
    ) -> tuple:
        """Locate the brightest point of the masked image after denoising.

        Median then Gaussian filtering reduces the influence of hot pixels.

        Returns:
            tuple: ``(row, col)`` of the maximum in the smoothed image.
        """
        scat_mask = self.data_display[DISPLAY_FIELD.index("scattering * mask")]
        cleaned = median_filter(scat_mask, size=median_size)
        smoothed = gaussian_filter(cleaned, sigma=gaussian_sigma)
        return np.unravel_index(np.argmax(smoothed), smoothed.shape)

    def _get_metadata(self, *args, **kwargs):
        raise NotImplementedError

    def get_parametertree_structure(self):
        return dict_to_params("metadata", self.metadata, self.meta_units_fmts)

    def update_metadata_from_changes(self, changes):
        for changed_param, _change_type, new_value in changes:
            self.metadata[changed_param.name()] = new_value

    def update_metadata(self, new_metadata):
        if new_metadata:
            self.metadata.update(new_metadata)

    def compute_qmap(self):
        # Always derive detector shape from the actual scattering image so that
        # a manually-edited ParameterTree value (detector_shape_x/y) or a
        # stale metadata entry can never produce a qmap whose shape disagrees
        # with data_display and causes a broadcast ValueError.
        self.metadata["detector_shape_x"] = self.shape[1]
        self.metadata["detector_shape_y"] = self.shape[0]
        self.qmap, self.qmap_unit = compute_qmap(self.stype, self.metadata)
        for index, value in enumerate(self.qmap.values()):
            self.data_display[len(DISPLAY_FIELD) + index] = value
        labels = list(DISPLAY_FIELD) + list(self.qmap.keys())
        return self.qmap, self.qmap_unit, labels

    def get_center(self, mode="vh"):
        if mode not in ("xy", "vh"):
            raise ValueError(f"mode must be 'xy' or 'vh', got {mode!r}")
        display_center = compute_display_center(
            (self.metadata["beam_center_y"], self.metadata["beam_center_x"]),
            self.metadata["detector_distance"],
            self.metadata["pixel_size"],
            self.metadata.get("swing_angle_horizontal", 0),
            self.metadata.get("swing_angle_vertical", 0),
        )
        if mode == "xy":
            return (display_center[1], display_center[0])
        return display_center

    def swapxy(self):
        self.metadata["beam_center_x"], self.metadata["beam_center_y"] = (
            self.metadata["beam_center_y"],
            self.metadata["beam_center_x"],
        )

    def set_center_vh(self, new_center_vh):
        self.metadata["beam_center_y"] = float(new_center_vh[0])
        self.metadata["beam_center_x"] = float(new_center_vh[1])

    def get_coordinates(self, col, row, index):
        shape = self.shape
        if col < 0 or col >= shape[1]:
            return None
        if row < 0 or row >= shape[0]:
            return None

        if self.stype == "Reflection":
            labels = ["phi", "tth", "alpha_f", "qx", "qy", "qz", "qr", "q"]
        else:
            labels = ["phi", "TTH", "qx", "qy", "q"]

        msg = f"xy=[{col:d},{row:d}]  "
        if self.qmap:
            qmap_labels = list(self.qmap.keys())
            begin = len(DISPLAY_FIELD)
            selection = [begin + qmap_labels.index(k) for k in labels]
            selection.append(index)
            labels.append("data")

            values = self.data_display[:, row, col][selection]
            values = [smart_float(v) for v in values]
            for k, v in zip(labels, values):
                if k in ["qx", "qy", "qz", "q", "qr"]:
                    v = f"{v}Å⁻¹"
                elif k in ["tth", "alpha_f", "phi", "TTH"]:
                    v = f"{v}°"
                msg += f"{k}={v}, "

        return msg[:-2]

    def set_preview(self, img):
        self.data_display[DISPLAY_FIELD.index("preview")] = img
