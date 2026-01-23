import logging
import numpy as np
import traceback
import re
from ..qmap import compute_qmap, compute_display_center
from scipy.ndimage import median_filter, gaussian_filter

logger = logging.getLogger(__name__)


def dict_to_params(name, data_dict, meta_units_formats=None):
    def get_param_type(value):
        """Determines the parameter type based on the value's Python type."""
        if isinstance(value, bool):
            return "bool"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        else:
            return "str"

    # Convert the dictionary to a list of parameter definitions
    params = []
    for key, value in data_dict.items():
        param_type = get_param_type(value)
        line = {"name": key, "type": param_type, "value": value}

        if meta_units_formats and key in meta_units_formats:
            # Unpack the metadata tuple
            _, unit, fmt_str = meta_units_formats[key]
            # Set the unit as a suffix for display
            line["suffix"] = f" {unit}"  # Add a leading space for readability
            line["siPrefix"] = False
            # This is crucial for custom formatting of floats
            # 2. Parse format string to set the number of decimals
            match = re.search(r"%\.(\d+)f", fmt_str)
            if match:
                line["decimals"] = int(match.group(1))

        params.append(line)

    return {"name": name, "type": "group", "children": params}


def parameter_to_dict(parameter):
    """Recursively extract parameter values into a dictionary."""
    result = {}
    for child in parameter.children():
        if child.children():
            # Nested group — recurse
            result[child.name()] = parameter_to_dict(child)
        else:
            result[child.name()] = child.value()
    return result


def get_fake_metadata():
    """
    Generate a fake metadata dictionary for testing purposes.
    """
    metadata = {
        # 'datetime': "2022-05-08 14:00:51,799",
        "energy": 12.3,  # keV
        "detector_distance": 12.3456,  # meter
        "pixel_size": 75e-6,  # meter
        "beam_center_x": 512,
        "beam_center_y": 256,
        "stype": "transmission",
    }
    return metadata


def smart_float(x, precision=4):
    """
    Convert a float to a string, either in scientific notation or fixed-point notation.
    The precision is the number of digits after the decimal point.
    """
    if x == 0 or (1e-2 <= abs(x) < 1e2):
        return f"{x:.4f}".rstrip("0").rstrip(".")  # clean float
    else:
        return f"{x:.{precision}e}"  # scientific


DISPLAY_FIELD = [
    "scattering",
    "scattering * mask",
    "mask",
    "dqmap_partition",
    "sqmap_partition",
    "preview",
]


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
        nz_min = np.min(temp[temp > 0])
        temp[temp <= 0] = nz_min
        if mode == "log":
            return np.log10(temp)
        else:
            return temp

    def update_mask(self, mask=None):
        mask_loc = DISPLAY_FIELD.index("mask")
        if mask is None:
            mask = np.ones(self.shape, dtype=bool)
        self.data_display[mask_loc] = mask
        scat_mask_loc = DISPLAY_FIELD.index("scattering * mask")
        self.data_display[scat_mask_loc] = self.get_scat_with_mask(mask)

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
            for k, v in metadata.items():
                # convert to float for consistency and compatibility with downstream processing
                if isinstance(v, (int, np.floating)):
                    metadata[k] = float(v)
            return metadata
        except Exception as e:
            traceback.print_exc()
            logger.warning(
                "failed to get the real metadata, using default values instead"
            )
            return get_fake_metadata()

    def find_maximal_intensity_center(
        self, median_size: int = 3, gaussian_sigma: float = 1.0
    ) -> tuple:
        """
        Find the position of the maximum point in a 2D image robustly,
        using median and Gaussian filtering to reduce noise.

        Parameters:
            image (np.ndarray): 2D input image.
            median_size (int): Kernel size for median filtering (default: 3).
            gaussian_sigma (float): Sigma for Gaussian filtering (default: 1.0).

        Returns:
            tuple: Coordinates (row, col) of the maximum point in the smoothed image.
        """
        scat_mask_loc = DISPLAY_FIELD.index("scattering * mask")
        scat_mask = self.data_display[scat_mask_loc]
        cleaned = median_filter(scat_mask, size=median_size)
        smoothed = gaussian_filter(cleaned, sigma=gaussian_sigma)
        max_pos_vh = np.unravel_index(np.argmax(smoothed), smoothed.shape)
        return max_pos_vh

    def _get_metadata(self, *args, **kwargs):
        raise NotImplementedError

    def get_parametertree_structure(self):
        return dict_to_params("metadata", self.metadata, self.meta_units_fmts)

    def update_metadata_from_changes(self, changes):
        for changed_param, change_type, new_value in changes:
            # change_type can be 'value', 'name', 'parent', 'children', 'flags'
            # not used
            self.metadata[changed_param.name()] = new_value

    def update_metadata(self, new_metadata):
        if new_metadata:
            self.metadata.update(new_metadata)

    def compute_qmap(self):
        self.qmap, self.qmap_unit = compute_qmap(self.stype, self.metadata)
        for index, (k, v) in enumerate(self.qmap.items()):
            self.data_display[len(DISPLAY_FIELD) + index] = v
        labels = list(DISPLAY_FIELD) + list(self.qmap.keys())
        return self.qmap, self.qmap_unit, labels

    def get_center(self, mode="vh"):
        display_center = compute_display_center(
            (self.metadata["beam_center_y"], self.metadata["beam_center_x"]),
            self.metadata["detector_distance"],
            self.metadata["pixel_size"],
            self.metadata.get("swing_angle", 0),
        )
        if mode == "xy":
            return (display_center[1], display_center[0])
        elif mode == "vh":
            return display_center

    def swapxy(self):
        newx = self.metadata["beam_center_y"]
        newy = self.metadata["beam_center_x"]
        self.metadata["beam_center_x"] = newx
        self.metadata["beam_center_y"] = newy

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
        elif self.stype == "Transmission":
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
                elif k == "data":
                    v = f"{v}"
                msg += f"{k}={v}, "

        return msg[:-2]

    def set_preview(self, img):
        self.data_display[DISPLAY_FIELD.index("preview")] = img
        return
