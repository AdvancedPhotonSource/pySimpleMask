import logging
import numpy as np
import traceback
from ..qmap import compute_qmap

logger = logging.getLogger(__name__)


def dict_to_params(name, d):
    """Recursively convert a Python dictionary to ParameterTree structure."""
    children = []
    for key, value in d.items():
        if isinstance(value, dict):
            children.append(dict_to_params(key, value))
        else:
            param_type = "str"  # default
            if isinstance(value, int):
                param_type = "int"
            elif isinstance(value, float):
                param_type = "float"
            elif isinstance(value, bool):
                param_type = "bool"
            children.append({"name": key, "type": param_type, "value": value})
    return {"name": name, "type": "group", "children": children}


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
        "det_dist": 12.3456,  # meter
        "pix_dim": 75e-6,  # meter
        "bcx": 512,
        "bcy": 256,
        "stype": "transmission",
    }
    return metadata


def smart_float(x, precision=2):
    """
    Convert a float to a string, either in scientific notation or fixed-point notation.
    The precision is the number of digits after the decimal point.
    """
    if x == 0 or (1e-4 <= abs(x) < 1e4):
        return f"{x:.{precision}f}".rstrip("0").rstrip(".")  # clean float
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
        self.data_display = None

    def prepare_data(self, *args, **kwargs):
        self.metadata = self.get_metadata()
        self.scat = self.get_scattering(*args, **kwargs).astype(np.float32)
        self.shape = self.scat.shape
        self.metadata["shape"] = self.shape
        len_qmap = 7 if self.stype == "Transmission" else 11
        self.data_display = np.zeros((len(DISPLAY_FIELD) + len_qmap, *self.shape))
        self.scat_log = self.get_scat_with_mask(mask=None, mode="log")
        self.data_display[DISPLAY_FIELD.index("scattering")] = self.scat_log
        self.update_mask()

    def get_scat_with_mask(self, mask=None, mode="log"):
        if mask is None:
            mask = np.ones(self.shape, dtype=bool)
        min_mask = (self.scat > 0) * mask
        if np.sum(min_mask) == 0:
            return self.scat

        # nonzero min
        nz_min = np.min(self.scat[min_mask > 0])
        temp = np.copy(self.scat)
        temp[~min_mask] = nz_min
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
            metdata = self._get_metadata(*args, **kwargs)
            return metdata
        except Exception as e:
            traceback.print_exc()
            logger.warning(
                "failed to get the real metadata, using default values instead"
            )
            return get_fake_metadata()

    def _get_metadata(self, *args, **kwargs):
        raise NotImplementedError

    def get_parametertree_structure(self):
        return dict_to_params("metadata", self.metadata)

    def update_metadata_from_changes(self, changes):
        for changed_param, change_type, new_value in changes:
            # change_type can be 'value', 'name', 'parent', 'children', 'flags'
            # not used
            self.metadata[changed_param.name()] = new_value

    def update_metadata(self, new_metadata):
        self.metadata.update(new_metadata)

    def get_qmap(self):
        self.qmap, self.qmap_unit = compute_qmap(self.stype, self.metadata)
        for index, (k, v) in enumerate(self.qmap.items()):
            self.data_display[len(DISPLAY_FIELD) + index] = v
        labels = list(DISPLAY_FIELD) + list(self.qmap.keys())
        return self.qmap, self.qmap_unit, labels

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

        qmap_labels = list(self.qmap.keys())
        begin = len(DISPLAY_FIELD)
        selection = [begin + qmap_labels.index(k) for k in labels]
        selection.append(index)
        labels.append("data")

        values = self.data_display[:, row, col][selection]
        values = [smart_float(v) for v in values]

        msg = f"xy=[{col:d},{row:d}] "
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
