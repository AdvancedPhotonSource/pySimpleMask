import logging
import numpy as np


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


class FileReader(object):
    def __init__(self, fname) -> None:
        self.fname = fname
        self.ftype = "Base Class"
        self.stype = "Transmission"
        self.metadata = None
        self.shape = None

    def prepare_data(self, *args, **kwargs):
        self.metadata = self.get_metadata()
        self.scat = self.get_scattering(*args, **kwargs).astype(np.float32)
        self.shape = self.scat.shape
        self.scat_log = self.get_scat_with_mask(mask=None, mode="log")

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
            logger.warning(e)
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
