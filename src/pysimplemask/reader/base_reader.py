import logging


logger = logging.getLogger(__name__)


def get_fake_metadata(shape):
    # fake metadata
    logger.warning("failed to get the raw metadata, using default values instead")
    metadata = {
        # 'datetime': "2022-05-08 14:00:51,799",
        "energy": 12.3,  # keV
        "det_dist": 12.3456,  # meter
        "pix_dim": 75e-3,  # meter
        "bcx": 773.0,
        "bcy": 809.0,
        "stype": "transmission",
    }
    return metadata


class FileReader(object):
    def __init__(self, fname) -> None:
        self.fname = fname
        self.ftype = "Base Class"
        self.stype = "Transmission"
        pass

    def get_scattering(self, *args, **kwargs):
        raise NotImplementedError

    def get_data(self, *args, **kwargs):
        return None

    def load_meta(self, *args, **kwargs):
        raise NotImplementedError
