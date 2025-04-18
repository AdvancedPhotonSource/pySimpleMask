import logging


logger = logging.getLogger(__name__)


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
        pass

    def get_scattering(self, *args, **kwargs):
        raise NotImplementedError

    def get_metadata(self, *args, **kwargs):
        try:
            metdata = self._get_metadata(*args, **kwargs)
            return metdata
        except Exception as e:
            logger.warning(e)
            logger.warning("failed to get the real metadata, using default values instead")
            return get_fake_metadata()

    def _get_metadata(self, *args, **kwargs):
        raise NotImplementedError
