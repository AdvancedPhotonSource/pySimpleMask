import logging
from ..base_reader import FileReader
from . import HdfDataset, ImmDataset, RigakuDataset, Rigaku3MDataset

logger = logging.getLogger(__file__)


class APS8IDIReader(FileReader):
    def __init__(self, fname) -> None:
        super(APS8IDIReader, self).__init__(fname)
        self.handler = None
        self.ftype = "APS_8IDI"
        self.stype = "Transmission"
        rigaku_endings = tuple(f".bin.00{i}" for i in range(6))

        if fname.endswith(".bin"):
            logger.info("Rigaku 500k dataset")
            self.handler = RigakuDataset(fname, batch_size=1000)
        elif fname.endswith(rigaku_endings):
            logger.info("Rigaku 3M (6 x 500K) dataset")
            self.handler = Rigaku3MDataset(fname, batch_size=1000)
        elif fname.endswith(".imm"):
            logger.info("IMM dataset")
            self.handler = ImmDataset(fname, batch_size=100)
        elif fname.endswith(".h5") or fname.endswith(".hdf"):
            logger.info("APS HDF dataset")
            self.handler = HdfDataset(fname, batch_size=100)
        else:
            logger.error("Unsupported APS dataset")
            return None
        self.shape = self.handler.det_size

    def get_scattering(self, **kwargs):
        return self.handler.get_scattering(**kwargs)

    def _get_metadata(self):
        return self.handler._get_metadata()
