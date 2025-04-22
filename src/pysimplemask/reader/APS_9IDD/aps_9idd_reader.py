import logging

import h5py
import os
import glob
from ..base_reader import FileReader
from ..utils import sum_frames_parallel

logger = logging.getLogger(__name__)


class APS9IDDReader(FileReader):
    def __init__(self, fname) -> None:
        super().__init__(fname)
        self.ftype = "APS_9IDD"
        self.stype = "Reflection"

    def get_scattering(self, num_frames=-1, begin_idx=0, num_processes=None):
        return sum_frames_parallel(
            self.fname,
            dataset_name="/entry/data/data",
            start_frame=begin_idx,
            num_frames=num_frames,
            chunk_size=32,
            num_processes=num_processes,
        )

    def _get_metadata_test(self, *args, **kwargs):
        with h5py.File(self.fname, "r") as f:
            metadata = {
                "alpha_i_deg": f["/entry/metadata/alpha_incident_deg"][()],
                "bcx": f["/entry/metadata/beam_center"][0],
                "bcy": f["/entry/metadata/beam_center"][1],
                "det_dist": f["/entry/metadata/det_distance"][()],
                "pix_dim": f["/entry/metadata/det_pixel_size"][()],
                "det_shape": f["/entry/metadata/det_shape"][()],
                "det_y": f["/entry/metadata/det_y"][()],
                "det_z": f["/entry/metadata/det_z"][()],
                "energy": f["/entry/metadata/energy"][()],
            }
        return metadata

    def _get_metadata(self, *args, **kwargs):
        prefix = os.path.join(os.path.dirname(self.fname), "*_metadata.hdf")
        meta_fnames = glob.glob(prefix)
        assert (
            len(meta_fnames) > 0
        ), f"no *_metadata.hdf found in the folder of {self.fname}"
        if len(meta_fnames) > 1:
            logger.warning(
                f"multiple *_metadata.hdf found in the folder of {self.fname}. using the first one"
            )
        meta_fname = meta_fnames[0]
        logger.info(f"using metadata file: {meta_fname}")

        # read real metadata
        with h5py.File(meta_fname, "r") as f:
            metadata = {
                "alpha_i_deg": f["/entry/sample/incident_angle"][()],
                "bcx": f["/entry/instrument/detector_1/beam_center_x"][()],
                "bcy": f["/entry/instrument/detector_1/beam_center_y"][()],
                "det_dist": f["/entry/instrument/detector_1/distance"][()],
                "pix_dim": f["/entry/instrument/detector_1/x_pixel_size"][()],
                "det_shape": None,
                "det_x0": f["/entry/instrument/detector_1/beam_center_position_x"][()],
                "det_y0": f["/entry/instrument/detector_1/beam_center_position_y"][()],
                "det_x": f["/entry/instrument/detector_1/position_x"][()],
                "det_y": f["/entry/instrument/detector_1/position_y"][()],
                "energy": f["/entry/instrument/incident_beam/incident_energy"][()],
            }
        return metadata
