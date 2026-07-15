# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import h5py
import numpy as np
import pytest


@pytest.fixture
def make_hdf(tmp_path):
    def _make(frames, data_path="/entry/data/data", name="data.h5"):
        path = tmp_path / name
        with h5py.File(path, "w") as h:
            h[data_path] = np.asarray(frames)
        return str(path)

    return _make
