# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import sys
import types

import numpy as np

from pysimplemask.core.mask import MaskAssemble


def test_default_blemish_is_coerced_to_bool(monkeypatch):
    """ADHelper.loader.get_blemish may return whatever dtype the blemish TIFF was
    saved with (area-detector blemish files are commonly uint8). self.blemish feeds
    self.mask directly via the "default_blemish" apply path (SimpleMaskModel.mask_apply),
    bypassing the np.logical_and normalization that every other mask path goes through.
    A non-bool mask silently turns `xmap[mask]` (e.g. main_window.update_xmap_limits)
    from boolean masking into advanced integer indexing, blowing up a 2-D array into a
    3-D one shaped mask.shape + xmap.shape[1:].
    """
    shape = (4, 5)
    fake_blemish = np.ones(shape, dtype=np.uint8)

    fake_adhelper = types.ModuleType("ADHelper")
    fake_loader = types.ModuleType("ADHelper.loader")
    fake_loader.get_blemish = lambda detector_shape: fake_blemish
    monkeypatch.setitem(sys.modules, "ADHelper", fake_adhelper)
    monkeypatch.setitem(sys.modules, "ADHelper.loader", fake_loader)

    ma = MaskAssemble(shape=shape, saxs_lin=np.zeros(shape))

    assert ma.blemish.dtype == np.bool_
