# Copyright © UChicago Argonne LLC
# See LICENSE file for details
import numpy as np

from pysimplemask.core.mask import MaskAssemble
from pysimplemask.core.rasterize import (
    RoiPolygon,
    group_index_map,
    rasterize,
    rectangle_vertices,
)


def _draw_kwargs(shape, rois):
    keep = rasterize(shape, rois)
    return {
        "arr": np.logical_not(keep),
        "group_index_map": group_index_map(shape, rois),
    }


def test_reapplying_mask_draw_with_new_inclusive_roi_includes_both_regions():
    """Regression: evaluate+apply mask_draw with one inclusive ROI, then draw a
    second non-overlapping inclusive ROI and evaluate+apply again. Both
    regions must end up included — a naive AND against the mask left over
    from the first apply would exclude the second ROI, since AND can never
    recover a pixel a previous apply already zeroed out."""
    shape = (10, 10)
    ma = MaskAssemble(shape=shape, saxs_lin=np.zeros(shape))

    roi1 = RoiPolygon(
        rectangle_vertices(center=(2, 2), size=(3, 3)), "inclusive", group_index=1
    )
    ma.evaluate("mask_draw", **_draw_kwargs(shape, [roi1]))
    ma.apply("mask_draw")

    roi2 = RoiPolygon(
        rectangle_vertices(center=(7, 7), size=(3, 3)), "inclusive", group_index=1
    )
    ma.evaluate("mask_draw", **_draw_kwargs(shape, [roi1, roi2]))
    mask = ma.apply("mask_draw")

    assert mask[2, 2]  # inside roi1 — still included
    assert mask[7, 7]  # inside roi2 — must now be included too


def test_apply_mask_draw_still_respects_previously_applied_other_masks():
    """mask_draw's fresh recomputation must still be ANDed with whatever
    other worker masks (e.g. threshold) were already applied — the fix
    changes what mask_draw is combined with, not whether other masks count."""
    shape = (10, 10)
    saxs = np.tile(np.arange(10.0), (10, 1))  # column j has value j everywhere
    ma = MaskAssemble(shape=shape, saxs_lin=saxs)
    ma.evaluate("mask_threshold", low=0, high=5, low_enable=True, high_enable=True)
    ma.apply("mask_threshold")

    roi = RoiPolygon(
        rectangle_vertices(center=(4.5, 4.5), size=(10, 10)), "inclusive", group_index=1
    )  # covers the whole image
    ma.evaluate("mask_draw", **_draw_kwargs(shape, [roi]))
    mask = ma.apply("mask_draw")

    assert mask[0, 0]       # column 0: within threshold [0,5) and inside the ROI
    assert not mask[0, 9]   # column 9: excluded by the threshold regardless of the ROI


def test_reset_keeps_contribution_record_in_sync_with_mask_record():
    shape = (10, 10)
    ma = MaskAssemble(shape=shape, saxs_lin=np.zeros(shape))
    ma.evaluate("mask_threshold", low=0, high=1, low_enable=True, high_enable=True)
    ma.apply("mask_threshold")
    roi = RoiPolygon(rectangle_vertices(center=(2, 2), size=(3, 3)), "inclusive", group_index=1)
    ma.evaluate("mask_draw", **_draw_kwargs(shape, [roi]))
    ma.apply("mask_draw")

    ma.redo_undo("reset")

    assert len(ma.contribution_record) == len(ma.mask_record)


def _disk(shape, center, radius):
    yy, xx = np.indices(shape)
    return np.hypot(yy - center[0], xx - center[1]) < radius


def test_beamstop_survives_a_later_draw_apply():
    """Regression: the find_center beamstop used to go through mask_draw, so
    the next draw apply (which replaces mask_draw's previous contribution)
    silently unmasked it."""
    shape = (10, 10)
    ma = MaskAssemble(shape=shape, saxs_lin=np.zeros(shape))
    ma.evaluate("mask_beamstop", arr=_disk(shape, (5, 5), 1.5))
    ma.apply("mask_beamstop")

    roi = RoiPolygon(rectangle_vertices(center=(1, 1), size=(2, 2)), "exclusive")
    ma.evaluate("mask_draw", **_draw_kwargs(shape, [roi]))
    mask = ma.apply("mask_draw")

    assert not mask[5, 5]  # beamstop still masked
    assert not mask[1, 1]  # draw exclusion applied


def test_reapplying_beamstop_replaces_the_previous_one():
    """Re-running find_center moves the beam center; the old beamstop disk
    must not stay masked at the stale position."""
    shape = (10, 10)
    ma = MaskAssemble(shape=shape, saxs_lin=np.zeros(shape))
    ma.evaluate("mask_beamstop", arr=_disk(shape, (2, 2), 1.5))
    ma.apply("mask_beamstop")
    ma.evaluate("mask_beamstop", arr=_disk(shape, (7, 7), 1.5))
    mask = ma.apply("mask_beamstop")

    assert mask[2, 2]
    assert not mask[7, 7]


def test_reapplying_mask_parameter_with_added_or_row_includes_new_range():
    """Regression: the constraint table persists after apply, so adding an
    OR row and re-applying must widen the mask, not AND against the old one."""
    shape = (1, 20)
    qmap = {"q": np.arange(20.0).reshape(1, 20)}
    ma = MaskAssemble(shape=shape, saxs_lin=np.zeros(shape), qmap=qmap)
    ma.evaluate("mask_parameter", constraints=[("q", "AND", "A^-1", 0, 5)])
    ma.apply("mask_parameter")

    ma.evaluate(
        "mask_parameter",
        constraints=[("q", "AND", "A^-1", 0, 5), ("q", "OR", "A^-1", 10, 15)],
    )
    mask = ma.apply("mask_parameter")

    assert mask[0, 3]
    assert mask[0, 12]
    assert not mask[0, 8]


def test_reapply_after_undo_keeps_pointer_and_mask_in_sync():
    """Regression: apply() compared against mask_record[-1] instead of the
    current entry, so re-applying the undone mask returned it without moving
    mask_ptr and get_mask() disagreed with the returned mask."""
    shape = (10, 10)
    ma = MaskAssemble(shape=shape, saxs_lin=np.zeros(shape))
    roi = RoiPolygon(rectangle_vertices(center=(2, 2), size=(3, 3)), "exclusive")
    ma.evaluate("mask_draw", **_draw_kwargs(shape, [roi]))
    ma.apply("mask_draw")
    ma.redo_undo("undo")

    mask = ma.apply("mask_draw")

    np.testing.assert_array_equal(ma.get_mask(), mask)
    assert not mask[2, 2]
