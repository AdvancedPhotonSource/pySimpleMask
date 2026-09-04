# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Unit tests for XmapConstraintsTableModel (no QApplication needed)."""
from PySide6.QtCore import Qt

from pysimplemask.gui.model.table_model import (
    DrawGroupTableModel,
    XmapConstraintsTableModel,
)


def test_group_index_is_first_column_header():
    model = XmapConstraintsTableModel()
    assert model.headers[0] == "group-index"
    assert model.headers[1:] == ("map_name", "logic", "unit", "val_begin", "val_end")
    assert model.columnCount() == 6


def test_group_index_reflects_one_based_row_position():
    model = XmapConstraintsTableModel()
    model.addRow(["q", "AND", "A^-1", 0.0, 1.0])
    model.addRow(["phi", "OR", "deg", -10.0, 10.0])
    model.addRow(["chi", "AND", "deg", 0.0, 90.0])

    for row in range(model.rowCount()):
        idx = model.index(row, 0)
        assert model.data(idx, Qt.DisplayRole) == str(row + 1)

    # the underlying constraint data (used by core.mask.MaskParameter) is untouched
    assert model._data[0] == ["q", "AND", "A^-1", 0.0, 1.0]


def test_group_index_renumbers_after_row_removal():
    model = XmapConstraintsTableModel()
    model.addRow(["q", "AND", "A^-1", 0.0, 1.0])
    model.addRow(["phi", "OR", "deg", -10.0, 10.0])
    model.addRow(["chi", "AND", "deg", 0.0, 90.0])

    model.removeRow(0)

    assert model.data(model.index(0, 0), Qt.DisplayRole) == "1"
    assert model.data(model.index(1, 0), Qt.DisplayRole) == "2"
    assert model.data(model.index(0, 1), Qt.DisplayRole) == "phi"


def test_table_is_read_only():
    model = XmapConstraintsTableModel()
    model.addRow(["q", "AND", "A^-1", 0.0, 1.0])

    for col in range(model.columnCount()):
        flags = model.flags(model.index(0, col))
        assert flags & Qt.ItemIsEnabled
        assert not (flags & Qt.ItemIsEditable)


def test_float_values_display_fixed_point_with_seven_digits():
    model = XmapConstraintsTableModel()
    model.addRow(["q", "AND", "A^-1", 1.2e-6, 0.1])

    vbeg = model.data(model.index(0, 4), Qt.DisplayRole)
    vend = model.data(model.index(0, 5), Qt.DisplayRole)

    assert vbeg == "0.0000012"
    assert vend == "0.1000000"
    assert "e" not in vbeg.lower()


def test_add_row_still_validated_against_constraint_columns_only():
    model = XmapConstraintsTableModel()
    # 5 values (no group-index) is the valid shape for addRow
    model.addRow(["q", "AND", "A^-1", 0.0, 1.0])
    assert model.rowCount() == 1

    # wrong shape is ignored
    model.addRow(["q", "AND", "A^-1", 0.0])
    assert model.rowCount() == 1


def test_draw_group_table_headers():
    model = DrawGroupTableModel()
    assert model.headers == ("group-index", "shape", "type", "center (row, col)")
    assert model.columnCount() == 4


def test_draw_group_table_group_index_reflects_row_position():
    model = DrawGroupTableModel()
    model.addRow("roi_000000", "Circle", (10.0, 20.0), "inclusive")
    model.addRow("roi_000001", "Polygon", (30.0, 40.0), "inclusive")

    assert model.data(model.index(0, 0), Qt.DisplayRole) == "1"
    assert model.data(model.index(1, 0), Qt.DisplayRole) == "2"
    assert model.data(model.index(0, 1), Qt.DisplayRole) == "Circle"
    assert model.data(model.index(1, 3), Qt.DisplayRole) == "(30.0, 40.0)"


def test_draw_group_table_type_column_shows_mode():
    model = DrawGroupTableModel()
    model.addRow("roi_000000", "Circle", (0.0, 0.0), "inclusive")
    model.addRow("roi_000001", "Rectangle", (0.0, 0.0), "exclusive")

    assert model.data(model.index(0, 2), Qt.DisplayRole) == "inclusive"
    assert model.data(model.index(1, 2), Qt.DisplayRole) == "exclusive"


def test_draw_group_table_update_center_updates_matching_row():
    model = DrawGroupTableModel()
    model.addRow("roi_000000", "Circle", (0.0, 0.0), "inclusive")
    model.addRow("roi_000001", "Polygon", (1.0, 1.0), "inclusive")

    model.updateCenter("roi_000001", (5.0, 6.0))

    assert model.data(model.index(0, 3), Qt.DisplayRole) == "(0.0, 0.0)"
    assert model.data(model.index(1, 3), Qt.DisplayRole) == "(5.0, 6.0)"


def test_draw_group_table_update_center_unknown_key_is_noop():
    model = DrawGroupTableModel()
    model.addRow("roi_000000", "Circle", (0.0, 0.0), "inclusive")

    model.updateCenter("roi_does_not_exist", (9.0, 9.0))

    assert model.data(model.index(0, 3), Qt.DisplayRole) == "(0.0, 0.0)"


def test_draw_group_table_exclusive_row_group_index_is_zero():
    model = DrawGroupTableModel()
    model.addRow("roi_000000", "Rectangle", (0.0, 0.0), "exclusive")
    assert model.data(model.index(0, 0), Qt.DisplayRole) == "0"


def test_draw_group_table_inclusive_numbering_skips_interleaved_exclusive_rows():
    """group-index counts only inclusive rows; exclusive rows always show 0
    and don't consume a number."""
    model = DrawGroupTableModel()
    model.addRow("roi_000000", "Rectangle", (0.0, 0.0), "exclusive")
    model.addRow("roi_000001", "Circle", (1.0, 1.0), "inclusive")
    model.addRow("roi_000002", "Rectangle", (2.0, 2.0), "exclusive")
    model.addRow("roi_000003", "Circle", (3.0, 3.0), "inclusive")

    assert [model.data(model.index(r, 0), Qt.DisplayRole) for r in range(4)] == [
        "0", "1", "0", "2",
    ]


def test_draw_group_table_inclusive_roi_keys_excludes_exclusive_rows():
    model = DrawGroupTableModel()
    model.addRow("roi_000000", "Rectangle", (0.0, 0.0), "exclusive")
    model.addRow("roi_000001", "Circle", (1.0, 1.0), "inclusive")
    model.addRow("roi_000002", "Circle", (2.0, 2.0), "inclusive")

    assert model.inclusive_roi_keys() == ["roi_000001", "roi_000002"]


def test_draw_group_table_roi_keys_in_row_order():
    model = DrawGroupTableModel()
    model.addRow("roi_000000", "Circle", (0.0, 0.0), "inclusive")
    model.addRow("roi_000001", "Polygon", (0.0, 0.0), "exclusive")
    assert model.roi_keys() == ["roi_000000", "roi_000001"]


def test_draw_group_table_remove_roi_key_renumbers_remaining_rows():
    model = DrawGroupTableModel()
    model.addRow("roi_000000", "Circle", (0.0, 0.0), "inclusive")
    model.addRow("roi_000001", "Polygon", (1.0, 1.0), "inclusive")
    model.addRow("roi_000002", "Rectangle", (2.0, 2.0), "inclusive")

    model.removeRoiKey("roi_000001")

    assert model.roi_keys() == ["roi_000000", "roi_000002"]
    assert model.data(model.index(0, 0), Qt.DisplayRole) == "1"
    assert model.data(model.index(1, 0), Qt.DisplayRole) == "2"
    assert model.data(model.index(1, 1), Qt.DisplayRole) == "Rectangle"


def test_draw_group_table_remove_unknown_roi_key_is_noop():
    model = DrawGroupTableModel()
    model.addRow("roi_000000", "Circle", (0.0, 0.0), "inclusive")
    model.removeRoiKey("roi_does_not_exist")
    assert model.rowCount() == 1


def test_draw_group_table_clear():
    model = DrawGroupTableModel()
    model.addRow("roi_000000", "Circle", (0.0, 0.0), "inclusive")
    model.clear()
    assert model.rowCount() == 0
    assert model.roi_keys() == []


def test_draw_group_table_is_read_only():
    model = DrawGroupTableModel()
    model.addRow("roi_000000", "Circle", (0.0, 0.0), "inclusive")
    for col in range(model.columnCount()):
        flags = model.flags(model.index(0, col))
        assert flags & Qt.ItemIsEnabled
        assert not (flags & Qt.ItemIsEditable)
