# Copyright © UChicago Argonne LLC
# See LICENSE file for details
"""Unit tests for XmapConstraintsTableModel (no QApplication needed)."""
from PySide6.QtCore import Qt

from pysimplemask.gui.model.table_model import XmapConstraintsTableModel


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
