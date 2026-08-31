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


def test_group_index_column_not_editable():
    model = XmapConstraintsTableModel()
    model.addRow(["q", "AND", "A^-1", 0.0, 1.0])

    group_index_flags = model.flags(model.index(0, 0))
    data_flags = model.flags(model.index(0, 1))

    assert not (group_index_flags & Qt.ItemIsEditable)
    assert data_flags & Qt.ItemIsEditable


def test_add_row_still_validated_against_constraint_columns_only():
    model = XmapConstraintsTableModel()
    # 5 values (no group-index) is the valid shape for addRow
    model.addRow(["q", "AND", "A^-1", 0.0, 1.0])
    assert model.rowCount() == 1

    # wrong shape is ignored
    model.addRow(["q", "AND", "A^-1", 0.0])
    assert model.rowCount() == 1
