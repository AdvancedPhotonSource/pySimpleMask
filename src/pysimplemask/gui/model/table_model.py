# Copyright © UChicago Argonne LLC
# See LICENSE file for details
from PySide6 import QtCore
from PySide6.QtCore import QModelIndex, Qt


class XmapConstraintsTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self._data = data or []  # Renamed from `self.data` to `self._data`
        self.data_headers = ('map_name', 'logic', 'unit', 'val_begin', 'val_end')
        # group-index is derived from row position (1-based), not stored per-row,
        # so it stays correct after rows are added/removed/reordered.
        self.headers = ('group-index',) + self.data_headers

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        """Returns the headers for the table."""

        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if section < len(self.headers):
                    return self.headers[section]
            else:
                return f"mask_{section + 1}"
        return None

    def columnCount(self, parent=None):
        """Returns the number of columns."""
        return len(self.headers)

    def rowCount(self, parent=None):
        """Returns the number of rows."""
        return len(self._data)

    def data(self, index: QModelIndex, role: int):
        """Returns the data for display."""
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            row, col = index.row(), index.column()
            if col == 0:
                return str(row + 1)
            value = self._data[row][col - 1]
            if isinstance(value, float):
                return f"{value:.7f}"  # fixed-point; never scientific notation
            return str(value)

        return None  # Fix for unsupported roles

    def addRow(self, row_data):
        """Adds a new row to the model."""
        if not row_data or len(row_data) != len(self.data_headers):
            return  # Ignore invalid row data

        row_index = self.rowCount()
        self.beginInsertRows(QtCore.QModelIndex(), row_index, row_index)
        self._data.append(row_data)
        self.endInsertRows()

    def flags(self, index):
        """The table is read-only; rows are managed via addRow/removeRow."""
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def removeRow(self, row):
            """Remove a row from the table."""
            if 0 <= row < len(self._data):
                self.beginRemoveRows(self.index(row, 0), row, row)
                del self._data[row]  # Remove the row
                self.endRemoveRows()
                self.layoutChanged.emit() 
    
    def clear(self):
        """Clears all rows from the model."""
        if not self._data:
            return
        self.beginResetModel()  # Signals views to reset the model
        self._data.clear()
        self.endResetModel()  # Notifies views of the change


class DrawGroupTableModel(QtCore.QAbstractTableModel):
    """Tracks every draw ROI in the order 'Draw' was pressed (Draw tab's
    tableView_draw), inclusive and exclusive alike.

    group-index is derived from row position among inclusive rows only
    (1-based), same convention as XmapConstraintsTableModel, so it stays
    correct after rows are removed. Exclusive rows always show 0 — they are
    never part of a group and are never used for group-index sub-partitioning.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = []  # list of {"roi_key", "shape", "center", "mode"}
        self.data_headers = ("shape", "type", "center (row, col)")
        self.headers = ("group-index",) + self.data_headers

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if section < len(self.headers):
                    return self.headers[section]
            else:
                return str(section + 1)
        return None

    def columnCount(self, parent=None):
        return len(self.headers)

    def rowCount(self, parent=None):
        return len(self._data)

    def data(self, index: QModelIndex, role: int):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            row, col = index.row(), index.column()
            row_data = self._data[row]
            if col == 0:
                if row_data["mode"] != "inclusive":
                    return "0"
                return str(
                    sum(1 for r in self._data[: row + 1] if r["mode"] == "inclusive")
                )
            if col == 1:
                return row_data["shape"]
            if col == 2:
                return row_data["mode"]
            if col == 3:
                cr, cc = row_data["center"]
                return f"({cr:.1f}, {cc:.1f})"

        return None

    def addRow(self, roi_key, shape, center, mode):
        """Adds a new tracked ROI row. mode is "inclusive" or "exclusive";
        only inclusive rows get a nonzero group-index."""
        row_index = self.rowCount()
        self.beginInsertRows(QtCore.QModelIndex(), row_index, row_index)
        self._data.append(
            {"roi_key": roi_key, "shape": shape, "center": center, "mode": mode}
        )
        self.endInsertRows()

    def removeRow(self, row):
        if 0 <= row < len(self._data):
            self.beginRemoveRows(QtCore.QModelIndex(), row, row)
            del self._data[row]
            self.endRemoveRows()

    def updateCenter(self, roi_key, center):
        """Update the displayed center for roi_key's row, if tracked (no-op
        otherwise) — called when the user drags/resizes a live drawn ROI."""
        for i, row in enumerate(self._data):
            if row["roi_key"] == roi_key:
                row["center"] = center
                col = self.headers.index("center (row, col)")
                idx = self.index(i, col)
                self.dataChanged.emit(idx, idx, [Qt.DisplayRole])
                return

    def removeRoiKey(self, roi_key):
        """Remove the row tracking roi_key, if any (no-op otherwise)."""
        for i, row in enumerate(self._data):
            if row["roi_key"] == roi_key:
                self.removeRow(i)
                return

    def roi_keys(self):
        """All tracked roi_keys in row order, inclusive and exclusive alike."""
        return [row["roi_key"] for row in self._data]

    def inclusive_roi_keys(self):
        """Inclusive-only roi_keys in row order; inclusive_roi_keys()[i] is
        group-index i+1. Excludes exclusive rows (always group-index 0)."""
        return [row["roi_key"] for row in self._data if row["mode"] == "inclusive"]

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled

    def clear(self):
        if not self._data:
            return
        self.beginResetModel()
        self._data.clear()
        self.endResetModel()