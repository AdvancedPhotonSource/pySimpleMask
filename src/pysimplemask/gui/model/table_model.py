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