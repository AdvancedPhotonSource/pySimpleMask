from PySide6 import QtCore
from PySide6.QtCore import QModelIndex, Qt


class XmapConstraintsTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self._data = data or []  # Renamed from `self.data` to `self._data`
        self.headers = ('map_name', 'unit', 'val_begin', 'val_end', 'logic')

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        """Returns the headers for the table."""

        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                if section < len(self.headers):
                    return self.headers[section]
            else:
                return f"constraint_{section + 1}"
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
            return str(self._data[row][col])  # Formatting numbers

        return None  # Fix for unsupported roles

    def addRow(self, row_data):
        """Adds a new row to the model."""
        if not row_data or len(row_data) != self.columnCount():
            return  # Ignore invalid row data

        row_index = self.rowCount()
        self.beginInsertRows(QtCore.QModelIndex(), row_index, row_index)
        self._data.append(row_data)
        self.endInsertRows()

    def flags(self, index):
        """Makes the table editable."""
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsSelectable | Qt.ItemIsEnabled | Qt.ItemIsEditable

    # def setData(self, index, value, role=Qt.EditRole):
    #     """Allows editing of cells."""
    #     if index.isValid() and role == Qt.EditRole:
    #         self._data[index.row()][index.column()] = value
    #         self.dataChanged.emit(index, index, [Qt.DisplayRole])
    #         return True
    #     return False

    def removeRow(self, row):
            """Remove a row from the table."""
            if 0 <= row < len(self._data):
                self.beginRemoveRows(self.index(row, 0), row, row)
                del self._data[row]  # Remove the row
                self.endRemoveRows()
                self.layoutChanged.emit() 