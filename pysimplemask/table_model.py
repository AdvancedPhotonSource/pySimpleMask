from PyQt5 import QtCore
from PyQt5.QtCore import QModelIndex, Qt

# code copied from
# https://stackoverflow.com/questions/63012839

class QringTableModel(QtCore.QAbstractTableModel):
    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self.data = data

    def headerData(self, section: int, orientation: Qt.Orientation, role: int):
        if role == QtCore.Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return ('q_begin', 'q_end', 'phi_begin', 'phi_end')[section]
            else:
                return "roi_" + str(section)

    def columnCount(self, parent=None):
        return len(self.data[0])

    def rowCount(self, parent=None):
        return len(self.data)

    def data(self, index: QModelIndex, role: int):
        if role == QtCore.Qt.DisplayRole:
            row = index.row()
            col = index.column()
            return str(round(self.data[row][col], 5))
