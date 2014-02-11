from PySide import QtCore

class IdeaTreeNodeListModel(QtCore.QAbstractListModel):
  def __init__(self, itn, parent=None):
    super(IdeaTreeNodeListModel, self).__init__(parent)
    self.node = itn 
 
# QAbstractListModel required methods
  def rowCount(self, parent):
    return len(self.node.ideas)
  def data(self, index, role=QtCore.Qt.DisplayRole):
    row = index.row()
    if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.ToolTipRole:
      return self.node.ideas[row][0]
#    if role == QtCore.Qt.TextAlignmentRole and self.alignment_data:
#      return self.alignment_data[col]
    return None
# End QAbstractListModel required methods


