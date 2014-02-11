# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'parent_chooser.ui'
#
# Created: Thu Jan 23 17:10:51 2014
#      by: pyside-uic 0.2.14 running on PySide 1.2.1
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(438, 469)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        self.btn_change_parent = QtGui.QPushButton(Dialog)
        self.btn_change_parent.setGeometry(QtCore.QRect(10, 420, 211, 41))
        self.btn_change_parent.setObjectName("btn_change_parent")
        self.btn_cancel = QtGui.QPushButton(Dialog)
        self.btn_cancel.setGeometry(QtCore.QRect(230, 420, 201, 41))
        self.btn_cancel.setObjectName("btn_cancel")
        self.tree_options = QtGui.QTreeView(Dialog)
        self.tree_options.setGeometry(QtCore.QRect(10, 10, 421, 401))
        self.tree_options.setObjectName("tree_options")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Change Parent", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_change_parent.setText(QtGui.QApplication.translate("Dialog", "Change Parent", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_cancel.setText(QtGui.QApplication.translate("Dialog", "Cancel", None, QtGui.QApplication.UnicodeUTF8))

