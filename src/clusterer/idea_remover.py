# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'idea_remover.ui'
#
# Created: Thu Jan 30 11:47:38 2014
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
        self.btn_remove_ideas = QtGui.QPushButton(Dialog)
        self.btn_remove_ideas.setGeometry(QtCore.QRect(10, 420, 211, 41))
        self.btn_remove_ideas.setObjectName("btn_remove_ideas")
        self.btn_cancel = QtGui.QPushButton(Dialog)
        self.btn_cancel.setGeometry(QtCore.QRect(230, 420, 201, 41))
        self.btn_cancel.setObjectName("btn_cancel")
        self.lst_node_ideas = QtGui.QListView(Dialog)
        self.lst_node_ideas.setGeometry(QtCore.QRect(10, 10, 421, 401))
        self.lst_node_ideas.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.lst_node_ideas.setObjectName("lst_node_ideas")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QtGui.QApplication.translate("Dialog", "Change Parent", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_remove_ideas.setText(QtGui.QApplication.translate("Dialog", "Remove Selections", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_cancel.setText(QtGui.QApplication.translate("Dialog", "Cancel", None, QtGui.QApplication.UnicodeUTF8))

