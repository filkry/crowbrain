# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'clustering_app.ui'
#
# Created: Wed Sep  4 12:29:07 2013
#      by: pyside-uic 0.2.14 running on PySide 1.2.0
#
# WARNING! All changes made in this file will be lost!

from PySide import QtCore, QtGui

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(840, 755)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout_2.addWidget(self.label)
        self.combo_box_data_set = QtGui.QComboBox(self.centralwidget)
        self.combo_box_data_set.setObjectName("combo_box_data_set")
        self.horizontalLayout_2.addWidget(self.combo_box_data_set)
        self.horizontalLayout_2.setStretch(0, 1)
        self.horizontalLayout_2.setStretch(1, 1)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout_4 = QtGui.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.list_ideas = QtGui.QListView(self.centralwidget)
        self.list_ideas.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.list_ideas.setObjectName("list_ideas")
        self.verticalLayout_4.addWidget(self.list_ideas)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.button_sort_by_list = QtGui.QPushButton(self.centralwidget)
        self.button_sort_by_list.setObjectName("button_sort_by_list")
        self.horizontalLayout_5.addWidget(self.button_sort_by_list)
        self.button_move_down = QtGui.QPushButton(self.centralwidget)
        self.button_move_down.setObjectName("button_move_down")
        self.horizontalLayout_5.addWidget(self.button_move_down)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.button_resolve_lost = QtGui.QPushButton(self.centralwidget)
        self.button_resolve_lost.setObjectName("button_resolve_lost")
        self.horizontalLayout_8.addWidget(self.button_resolve_lost)
        self.verticalLayout_4.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_3.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtGui.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.tree_main = QtGui.QTreeView(self.centralwidget)
        self.tree_main.setObjectName("tree_main")
        self.verticalLayout_5.addWidget(self.tree_main)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.line_regex = QtGui.QLineEdit(self.centralwidget)
        self.line_regex.setObjectName("line_regex")
        self.horizontalLayout.addWidget(self.line_regex)
        self.button_next_regex = QtGui.QPushButton(self.centralwidget)
        self.button_next_regex.setObjectName("button_next_regex")
        self.horizontalLayout.addWidget(self.button_next_regex)
        self.verticalLayout_5.addLayout(self.horizontalLayout)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.button_move_up = QtGui.QPushButton(self.centralwidget)
        self.button_move_up.setObjectName("button_move_up")
        self.horizontalLayout_7.addWidget(self.button_move_up)
        self.btn_import = QtGui.QPushButton(self.centralwidget)
        self.btn_import.setObjectName("btn_import")
        self.horizontalLayout_7.addWidget(self.btn_import)
        self.verticalLayout_5.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_3.addLayout(self.verticalLayout_5)
        self.verticalLayout_6 = QtGui.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_3.addLayout(self.verticalLayout_6)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.verticalLayout.addLayout(self.verticalLayout_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 840, 23))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtGui.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtGui.QToolBar(MainWindow)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.menu_item_save = QtGui.QAction(MainWindow)
        self.menu_item_save.setObjectName("menu_item_save")
        self.menu_item_export = QtGui.QAction(MainWindow)
        self.menu_item_export.setObjectName("menu_item_export")
        self.menu_item_quit = QtGui.QAction(MainWindow)
        self.menu_item_quit.setObjectName("menu_item_quit")
        self.menuFile.addAction(self.menu_item_save)
        self.menuFile.addAction(self.menu_item_export)
        self.menuFile.addAction(self.menu_item_quit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QtGui.QApplication.translate("MainWindow", "MainWindow", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("MainWindow", "Data set:", None, QtGui.QApplication.UnicodeUTF8))
        self.button_sort_by_list.setText(QtGui.QApplication.translate("MainWindow", "Sort by Selection", None, QtGui.QApplication.UnicodeUTF8))
        self.button_move_down.setText(QtGui.QApplication.translate("MainWindow", "Add Selected", None, QtGui.QApplication.UnicodeUTF8))
        self.button_resolve_lost.setText(QtGui.QApplication.translate("MainWindow", "Resolve Lost Ideas", None, QtGui.QApplication.UnicodeUTF8))
        self.button_next_regex.setText(QtGui.QApplication.translate("MainWindow", "Next", None, QtGui.QApplication.UnicodeUTF8))
        self.button_move_up.setText(QtGui.QApplication.translate("MainWindow", "Remove Selected", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_import.setText(QtGui.QApplication.translate("MainWindow", "Import From Text", None, QtGui.QApplication.UnicodeUTF8))
        self.menuFile.setTitle(QtGui.QApplication.translate("MainWindow", "File", None, QtGui.QApplication.UnicodeUTF8))
        self.toolBar.setWindowTitle(QtGui.QApplication.translate("MainWindow", "toolBar", None, QtGui.QApplication.UnicodeUTF8))
        self.menu_item_save.setText(QtGui.QApplication.translate("MainWindow", "&Save", None, QtGui.QApplication.UnicodeUTF8))
        self.menu_item_export.setText(QtGui.QApplication.translate("MainWindow", "Export Clusters", None, QtGui.QApplication.UnicodeUTF8))
        self.menu_item_quit.setText(QtGui.QApplication.translate("MainWindow", "Quit", None, QtGui.QApplication.UnicodeUTF8))

