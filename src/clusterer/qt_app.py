#!/usr/bin/python
import os
import sys
import json
import re
import string
from collections import deque, defaultdict

# UI imports
from PySide import QtCore, QtGui
import clustering_app
import import_cluster_text
import similarity_selector
import change_parent
import idea_remover
import coverage
import title
import IdeaTreeModel, IdeaListModel, IdeaTreeNodeListModel

db_file_name = '/home/crowdbrainstorm/Desktop/clusters.db'

def coverage_prompt(node1, node2):
    dialog = QtGui.QDialog()
    dialog.ui = coverage.Ui_Dialog()
    dialog.ui.setupUi(dialog)

    dialog.ui.lbl_idea1.setText(node1.long_label())
    dialog.ui.lbl_idea2.setText(node2.long_label())
    dialog.ui.lbl_idea1.setToolTip(node1.tooltip())
    dialog.ui.lbl_idea2.setToolTip(node2.tooltip())

    dialog.ui.btn_both_high.clicked.connect(lambda x=0: dialog.done(x))
    dialog.ui.btn_both_low.clicked.connect(lambda x=1: dialog.done(x))
    dialog.ui.btn_12.clicked.connect(lambda x=2: dialog.done(x))
    dialog.ui.btn_21.clicked.connect(lambda x=3: dialog.done(x))

    ret = dialog.exec()

    if ret == 0:
      return True, True
    elif ret == 1:
      return False, False
    elif ret == 2:
      return True, False
    elif ret == 3:
      return False, True

def similarity_prompt(node, compare_root, idea_model):
    scores = []

    if len(compare_root.children) == 0:
      return None

    nis = node.get_all_ideas()

    for c in compare_root.all_child_nodes():
      cis = c.get_all_ideas()
      if len(cis) == 0:
        print("No idea under", c.label())
      s = idea_model.mean_similarity([i[1] for i in nis],
                                          [i[1] for i in cis])
      scores.append((c, s))

    top = sorted(scores, key=lambda s: s[1], reverse = True)[:5]

    dialog = QtGui.QDialog()
    dialog.ui = similarity_selector.Ui_Dialog()
    dialog.ui.setupUi(dialog)

    top_btns = [dialog.ui.btn_top1, dialog.ui.btn_top2, dialog.ui.btn_top3,
                dialog.ui.btn_top4, dialog.ui.btn_top5]

    dialog.ui.label_idea.setText(node.long_label())

    for i, (node, score) in enumerate(top):
      btn = top_btns[i]
      btn.setText(node.long_label())
      btn.setToolTip(node.tooltip())
      btn.clicked.connect(lambda x=i: dialog.done(x))

    if len(top) < len(top_btns):
      for btn in top_btns[len(top):]:
        btn.setEnabled(False)

    itm = IdeaTreeModel.IdeaTreeModel(idea_model.cur_question_code, db_file_name, root = compare_root)
    dialog.ui.tree_options.setModel(itm)

    dialog.ui.btn_fromlist.clicked.connect(lambda x=5: dialog.done(x))
    dialog.ui.btn_none.clicked.connect(lambda x=6: dialog.done(x))

    ret = dialog.exec()

    if ret <= 4:
      return top[ret][0]
    elif ret == 5:
      sel = dialog.ui.tree_options.selectedIndexes()
      for index in sel:
        if index.isValid():
          return index.internalPointer()
      assert(False)
    elif ret == 6:
      return None

    return None

def select_node_prompt(itm):
    dialog = QtGui.QDialog()
    dialog.ui = change_parent.Ui_Dialog()
    dialog.ui.setupUi(dialog)

    dialog.ui.tree_options.setModel(itm)

    dialog.ui.btn_change_parent.clicked.connect(lambda x=0: dialog.done(x))
    dialog.ui.btn_cancel.clicked.connect(lambda x=1: dialog.done(x))

    ret = dialog.exec()

    if ret == 0:
      sel = dialog.ui.tree_options.selectedIndexes()
      if len(sel) > 0:
        index = sel[0]
        return index
    return None

def select_ideas_prompt(node):
    dialog = QtGui.QDialog()
    dialog.ui = idea_remover.Ui_Dialog()
    dialog.ui.setupUi(dialog)

    dialog.ui.lst_node_ideas.setModel(IdeaTreeNodeListModel.IdeaTreeNodeListModel(node))

    dialog.ui.btn_remove_ideas.clicked.connect(lambda x=0: dialog.done(x))
    dialog.ui.btn_cancel.clicked.connect(lambda x=1: dialog.done(x))

    ret = dialog.exec()

    if ret == 0:
      sels = dialog.ui.lst_node_ideas.selectedIndexes()
      rows = [index.row() for index in sels if index.isValid()]
      remove_ids = [node.ideas[row][1] for row in rows]
      return remove_ids
    return []

def cluster_label_prompt(node):
    dialog = QtGui.QDialog()
    dialog.ui = title.Ui_Dialog()
    dialog.ui.setupUi(dialog)

    if node._label is not None:
      dialog.ui.lineEdit.insert(node._label)

    dialog.ui.btn_ok.clicked.connect(lambda x=0: dialog.done(x))
    dialog.ui.btn_none.clicked.connect(lambda x=1: dialog.done(x))

    dialog.ui.text_ideas.document().setPlainText(node.tooltip())

    ret = dialog.exec()

    if ret == 0:
      return dialog.ui.lineEdit.text()
    return node._label


class AppWindow(QtGui.QMainWindow):
  def __init__(self, parent=None):
    super(AppWindow, self).__init__(parent)
    self.ui = clustering_app.Ui_MainWindow()
    self.ui.setupUi(self)
    self.idea_model = IdeaListModel.IdeaListModel(db_file_name)

    self.idea_tree_models = dict()
    for qc in self.idea_model.get_question_codes():
      self.idea_tree_models[qc] = IdeaTreeModel.IdeaTreeModel(qc, db_file_name)

    self._finish_ui()
    self.regex_matches = []

  def _finish_ui(self):
    # Connect up all the buttons
    self.ui.button_move_down.clicked.connect(self.handle_move_selection_down)
    self.ui.button_rename.clicked.connect(self.handle_button_rename)
    self.ui.button_remove_ideas.clicked.connect(self.handle_remove_ideas)
    self.ui.button_add_parent.clicked.connect(self.handle_add_parent)
    self.ui.button_change_parent.clicked.connect(self.handle_change_parent)
    self.ui.button_move_up.clicked.connect(self.handle_move_selection_up)
    self.ui.button_sort_by_list.clicked.connect(self.handle_sort_by_list_selection)
    self.ui.button_next_regex.clicked.connect(self.handle_next_regex)
    self.ui.button_resolve_lost.clicked.connect(self.handle_resolve_lost)
    self.ui.btn_import.clicked.connect(self.handle_import)

    # Connect menu items
    self.ui.menu_item_save.triggered.connect(self.save_clusters)
    self.ui.menu_item_export.triggered.connect(self.handle_export)
    self.ui.menu_item_quit.triggered.connect(self.close_app)

    # Init list
    self.ui.list_ideas.setModel(self.idea_model)

    # Regex textbox
    self.ui.line_regex.textChanged.connect(self.clear_regex)

    # Add entries to the combo box
    self.ui.combo_box_data_set.addItems(self.idea_model.get_question_codes())
    self.ui.combo_box_data_set.currentIndexChanged.connect(self.handle_combo_box_changed)

    # Set/change models
    qc = self.ui.combo_box_data_set.currentText()
    self.idea_model.set_question_code(qc)
    self.ui.tree_main.setModel(self.idea_tree_models[qc])

    self.idea_model.modelReset.connect(lambda: self.update_remaining())

  def update_remaining(self):
    self.ui.lbl_remaining.setText(str(self.idea_model.rowCount(None)) + " remaining")

  def handle_resolve_lost(self):
    qc = self.idea_model.cur_question_code
    self.idea_model.resolve(self.idea_tree_models[qc])

  def clear_regex(self):
    self.regex_matches = []

  def handle_next_regex(self):
    if not self.regex_matches:
      qc = self.idea_model.cur_question_code
      itm = self.idea_tree_models[qc]
      nodes = itm.root.all_child_nodes()

      test = re.compile(self.ui.line_regex.text(), re.IGNORECASE)

      self.regex_matches = [node for node in nodes
                                 for (idea, i, time) in node.ideas
                                 if test.match(idea) is not None]

      # convenience hack; if no results, add .* and .* to end and beginning

      if len(self.regex_matches) == 0:
        test = re.compile(".*" + self.ui.line_regex.text() + ".*", re.IGNORECASE)
        self.regex_matches = [node for node in nodes
                                   for (idea, i, time) in node.ideas
                                   if test.match(idea) is not None]

      print("number of matches: %i" % len(self.regex_matches))

    if self.regex_matches:
      self.highlight_node(self.regex_matches[0])
      del self.regex_matches[0]
    
  def showEvent(self, e):
    return super(AppWindow, self).showEvent(e)
  def closeEvent(self, e):
    self.save_clusters()
    return super(AppWindow, self).closeEvent(e)

  def handle_import(self):
    dialog = QtGui.QDialog()
    dialog.ui = import_cluster_text.Ui_Dialog()
    dialog.ui.setupUi(dialog)

    dialog.exec()

    text = dialog.ui.text_input.document().toPlainText()

    qc = self.idea_model.cur_question_code
    self.idea_tree_models[qc].import_from_text(text, self.idea_model)

  def handle_move_selection_down(self):
    qc = self.idea_model.cur_question_code
    itm = self.idea_tree_models[qc]

    indexes = [i.row() for i in self.ui.list_ideas.selectedIndexes()]
    ideas = self.idea_model.get_ideas_for_indices(indexes)

    final_node = itm.add_ideas(ideas, cluster_label_prompt, similarity_prompt,
	coverage_prompt, self.idea_model)

    self.highlight_node(final_node)
    self.idea_model.make_ideas_used([i[0] for i in ideas])

  def highlight_node(self, node):
    qc = self.idea_model.cur_question_code
    itm = self.idea_tree_models[qc]

    index = itm.createIndex(node.row(), 0, node)
    self.ui.tree_main.scrollTo(index)
    self.ui.tree_main.selectionModel().setCurrentIndex(index,
      QtGui.QItemSelectionModel.Select)
    
  def handle_button_rename(self):
    sel = self.ui.tree_main.selectedIndexes()
    if len(sel) > 0:
      index = sel[0]
      qc = self.idea_model.cur_question_code
      itm = self.idea_tree_models[qc]
      itm.change_node_label(index, cluster_label_prompt)

  def handle_remove_ideas(self):
    sel = self.ui.tree_main.selectedIndexes()
    indices = [index for index in self.ui.tree_main.selectedIndexes() if index.isValid()]
    if len(indices) > 0:
        index = indices[0]

        qc = self.idea_model.cur_question_code
        itm = self.idea_tree_models[qc]
        removed_ids = itm.remove_ideas(index, select_ideas_prompt)
        self.idea_model.make_ideas_unused(removed_ids)

  def handle_add_parent(self):
    sel = self.ui.tree_main.selectedIndexes()
    indices = [index for index in self.ui.tree_main.selectedIndexes() if index.isValid()]
    if len(indices) > 0:
        qc = self.idea_model.cur_question_code
        itm = self.idea_tree_models[qc]
        itm.add_parent(indices, cluster_label_prompt)

  def handle_change_parent(self):
    sel = self.ui.tree_main.selectedIndexes()
    indices = [index for index in self.ui.tree_main.selectedIndexes() if index.isValid()]
    if len(indices) > 0:
        qc = self.idea_model.cur_question_code
        itm = self.idea_tree_models[qc]
        itm.change_parent(indices, select_node_prompt)

  def handle_move_selection_up(self):
    btn = QtGui.QMessageBox.question(self, 'Confirmation', 
            'Really remove this node?',
            buttons = QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

    if btn == QtGui.QMessageBox.Yes:
        sel = self.ui.tree_main.selectedIndexes()
        qc = self.idea_model.cur_question_code
        itm = self.idea_tree_models[qc]
        for index in sel:
            ids = itm.remove_node_reparent_children(index)
            self.idea_model.make_ideas_unused(ids)

  def handle_sort_by_list_selection(self):
    selected_indexes = sorted([i.row() for i in self.ui.list_ideas.selectedIndexes()])
    if selected_indexes:
      self.idea_model.sort_based_on_similarity(self.idea_model.cur_ideas[selected_indexes[0]][0])
      self.ui.list_ideas.scrollToTop()

  def handle_find_similar(self):
    selected_indexes = sorted([i.row() for i in self.ui.list_ideas.selectedIndexes()])
    previously_selected_idea_id = -1
    if selected_indexes:
      previously_selected_idea_id = self.idea_model.cur_ideas[selected_indexes[0]][0]
    selection_cursor = self.ui.text_edit_clusters.textCursor()
    selection_cursor.movePosition(QtGui.QTextCursor.StartOfLine)
    selection_cursor.movePosition(QtGui.QTextCursor.EndOfLine, QtGui.QTextCursor.KeepAnchor)
    self.ui.text_edit_clusters.setTextCursor(selection_cursor)
    selected_text = self.ui.text_edit_clusters.textCursor().selectedText()
    matches = self.idea_model.extract_idea_ids_from_text(selected_text)
    if matches:
      idea_id = matches[0]
      self.idea_model.sort_based_on_similarity(idea_id)
      self.ui.list_ideas.scrollToTop()
    if previously_selected_idea_id > 0:
      cur_id_list = [i[0] for i in self.idea_model.cur_ideas]
      if previously_selected_idea_id in cur_id_list:
        row_num = cur_id_list.index(previously_selected_idea_id)
        self.ui.list_ideas.selectionModel().select(self.ui.list_ideas.model().createIndex(row_num,0), QtGui.QItemSelectionModel.ClearAndSelect)

  def handle_export(self):
    output_folder = 'out/'

    for question_code in self.idea_model.get_question_codes():
      itm = self.idea_tree_models[question_code]
      itm.export_clusters(output_folder)
      itm.export_clusters_text(output_folder)


  def handle_combo_box_changed(self):
    self.save_clusters()
    qc = self.ui.combo_box_data_set.currentText()
    self.idea_model.set_question_code(qc)
    self.ui.tree_main.setModel(self.idea_tree_models[qc])

  def save_clusters(self):
    for qc in self.idea_model.get_question_codes():
      self.idea_tree_models[qc].save()

  def close_app(self):
    self.save_clusters()
    app.closeAllWindows()

app = None
app_window = None
def run_gui():
  global app, app_window
  app = QtGui.QApplication(sys.argv)
  app_window = AppWindow()
  app_window.show()
  sys.exit(app.exec_())

run_gui()
handle_find_regex
