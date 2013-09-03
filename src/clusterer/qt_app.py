#!/usr/bin/python
import os
import sys
import json
import re
import string
from PySide import QtCore, QtGui
import sqlite3 as db
import clustering_app
import import_cluster_text
import csv
from collections import deque, defaultdict

db_file_name = '/home/fil/Dropbox/clusters.db'

def get_line_indent(l):
  depth_count = 0
  l = l.strip()
  front = l
  while front and front[0] == '-':
    depth_count = depth_count + 1
    front = front[1:]
  return depth_count

def extract_idea_ids_from_text(s):
  matches = []
  for match in re.findall('\(_id:[0-9]+\)', s):
    matches.append(int(re.findall('[0-9]+', match)[0]))
  return matches

class IdeaTreeNode(object):
  def __init__(self, ideas, parent, label = None):
    assert not isinstance(ideas, str)
    self._label = label
    self.ideas = ideas # list of (text, id) tuples
    self.parent = parent
    self.children = []

  def merge(self, other_node):
    self.ideas.extend(other_node.ideas)

  def append_child(self, child_node):
    self.children.append(child_node)

  def child_count(self):
    return len(self.children)

  def child(self, row):
    return self.children[row]

  def label(self):
    if self._label:
      return self._label
    elif len(self.ideas) > 0:
      return self.ideas[0][0][:30]
    else:
      return "NO LABEL, NO IDEAS"

  def tooltip(self):
    return '\n'.join(i[0] for i in self.ideas)

  def row(self):
    if self.parent:
      return self.parent.children.index(self)
    return 0

  def get_parent(self):
    return self.parent

class IdeaTreeModel(QtCore.QAbstractItemModel):
  def __init__(self, question_code, parent=None):
    super(IdeaTreeModel, self).__init__(parent)
    self.root = IdeaTreeNode([], None, question_code)
    self.conn = db.connect(db_file_name)
    self.question_code = question_code

  def columnCount(self, parent = None):
    return 1

  def data(self, index, role):
    item = index.internalPointer()
    if role == QtCore.Qt.DisplayRole:
      return item.label()
    elif role == QtCore.Qt.ToolTipRole:
      return item.tooltip()
    return None

  def index(self, row, column, parent):
    p = parent.internalPointer() if parent.isValid() else self.root
    c = p.child(row)
    return self.createIndex(row, column, c)

  def parent(self, index):
    c = index.internalPointer()
    p = c.get_parent()
    if p == self.root:
      return QtCore.QModelIndex()

    return self.createIndex(p.row(), 0, p)

  def rowCount(self, parent=QtCore.QModelIndex()):
    p = parent.internalPointer() if parent.isValid() else self.root
    return p.child_count()

  def get_next_depth(self, dq):
    for l in dq:
      if len(l) == 0:
        continue
      return get_line_indent(l)

  # Please note that all this code is awful
  def _get_line_node(self, l, current_node, used_ids, idea_model):
    ids = extract_idea_ids_from_text(l)
    assert len(ids) <= 1

    if not ids:
      # Check if this is a label
      if len(l) > 0:
        return IdeaTreeNode([], current_node, l.strip('-'))
      # Otherwise this is a newline, close clusters up to level of depth
      else:
        return None

    idea, ids = idea_model.get_ids_for_text(l)
    new_ids = [i for i in ids if not i in used_ids]
    new_id = new_ids[0]
    used_ids.append(new_id)
    return IdeaTreeNode([(idea, new_id)], current_node, idea)

  def import_from_text(self, text, idea_model):
    self.beginResetModel()
    used_ids = []

    # Clear the root node
    self.root = IdeaTreeNode([], None, self.root.label())

    current_node = self.root
    current_depth = 0

    look_ahead = deque(text.splitlines())
    while len(look_ahead) > 0:
      line = look_ahead.popleft()
      next_node = self._get_line_node(line, current_node, used_ids, idea_model)

      if not next_node:
        next_depth = self.get_next_depth(look_ahead)
        while current_depth > next_depth - 1:
          current_node = current_node.get_parent()
          current_depth -= 1
      else:
        depth_count = get_line_indent(line)

        if depth_count == current_depth:
          current_node.merge(next_node)

        elif depth_count > current_depth:
          assert depth_count - 1 == current_depth
          current_node.append_child(next_node)
          current_node = next_node
          current_depth += 1

        elif depth_count < current_depth:
          while depth_count < current_depth:
            current_node = current_node.get_parent()
            current_depth -= 1

    self.endResetModel()

    def _save_node(self, node, cursor):
      # Create cluster
      cursor.execute("""INSERT INTO clusters(question_code, label)
                        VALUES (?, ?)""", (self.question_code, node.label()))
      clus_id = cursor.lastrowid

      # Map ideas
      for i, text in node.ideas:
        cursor.execute("""INSERT INTO idea_clusters(idea_id, cluster_id)
                          VALUES (?, ?)""", (i, clus_id))

      # Create children
      child_ids = [self._save_node(c, cursor) for c in node.children]
      
      # Create parent-child relationships
      for child_id in child_ids:
        cursor.execute("""INSERT INTO cluster_hierarchy(child, parent)
                          VALUES(?, ?)""", (child_id, clus_id))

      return clus_id

    def save(self):
      cursor = self.conn.cursor()
      cursor.execute("DELETE FROM clusters WHERE question_code=?")

      self._save_node(self.root, cursor)

      cursor.close()

    def load(self):
      self.beginResetModel()
      cursor = self.conn.cursor()

      cluster_dict = dict()

      # Load clusters
      cursor.execute("""SELECT id, label FROM clusters
                        WHERE question_code = ?""", (self.question_code,))
      for i, l in cursor.fetchall():
        node = IdeaTreeNode([], None, l)
        cluster_dict[i] = node

      # Organize into tree
      cursor.execute("""SELECT child, parent FROM cluster_hierarchy
                        WHERE child IN (SELECT id FROM clusters
                                        WHERE question_code = ?)""",
                     (self.question_code,))
      for c, p in cursor.fetchall():
        child = cluster_dict[c]
        parent = cluster_dict[p]

        child.parent = parent
        parent.append_child(child)

      # Load ideas into clusters
      cursor.execute("""SELECT cluster_id, id, idea
                        FROM ideas INNER JOIN idea_clusters
                        ON ideas.id = idea_clusters.idea_id
                        WHERE ideas.question_code = ?""",
                     (self.question_code,))
      for cid, iid, idea in cursor.fetchall():
        cluster_dict[cid].ideas.append((iid, idea))

      cursor.close()
      self.endResetModel()

    def export_clusters(self, filename):

      cursor = self.conn.cursor()
      with open("%s_%s_clusters.csv" % (filename, question_code), 'w') as cfout:
        cwriter = csv.writer(cfout)
        cwriter.writerow(['question_code', 'cluster', 'cluster_parent', 'cluster_label'])

        cursor.execute("""SELECT child, parent, label
                          FROM cluster_hierarchy INNER JOIN clusters
                          ON cluster_hierarchy.child = clusters.id
                          WHERE  clusters.question_code = ?""",
                       (self.question_code,))
        for c, p, l in cursor.fetchall():
          cwriter.writerow([self.question_code, c, p, l])

      with open("%s_%s.csv" % (filename, question_code), 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(['question_code', 'idea_id', 'cluster_id', 'idea', 'idea_num',
                         'worker_id', 'post_date', 'num_ideas_requested',])

        cursor.execute("""SELECT ideas.id, idea_clusters.cluster_id, idea, idea_num,
                                 worker_id, 'post_date', 'num_ideas_requested'
                          FROM ideas INNER JOIN idea_clusters
                          ON ideas.id = idea_clusters.idea_id
                          WHERE ideas.question_code = ?""",
                       (self.question_code,))
        for iid, cid, i, num, wid, date, nr in cursor.fetchall():
          writer.writerow([self.question_code, iid, cid, i, num, wid,
                           date, nr])
      
      cursor.close()
      print("export complete")



class IdeaListModel(QtCore.QAbstractListModel):
  def __init__(self, parent=None):
    super(IdeaListModel, self).__init__(parent)
    self.conn = db.connect(db_file_name)
    self.cur_question_code = ""
    self.cur_ideas = [] # Each entry: (id, idea)
    self.last_label = None # hack
    self._init_from_db()
  def _init_from_db(self):
    self.cur_question_code = self.get_question_codes()[0]
    self.read_in_ideas()
  def read_in_ideas(self):
    self.cur_ideas = []
    cursor = self.conn.cursor()
    cursor.execute("SELECT id, idea FROM ideas WHERE question_code = ? AND id NOT IN (SELECT idea_id FROM used_ideas) ORDER BY id ASC", (self.cur_question_code,))
    for idea_id, idea in cursor.fetchall():
      self.cur_ideas.append((idea_id, idea))
    cursor.close()

  def resolve_right(self):
    # This is so awful, please never look at this awful code again
    text = self.get_cluster_text()
    
  def get_ids_for_text(self, text):
    l = text
    id_loc = l.find('(_')
    ids = []
    if id_loc > 0:
        old_id = l[id_loc+2:-1]
        l_p = l[0:id_loc - 2]
        l_p = l_p.strip('-')

        cursor = self.conn.cursor()   
        cursor.execute("SELECT id FROM ideas WHERE question_code = ? AND idea = ?", (self.cur_question_code, l_p,))
        for (idea_id,) in cursor.fetchall():
            ids.append(idea_id)

        return l_p, ids
    else:
      return None

  def resolve(self):
    # fix lost ideas
    text = self.get_cluster_text()
    real_used_ids = self.extract_idea_ids_from_text(text)

    print(real_used_ids)

    lost = []
    cursor = self.conn.cursor()
    cursor.execute("SELECT id FROM ideas WHERE question_code = ? AND id IN (SELECT idea_id FROM used_ideas)", (self.cur_question_code,))
    for (idea_id,) in cursor.fetchall():
      if idea_id not in real_used_ids:
        lost.append(idea_id)

    print("made ideas unused:")
    print(lost)
    self.make_ideas_unused(lost)


  def get_question_codes(self):
    cursor = self.conn.cursor()
    cursor.execute("SELECT DISTINCT question_code FROM ideas ORDER BY question_code ASC")
    question_codes = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return question_codes
  def set_question_code(self, question_code):
    self.beginResetModel()
    self.cur_question_code = question_code
    self.read_in_ideas()
    self.endResetModel()
  def sort_based_on_similarity(self, idea_id):
    self.beginResetModel()
    cursor = self.conn.cursor()
    cursor.execute("SELECT id, idea FROM ideas, similarity_scores WHERE ideas.id = similarity_scores.idea2 AND question_code = ? AND similarity_scores.idea1 = ? AND similarity_scores.idea2 NOT IN (SELECT idea_id FROM used_ideas) ORDER BY score DESC", (self.cur_question_code, idea_id)) 
    self.cur_ideas = cursor.fetchall()
    cursor.close()
    self.endResetModel()
  def get_idea(self, idea_id):
    cursor = self.conn.cursor()
    cursor.execute("SELECT idea FROM ideas WHERE id = ?", (idea_id,))
    idea = cursor.fetchone()[0]
    cursor.close()
    return idea
  # Returns list of strings to display
  def make_ideas_used(self, indexes):
    self.beginResetModel()
    cursor = self.conn.cursor()
    items = [self.cur_ideas[i] for i in indexes]
    for item in items:
      cursor.execute("INSERT INTO used_ideas(idea_id) VALUES(?)", (item[0],))
      del self.cur_ideas[self.cur_ideas.index(item)]
    cursor.close()
    self.conn.commit()
    self.endResetModel()
    return [self.get_idea_string(i) for i in items]
  def make_ideas_unused(self, idea_id_list):
    self.beginResetModel()
    cursor = self.conn.cursor()
    for idea_id in idea_id_list:
      cursor.execute("DELETE FROM used_ideas WHERE idea_id = ?", (idea_id,))
      self.cur_ideas.insert(0, (idea_id, self.get_idea(idea_id)))
    self.conn.commit()
    cursor.close()
    self.endResetModel()
  def update_cluster_text(self, cluster_text):
    cursor = self.conn.cursor()
    cursor.execute("DELETE FROM clusters WHERE question_code = ?", (self.cur_question_code,))
    cursor.execute("INSERT INTO clusters(cluster_text, question_code) VALUES(?,?)", (cluster_text, self.cur_question_code))
    self.conn.commit()
    cursor.close()
  def get_cluster_text(self):
    cursor = self.conn.cursor()
    cursor.execute("SELECT cluster_text FROM clusters WHERE question_code = ?", (self.cur_question_code,)) 
    row = cursor.fetchone()
    if row:
      text = row[0]
    else:
      text = ""
    cursor.close()
    return text
  def get_idea_string(self, idea_tuple):
    return idea_tuple[1] + '  (_id:%d)' % idea_tuple[0]
 
# QAbstractListModel required methods
  def rowCount(self, parent):
    return len(self.cur_ideas)
  def data(self, index, role=QtCore.Qt.DisplayRole):
    row = index.row()
    if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.ToolTipRole:
      return self.get_idea_string(self.cur_ideas[row])
#    if role == QtCore.Qt.TextAlignmentRole and self.alignment_data:
#      return self.alignment_data[col]
    return None
# End QAbstractListModel required methods



class AppWindow(QtGui.QMainWindow):
  def __init__(self, parent=None):
    super(AppWindow, self).__init__(parent)
    self.ui = clustering_app.Ui_MainWindow()
    self.ui.setupUi(self)
    self.idea_model = IdeaListModel()

    self.idea_tree_models = dict()
    for qc in self.idea_model.get_question_codes():
      self.idea_tree_models[qc] = IdeaTreeModel(qc)

    self._finish_ui()
    self.regex_matches = []

  def _finish_ui(self):
    # Connect up all the buttons
    self.ui.button_move_down.clicked.connect(self.handle_move_selection_down)
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

  def handle_resolve_lost(self):
    self.idea_model.resolve()

  def clear_regex(self):
    self.regex_matches = []

  def handle_next_regex(self):
    if not self.regex_matches:
      test = re.compile(self.ui.line_regex.text(), re.IGNORECASE)
      text = self.idea_model.get_cluster_text()

      self.regex_matches = [i for (i, l) in enumerate(text.splitlines()) if test.match(l) is not None]

      # convenience hack; if no results, add .* and .* to end and beginning

      if len(self.regex_matches) == 0:
        test = re.compile(".*" + self.ui.line_regex.text() + ".*", re.IGNORECASE)
        self.regex_matches = [i for (i, l) in enumerate(text.splitlines()) if test.match(l) is not None]

      print("number of matches: %i" % len(self.regex_matches))

    if self.regex_matches:
      self.highlight_line(self.regex_matches[0])
      del self.regex_matches[0]


  def highlight_line(self, line):
    selection_cursor = self.ui.text_edit_clusters.textCursor()
    document = self.ui.text_edit_clusters.document()
    block = document.findBlockByLineNumber(line)

    selection_cursor.setPosition(block.position())

    self.ui.text_edit_clusters.setTextCursor(selection_cursor)
    self.ui.text_edit_clusters.moveCursor(QtGui.QTextCursor.EndOfLine, QtGui.QTextCursor.KeepAnchor)

    
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
    print("TODO")
    
  def handle_move_selection_up(self):
    print("TODO")

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
    print("TODO")
  def handle_combo_box_changed(self):
    self.save_clusters()
    qc = self.ui.combo_box_data_set.currentText()
    self.idea_model.set_question_code(qc)
    self.ui.tree_main.setModel(self.idea_tree_models[qc])

  def save_clusters(self):
    print("TODO SAVE")

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
