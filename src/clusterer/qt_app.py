#!/usr/bin/python
import os
import sys
import json
import pickle
import re
import string
from PySide import QtCore, QtGui
import sqlite3 as db
import clustering_app
import import_cluster_text
import similarity_selector
import change_parent
import coverage
import title
import csv
import datetime
from collections import deque, defaultdict

db_file_name = '/home/fil/Desktop/clusters.db'


def get_current_time():
  return datetime.datetime.now()

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
    if label is not None and len(label) > 0:
      self._label = label
    else:
      self._label = None
    self.ideas = ideas # list of (text, id, time) tuples
    self.parent = parent
    self.children = []

  def as_text(self):
    outlines = []

    if self._label and len(self._label) > 0:
      outlines.append(self._label)
    for text, i, time in self.ideas:
      outlines.append(text + '  (_id:' + str(i) + ')')

    for i, c in enumerate(self.children):
      recur = c.as_text()
      outlines += ['' if l == '' else ('-' + l)
                        for l in recur]

      outlines.append('')

    return outlines

  def add_ideas(self, ideas):
    present_ids = [iid for idea, iid, time in self.ideas]
    ideas = [(idea, iid, time) for (idea, iid, time) in ideas if not iid in present_ids]
    self.ideas.extend(ideas)

  def merge(self, other_node):
    self.add_ideas(other_node.ideas)

  def sort_children(self):
    self.children = sorted(self.children, key=lambda x: x.label().lower())

  def append_child(self, child_node):
    if not child_node.parent is None:
        child_node.parent.remove_child(child_node)
    self.children.append(child_node)
    child_node.parent = self
    self.sort_children()

  def remove_child(self, child_node):
    if child_node in self.children:
      child_node.parent = None
      del self.children[self.children.index(child_node)]

  def child_count(self):
    return len(self.children)

  def all_child_nodes(self):
    ret = self.children.copy()

    for c in self.children:
      ret += c.all_child_nodes()

    return ret

  def child(self, row):
    if len(self.children) <= row:
      return None
    return self.children[row]

  def label(self):
    if self._label:
      return self._label
    elif len(self.get_all_ideas()) > 0:
      return self.get_all_ideas()[0][0]
    else:
      return "NO LABEL, NO IDEAS"

  def long_label(self):
    if self.parent is not None:
      l = self.parent.label()
      if l == 'forgot_name' or l == 'iPod' or l == 'charity' or l == 'turk':
        return self.label()
      else:
        return self.parent.long_label() + "/" + self.label()
    else:
      return self.label()

  def tooltip(self):
    ret = '\n'.join(i[0] for i in self.ideas)

    for c in self.children:
      tt = c.tooltip()
      ret = ret + '\n' + '\n'.join(['\t' + t for t in tt.split('\n')]) + '\n'

    return ret

  def row(self):
    if self.parent:
      return self.parent.children.index(self)
    return 0

  def get_parent(self):
    return self.parent

    ids = [i[1] for i in self.ideas]

    for c in self.children:
      ids += c.get_ids()

    return ids

  def get_all_ideas(self):
    ret = self.ideas.copy()

    for c in self.children:
      ret += c.get_all_ideas()

    return ret

class IdeaTreeModel(QtCore.QAbstractItemModel):
  def __init__(self, question_code, parent=None, root = None):
    super(IdeaTreeModel, self).__init__(parent)
    self.root = IdeaTreeNode([], None, question_code) if not root else root
    self.conn = db.connect(db_file_name)
    self.question_code = question_code

    if root is None:
      self.load()

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
    if c is None:
      return None
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
    if len(new_ids) == 0:
        print("Couldn't find idea:", l)
        return None
    new_id = new_ids[0]
    used_ids.append(new_id)
    current_time = get_current_time()
    return IdeaTreeNode([(idea, new_id, current_time)], current_node, idea)

  def import_from_text(self, text, idea_model):
    if len(text) == 0:
        return

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
          if depth_count - 1 != current_depth:
            print(line)
          assert depth_count - 1 == current_depth
          current_node.append_child(next_node)
          current_node = next_node
          current_depth += 1

        elif depth_count < current_depth:
          while depth_count < current_depth:
            current_node = current_node.get_parent()
            current_depth -= 1

    self.endResetModel()

    idea_model.make_ideas_used(used_ids)

  def bad_fake_reset(self):
    self.beginResetModel()
    self.endResetModel()

  def _save_node(self, node, cursor):
    # Create cluster
    cursor.execute("""INSERT INTO clusters(question_code, label)
                      VALUES (?, ?)""", (self.question_code, node._label))
    clus_id = cursor.lastrowid

    # Map ideas
    for text, i, time in node.ideas:
      cursor.execute("""INSERT INTO idea_clusters(idea_id, cluster_id, time_set)
                        VALUES (?, ?, ?)""", (i, clus_id, time))

    # Create children
    child_ids = [self._save_node(c, cursor) for c in node.children]
    
    # Create parent-child relationships
    for child_id in child_ids:
      cursor.execute("""INSERT INTO cluster_hierarchy(child, parent)
                        VALUES(?, ?)""", (child_id, clus_id))

    return clus_id

  def save(self):
    cursor = self.conn.cursor()
    cursor.execute("DELETE FROM clusters WHERE question_code=?",
                   (self.question_code,))

    self._save_node(self.root, cursor)

    self.conn.commit()
    cursor.close()
    self.conn.commit()

  def load(self):
    self.beginResetModel()
    cursor = self.conn.cursor()

    root_cluster = None
    cluster_dict = dict()

    # Load clusters
    cursor.execute("""SELECT id, label FROM clusters
                      WHERE question_code = ?""", (self.question_code,))
    for i, l in cursor.fetchall():
      node = IdeaTreeNode([], None, l)
      cluster_dict[i] = node
      if root_cluster is None:
        root_cluster = node

    if root_cluster is None:
      return

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
    cursor.execute("""SELECT cluster_id, id, idea, time_set
                      FROM ideas INNER JOIN idea_clusters
                      ON ideas.id = idea_clusters.idea_id
                      WHERE ideas.question_code = ?""",
                   (self.question_code,))
    for cid, iid, idea, time in cursor.fetchall():
      if cid in cluster_dict:
        cluster_dict[cid].add_ideas([(idea, iid, time)])

    for key in cluster_dict:
      # Don't load useless clusters (artifacts from import)
      c = cluster_dict[key]
      if len(c.get_all_ideas()) == 0:
        if not c.parent is None:
          c.parent.remove_child(c)

    cursor.close()

    self.root = root_cluster

    self.endResetModel()

  def get_ids(self, index):
    if index.isValid():
      return index.internalPointer().get_ids()
    else:
      return []

  def get_node(self, index):
      if index.isValid():
          node = index.internalPointer()
          return node

  def remove_node(self, index):
    self.beginResetModel()
    # This is probably a memory leak, I might be keeping a reference around
    if index.isValid():
      node = index.internalPointer()
      node.parent.remove_child(node)

    self.endResetModel()

  def remove_node_reparent_children(self, index):
      self.beginResetModel()
      ids = []
      if index.isValid():
          node = index.internalPointer()
          parent = node.parent
          ids = [i for (text, i, time) in node.ideas]
          for child in node.children.copy():
              parent.append_child(child)
          parent.remove_child(node)

      self.endResetModel()
      return ids


  def export_clusters_text(self, filename):
    with open("%s_%s_clusters.txt" % (filename, self.question_code), 'w') as cfout:
      lines = self.root.as_text()
      cfout.write('\n'.join(lines))

    pf = "%s_%s_IdeaTreeNode.pickle" % (filename, self.question_code)
    print(pf)
    with open(pf, 'wb') as cfout:
        pickle.dump(self.root, cfout)


  def export_clusters(self, filename):
    cursor = self.conn.cursor()
    with open("%s_%s_clusters.csv" % (filename, self.question_code), 'w') as cfout:
      cwriter = csv.writer(cfout)
      cwriter.writerow(['question_code', 'cluster', 'cluster_parent', 'cluster_label'])

      cursor.execute("""SELECT child, parent, label
                        FROM cluster_hierarchy INNER JOIN clusters
                        ON cluster_hierarchy.child = clusters.id
                        WHERE  clusters.question_code = ?""",
                     (self.question_code,))
      for c, p, l in cursor.fetchall():
        cwriter.writerow([self.question_code, c, p, l])

    with open("%s_%s.csv" % (filename, self.question_code), 'w') as fout:
      writer = csv.writer(fout)
      writer.writerow(['question_code', 'idea_id', 'cluster_id', 'idea', 'idea_num',
                       'worker_id', 'post_date', 'num_ideas_requested',])

      cursor.execute("""SELECT ideas.id, idea_clusters.cluster_id, idea, idea_num,
                               worker_id, post_date, num_ideas_requested
                        FROM ideas INNER JOIN idea_clusters
                        ON ideas.id = idea_clusters.idea_id
                        WHERE ideas.question_code = ?""",
                     (self.question_code,))
      for iid, cid, i, num, wid, date, nr in cursor.fetchall():
        writer.writerow([self.question_code, iid, cid, i, num, wid,
                         date, nr])
    
    cursor.close()



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
  
  def get_ideas_for_indices(self, indices):
    return [self.cur_ideas[i] for i in indices]

  def mean_similarity(self, group1_ids, group2_ids):
    cursor = self.conn.cursor()
    tups = [(x, y) for x in group1_ids for y in group2_ids]

    scores = []

    for id1, id2 in tups:
      cursor.execute("""SELECT score FROM similarity_scores
                        WHERE idea1 = ? AND idea2 = ?""", (id1, id2))
      res = cursor.fetchone()
      if res:
        scores.append(res[0])
      else:
        print("Missing similarity score", id1, id2)

    return sum(scores) / len(scores)

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
      print("Couldn't find ID", l)
      return None

  def resolve(self, idea_tree_model):
    # fix lost ideas
    real_used_ids = idea_tree_model.root.get_ids()

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

  def make_ideas_used(self, ids):
    self.beginResetModel()
    cursor = self.conn.cursor()
    for i in ids:
      cursor.execute("INSERT INTO used_ideas(idea_id) VALUES(?)", (i,))
    cursor.close()
    self.conn.commit()
    self.cur_ideas = [(i, idea) for (i, idea) in self.cur_ideas if i not in ids]
    self.endResetModel()

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
    self.ui.button_rename.clicked.connect(self.handle_button_rename)
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

  def coverage_prompt(self, node1, node2):
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

  def similarity_prompt(self, node, compare_root):
    scores = []

    if len(compare_root.children) == 0:
      return None

    nis = node.get_all_ideas()

    for c in compare_root.all_child_nodes():
      cis = c.get_all_ideas()
      if len(cis) == 0:
        print("No idea under", c.label())
      s = self.idea_model.mean_similarity([i[1] for i in nis],
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

    itm = IdeaTreeModel(self.idea_model.cur_question_code, root = compare_root)
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

  def change_parent_prompt(self, nodes):
    dialog = QtGui.QDialog()
    dialog.ui = change_parent.Ui_Dialog()
    dialog.ui.setupUi(dialog)

    qc = self.idea_model.cur_question_code
    itm = self.idea_tree_models[qc]
    dialog.ui.tree_options.setModel(itm)

    dialog.ui.btn_change_parent.clicked.connect(lambda x=0: dialog.done(x))
    dialog.ui.btn_cancel.clicked.connect(lambda x=1: dialog.done(x))

    ret = dialog.exec()

    if ret == 0:
      sel = dialog.ui.tree_options.selectedIndexes()
      if len(sel) > 0:
        index = sel[0]
        if index.isValid():
          for node in nodes:
            p = index.internalPointer()
            p.append_child(node)
            return

  def cluster_label_prompt(self, node):
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
      node._label = dialog.ui.lineEdit.text()

  def handle_move_selection_down(self):
    qc = self.idea_model.cur_question_code
    itm = self.idea_tree_models[qc]

    indexes = [i.row() for i in self.ui.list_ideas.selectedIndexes()]
    ideas = self.idea_model.get_ideas_for_indices(indexes)

    # I stored it backwards from Mike
    current_time = get_current_time()
    ideas_flipped = [(i[1], i[0], current_time) for i in ideas]

    new_node = IdeaTreeNode(ideas_flipped, None)
    current_node = itm.root

    while True:
      best_match = self.similarity_prompt(new_node, current_node)

      if best_match is None:
        if new_node._label is None:
          self.cluster_label_prompt(new_node)
        current_node.append_child(new_node)
        break

      else:
        bm_parent = best_match.parent
        if bm_parent is None:
          assert(False)

        # each is true if high, false if low
        cov_n_b, cov_b_n = self.coverage_prompt(new_node, best_match)
        if cov_n_b == cov_b_n and cov_n_b:
          best_match.merge(new_node)
          new_node = best_match

          if best_match._label is None:
            self.cluster_label_prompt(best_match)
          break
        elif cov_n_b == cov_b_n:
          # Create new node under the current node
          # TODO: move this into the tree class
          new_parent = IdeaTreeNode([], current_node, None)
          bm_parent.append_child(new_parent)

          bm_parent.remove_child(best_match)

          new_parent.append_child(best_match)
          new_parent.append_child(new_node)

          if new_node._label is None:
            self.cluster_label_prompt(new_node)

          self.cluster_label_prompt(best_match)
          self.cluster_label_prompt(new_parent)
          break
        elif cov_n_b > cov_b_n:
          # TODO: move this into tree class
          if new_node._label is None:
            self.cluster_label_prompt(new_node)

          bm_parent.remove_child(best_match)
          bm_parent.append_child(new_node)
          new_node.append_child(best_match)

          self.cluster_label_prompt(best_match)
          break
        else:
          current_node = best_match

    itm.bad_fake_reset()

    self.highlight_node(new_node)

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
      if index.isValid():
        qc = self.idea_model.cur_question_code
        itm = self.idea_tree_models[qc]
        node = itm.get_node(index)
        self.cluster_label_prompt(node)
        itm.bad_fake_reset()

  def handle_add_parent(self):
    sel = self.ui.tree_main.selectedIndexes()
    indices = [index for index in self.ui.tree_main.selectedIndexes() if index.isValid()]
    if len(indices) > 0:
        qc = self.idea_model.cur_question_code
        itm = self.idea_tree_models[qc]

        nodes = [itm.get_node(index) for index in indices]

        new_node = IdeaTreeNode([], None)
        for node in nodes:
            new_node.append_child(node)
        itm.root.append_child(new_node)
        self.cluster_label_prompt(new_node)

        itm.bad_fake_reset()

  def handle_change_parent(self):
    sel = self.ui.tree_main.selectedIndexes()
    indices = [index for index in self.ui.tree_main.selectedIndexes() if index.isValid()]
    if len(indices) > 0:
        qc = self.idea_model.cur_question_code
        itm = self.idea_tree_models[qc]

        nodes = [itm.get_node(index) for index in indices]
        self.change_parent_prompt(nodes)
        itm.bad_fake_reset()

  def handle_move_selection_up(self):
    btn = QtGui.QMessageBox.question(self, 'Confirmation', 
            'Really remove this node?',
            buttons = QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)

    if btn == QtGui.QMessageBox.Yes:
        sel = self.ui.tree_main.selectedIndexes()
        for index in sel:
          if index.isValid():
            qc = self.idea_model.cur_question_code
            itm = self.idea_tree_models[qc]

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
