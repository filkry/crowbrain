#!/usr/bin/python
import os
import sys
import json
import re
import string
from PySide import QtCore, QtGui
import sqlite3 as db
import clustering_app
import csv
from collections import deque, defaultdict

db_file_name = 'clusters.db'

def get_line_indent(l):
  depth_count = 0
  l = l.strip()
  front = l
  while front and front[0] == '-':
    depth_count = depth_count + 1
    front = front[1:]
  return depth_count

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
    old_count = len(set(self.extract_idea_ids_from_text(text)))

    new_text = ""

    present_ids = []
    not_found_answers = []

    id_stacks = defaultdict(list)

    # find real IDs
    cursor = self.conn.cursor()    
    for l in text.splitlines():
      id_loc = l.find('(_')
      if id_loc > 0:
        old_id = l[id_loc+2:-1]
        l_p = l[0:id_loc - 2]
        l_p = l_p.strip('-')
        
        if l_p not in id_stacks:
          cursor.execute("SELECT id FROM ideas WHERE question_code = ? AND idea = ?", (self.cur_question_code, l_p,))
          for (idea_id,) in cursor.fetchall():
            id_stacks[l_p].append(idea_id)

        # check if idea is somehow not in the new list
        if l_p not in id_stacks:
          not_found_answers.append(l)
        else:
          real_id = id_stacks[l_p][-1]
          del id_stacks[l_p][-1]

          new_text += l.replace(old_id, "id:" + str(real_id)) + '\n'
          present_ids.append(real_id)

          assert(l_p == self.cur_ideas[real_id -1][1])
      else: # if not an id, just take text as-is
        new_text += l + '\n'

    new_count = len(set(self.extract_idea_ids_from_text(new_text)))

    print(old_count, new_count)

    indices = list(set([i -1 for i in present_ids]))

    print(new_text)

    self.make_ideas_used(indices)
    self.update_cluster_text(new_text)

    print("Answers not found:", not_found_answers)

    return new_text


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
  def extract_idea_ids_from_text(self, s):
    matches = []
    for match in re.findall('\(_id:[0-9]+\)', s):
      matches.append(int(re.findall('[0-9]+', match)[0]))
    return matches
# Returns (idea_id, cluster_id, parent_cluster_id, new_current_cluster_num)

  def get_next_depth(self, dq):
    for l in dq:
      if len(l) == 0:
        continue
      return get_line_indent(l)

  # Please note that all this code is awful
  def _get_line_info(self, l, peek, cluster_stack, new_cluster_num, cwriter, question_code):
    ids = self.extract_idea_ids_from_text(l)

    if not ids:
      # Check if this is a label
      if len(l) > 0:
        self.last_label = ''.join([c for c in l if c not in "-"]) # should be a regex, but I can write this faster

      # Newline, close clusters up to level of next idea
      next_depth = self.get_next_depth(peek)
      while len(cluster_stack) > next_depth - 1:
        del cluster_stack[-1]
      return None

    idea_id = ids[0]

    depth_count = get_line_indent(l)

    if depth_count > len(cluster_stack):
      while depth_count > len(cluster_stack):
        cluster_stack.append(new_cluster_num)
        new_cluster_num = new_cluster_num + 1
        if len(cluster_stack) > 1:
          cwriter.writerow([question_code, cluster_stack[-1], cluster_stack[-2], self.last_label])
        else:
          cwriter.writerow([question_code, cluster_stack[-1], None, self.last_label])
        self.last_label = None

    elif depth_count < len(cluster_stack):
      while depth_count < len(cluster_stack):
        del cluster_stack[-1]

    parent_cluster_id = None if len(cluster_stack) < 2 else cluster_stack[-2]

    return (idea_id, cluster_stack[-1], parent_cluster_id, new_cluster_num)

  def export_clusters(self, filename):
    cursor = self.conn.cursor()
    with open(filename, 'w') as fout:
      with open("%s_clusters.csv" % filename, 'w') as cfout:
        writer = csv.writer(fout)
        writer.writerow(['question_code', 'cluster_num', 'parent_idea_id', 'idea_id', 'idea', 'idea_num', 'worker_id', 'post_date', 'num_ideas_requested',])

        cwriter = csv.writer(cfout)
        cwriter.writerow(['question_code', 'cluster', 'cluster_parent', 'cluster_label'])

        last_ws = False

        for question_code in self.get_question_codes():
          cluster_stack = []
          new_cluster_num = 0
          last_id = -1
          cursor.execute("SELECT cluster_text FROM clusters WHERE question_code = ?", (question_code,)) 
          row = cursor.fetchone()
          if row:
            text = row[0]
          else:
            text = ""

          look_ahead = deque()
          for l in text.splitlines():
            look_ahead.append(l)

            if len(look_ahead) < 10:
              continue

            line = look_ahead.popleft()

            print(question_code, ",", line, ",", cluster_stack)
            line_info = self._get_line_info(line, look_ahead, cluster_stack, new_cluster_num, cwriter, question_code)

            if line_info:
              last_ws = False
              (idea_id, cluster_id, parent_id, cn) = line_info
              new_cluster_num = cn
              cursor.execute("SELECT idea, idea_num, worker_id, post_date, num_ideas_requested FROM ideas WHERE id = ?", (idea_id,))
              row_data = cursor.fetchone()
              writer.writerow([question_code, cluster_id, parent_id, idea_id] + list(row_data))

          # Bad form, but speed > goodness
          while len(look_ahead) > 0:
            line = look_ahead.popleft()

            line_info = self._get_line_info(line, look_ahead, cluster_stack, new_cluster_num, cwriter, question_code)

            if line_info:
              last_ws = False
              (idea_id, cluster_id, parent_id, cn) = line_info
              new_cluster_num = cn
              cursor.execute("SELECT idea, idea_num, worker_id, post_date, num_ideas_requested FROM ideas WHERE id = ?", (idea_id,))
              row_data = cursor.fetchone()
              writer.writerow([question_code, cluster_id, parent_id, idea_id] + list(row_data))

    cursor.close()
    print("export complete")

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



# This is probably very un-Qt, but I just need something that works ASAP
class FakeOverviewModel(QtCore.QAbstractListModel):
  def __init__(self, real_model, parent=None):
    super(FakeOverviewModel, self).__init__(parent)
    self.real_model = real_model
    self.top_labels = None
    self.reset_model()

  def _find_top_labels(self):
    text = self.real_model.get_cluster_text()
    test = re.compile("^-[^-()]+$")

    self.top_labels = sorted([(i, l) for (i, l) in enumerate(text.splitlines()) if test.match(l) is not None],
                              key = lambda x: x[1].lower())

  def reset_model(self):
    self.beginResetModel()
    self._find_top_labels()
    self.endResetModel()

  def rowCount(self, parent):
    return len(self.top_labels)

  def get_row_for_item(self, row):
    return self.top_labels[row][0]

  def data(self, index, role=QtCore.Qt.DisplayRole):
    row = index.row()
    if role == QtCore.Qt.DisplayRole or role == QtCore.Qt.ToolTipRole:
      return self.top_labels[row][1]
    return None



class AppWindow(QtGui.QMainWindow):
  def __init__(self, parent=None):
    super(AppWindow, self).__init__(parent)
    self.ui = clustering_app.Ui_MainWindow()
    self.ui.setupUi(self)
    self.idea_model = IdeaListModel()
    self.overview_model = FakeOverviewModel(self.idea_model)
    self._finish_ui()
    self.regex_matches = []

  def _finish_ui(self):
    # Connect up all the buttons
    self.ui.button_move_down.clicked.connect(self.handle_move_selection_down)
    self.ui.button_move_down_split.clicked.connect(self.handle_move_selection_down_split)
    self.ui.button_move_up.clicked.connect(self.handle_move_selection_up)
    self.ui.button_sort_by_list.clicked.connect(self.handle_sort_by_list_selection)
    self.ui.button_find_similar.clicked.connect(self.handle_find_similar)
    self.ui.button_next_regex.clicked.connect(self.handle_next_regex)
    self.ui.button_inc_indent.clicked.connect(self.handle_inc_indent)
    self.ui.button_dec_indent.clicked.connect(self.handle_dec_indent)
    self.ui.button_resolve_lost.clicked.connect(self.handle_resolve_lost)
    self.ui.button_resolve_right.clicked.connect(self.handle_resolve_right)

    # Connect menu items
    self.ui.menu_item_save.triggered.connect(self.save_cluster_text)
    self.ui.menu_item_export.triggered.connect(self.handle_export)
    self.ui.menu_item_quit.triggered.connect(self.close_app)

    # Init list
    self.ui.list_ideas.setModel(self.idea_model)
    self.ui.list_overview.setModel(self.overview_model)

    # Regex textbox
    self.ui.line_regex.textChanged.connect(self.clear_regex)
    self.ui.text_edit_clusters.textChanged.connect(self.clear_regex)

    # Connect list click
    self.ui.list_overview.clicked.connect(self.handle_overview_list_click)

    # Add entries to the combo box
    self.ui.combo_box_data_set.addItems(self.idea_model.get_question_codes())
    self.idea_model.set_question_code(self.ui.combo_box_data_set.currentText())
    self.sync_cluster_text()
    self.ui.combo_box_data_set.currentIndexChanged.connect(self.handle_combo_box_changed)

  def handle_resolve_lost(self):
    self.idea_model.resolve()

  def handle_resolve_right(self):
    new_text = self.idea_model.resolve_right()
    self.ui.text_edit_clusters.document().setPlainText(new_text)

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

  def handle_overview_list_click(self, index):
    line = self.overview_model.get_row_for_item(index.row())
    self.highlight_line(line)
    
  def showEvent(self, e):
    return super(AppWindow, self).showEvent(e)
  def closeEvent(self, e):
    self.save_cluster_text()
    return super(AppWindow, self).closeEvent(e)


  def move_selection_down(self, split = False):
    selected_indexes = [i.row() for i in self.ui.list_ideas.selectedIndexes()]
    new_strings = self.idea_model.make_ideas_used(selected_indexes)
    self.ui.text_edit_clusters.moveCursor(QtGui.QTextCursor.StartOfLine)
    self.ui.text_edit_clusters.moveCursor(QtGui.QTextCursor.EndOfLine, QtGui.QTextCursor.KeepAnchor)
    selected_text = self.ui.text_edit_clusters.textCursor().selectedText()
    indent_level = get_line_indent(selected_text)
    self.ui.text_edit_clusters.moveCursor(QtGui.QTextCursor.EndOfLine)
    if not self.ui.text_edit_clusters.document().isEmpty():
      self.ui.text_edit_clusters.insertPlainText("\n")
    indent_text = "-" * indent_level
    if split:
      self.ui.text_edit_clusters.insertPlainText("\n")
    self.ui.text_edit_clusters.insertPlainText("\n".join([indent_text + s for s in new_strings]))
    self.save_cluster_text()

  def handle_move_selection_down_split(self):
    self.move_selection_down(True)

  def handle_move_selection_down(self):
    self.move_selection_down(False)
    
  def handle_move_selection_up(self):
    selection_cursor = self.ui.text_edit_clusters.textCursor()
    selection_cursor.setPosition(selection_cursor.selectionStart())
    selection_cursor.movePosition(QtGui.QTextCursor.StartOfLine)
    end_cursor = self.ui.text_edit_clusters.textCursor()
    end_cursor.setPosition(end_cursor.selectionEnd())
    end_cursor.movePosition(QtGui.QTextCursor.EndOfLine)
    selection_cursor.setPosition(end_cursor.position(), QtGui.QTextCursor.KeepAnchor)
    self.ui.text_edit_clusters.setTextCursor(selection_cursor)
    selected_text = self.ui.text_edit_clusters.textCursor().selectedText()
    matches = self.idea_model.extract_idea_ids_from_text(selected_text)
    self.idea_model.make_ideas_unused(matches)
    self.ui.text_edit_clusters.textCursor().deleteChar()
    self.ui.text_edit_clusters.textCursor().deleteChar()
  def handle_inc_indent(self):
    self.handle_indent(1)
  def handle_dec_indent(self):
    self.handle_indent(-1)
  def handle_indent(self, amount):
    selection_cursor = self.ui.text_edit_clusters.textCursor()
    selection_cursor.setPosition(selection_cursor.selectionStart())
    selection_cursor.movePosition(QtGui.QTextCursor.StartOfLine)
    end_cursor = self.ui.text_edit_clusters.textCursor()
    end_cursor.setPosition(end_cursor.selectionEnd())
    end_cursor.movePosition(QtGui.QTextCursor.StartOfLine)
    while selection_cursor.position() < end_cursor.position():
      self._do_indent(selection_cursor, amount)
      selection_cursor.movePosition(QtGui.QTextCursor.Down)
      selection_cursor.movePosition(QtGui.QTextCursor.StartOfLine)
    self._do_indent(selection_cursor, amount)
    self.save_cluster_text()
  def _do_indent(self, text_cursor, amount):
      if amount > 0:
        text_cursor.insertText('-'*amount)
      else:
        for i in range(-1*amount):
          if text_cursor.document().characterAt(text_cursor.position()) == '-':
            text_cursor.deleteChar()
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
  def handle_find_regex(self):
    pass
  def handle_export(self):
    self.save_cluster_text()
    (filename, selected_filter) = QtGui.QFileDialog.getSaveFileName(self, caption="Enter filename to save clusters", filter="CSV files (*.csv)")
    if filename:
      if not filename.lower().endswith('.csv'):
        filename = filename + '.csv'
      self.idea_model.export_clusters(filename)
  def handle_combo_box_changed(self):
    self.save_cluster_text()
    self.idea_model.set_question_code(self.ui.combo_box_data_set.currentText())
    self.sync_cluster_text()
    self.overview_model.reset_model()
  def sync_cluster_text(self):
    self.ui.text_edit_clusters.document().setPlainText(self.idea_model.get_cluster_text())
  def save_cluster_text(self):
    self.idea_model.update_cluster_text(self.ui.text_edit_clusters.document().toPlainText())
    self.overview_model.reset_model()
  def close_app(self):
    self.save_cluster_text()
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
