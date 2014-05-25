from PySide import QtCore
import sqlite3 as db

class IdeaListModel(QtCore.QAbstractListModel):
  def __init__(self, db_file_name, parent=None):
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
    real_used_ids = [id for (idea, id, time) in idea_tree_model.root.get_all_ideas()]

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


