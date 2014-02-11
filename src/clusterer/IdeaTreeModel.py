import datetime, csv
import sqlite3 as db
import pickle
from IdeaTreeNode import IdeaTreeNode
from PySide import QtCore

def get_current_time():
  return datetime.datetime.now()

class IdeaTreeModel(QtCore.QAbstractItemModel):
  def __init__(self, question_code, db_file_name, parent=None, root = None):
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

  def change_parent(self, node_indices, parent_selector):
      new_parent_ix = parent_selector(self)
      if(not new_parent_ix is None and new_parent_ix.isValid()):
        self.beginResetModel()
        new_parent = new_parent_ix.internalPointer()

        nodes = (index.internalPointer() for index in node_indices if index.isValid())
        for node in nodes:
          new_parent.append_child(node)
        self.endResetModel()

  def add_parent(self, node_indices, parent_label_selector):
      nodes = [ix.internalPointer() for ix in node_indices if ix.isValid()]
      if len(nodes) > 0:
        self.beginResetModel()
        new_node = IdeaTreeNode([], None)
        for node in nodes:
          new_node.append_child(node)
        self.root.append_child(new_node)
        new_node._label = parent_label_selector(new_node)
        self.endResetModel()

  def remove_ideas(self, node_index, id_selector):
    self.beginResetModel()
    ids = []
    if node_index.isValid():
        node = node_index.internalPointer()
        ids = id_selector(node)
        node.ideas = [(idea, i, time) for (idea, i, time) in node.ideas if not i in ids]
    self.endResetModel()
    return ids

  def change_node_label(self, node_index, label_selector):
    self.beginResetModel()
    if node_index.isValid():
      node = node_index.internalPointer()
      label = label_selector(node)
      node._label = label
    self.endResetModel()

  def add_ideas(self, ideas, label_selector, similarity_selector, coverage_selector,
                idea_model):
    self.beginResetModel()
    # I stored it backwards from Mike
    current_time = get_current_time()
    ideas_flipped = [(i[1], i[0], current_time) for i in ideas]

    new_node = IdeaTreeNode(ideas_flipped, None)
    current_node = self.root

    while True:
      best_match = similarity_selector(new_node, current_node, idea_model)

      if best_match is None:
        if new_node._label is None:
          new_node._label = label_selector(new_node)
        current_node.append_child(new_node)
        break

      else:
        bm_parent = best_match.parent
        if bm_parent is None:
          assert(False)

        # each is true if high, false if low
        cov_n_b, cov_b_n = coverage_selector(new_node, best_match)
        if cov_n_b == cov_b_n and cov_n_b:
          best_match.merge(new_node)
          new_node = best_match

          if best_match._label is None:
            best_match._label = label_selector(new_node)
          break
        elif cov_n_b == cov_b_n:
          # Create new node under the current node
          new_parent = IdeaTreeNode([], current_node, None)
          bm_parent.append_child(new_parent)

          bm_parent.remove_child(best_match)

          new_parent.append_child(best_match)
          new_parent.append_child(new_node)

          if new_node._label is None:
            new_node._label = label_selector(new_node)

          best_match._label = label_selector(best_match)
          new_parent._label = label_selector(new_parent)
          break
        elif cov_n_b > cov_b_n:
          if new_node._label is None:
            new_node._label = label_selector(new_node)

          bm_parent.remove_child(best_match)
          bm_parent.append_child(new_node)
          new_node.append_child(best_match)

          best_match._label = label_selector(best_match)
          break
        else:
          current_node = best_match

    self.endResetModel()
    return new_node


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


