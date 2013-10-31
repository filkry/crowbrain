import pickle
import sys
import random

class IdeaTreeNode(object):
  def __init__(self, ideas, parent, label = None):
    assert not isinstance(ideas, str)
    if label is not None and len(label) > 0:
      self._label = label
    else:
      self._label = None
    self.ideas = ideas # list of (text, id) tuples
    self.parent = parent
    self.children = []

  def as_text(self):
    outlines = []

    if self._label and len(self._label) > 0:
      outlines.append(self._label)
    for text, i in self.ideas:
      outlines.append(text + '  (_id:' + str(i) + ')')

    for i, c in enumerate(self.children):
      recur = c.as_text()
      outlines += ['' if l == '' else ('-' + l)
                        for l in recur]

      outlines.append('')

    return outlines

  def add_ideas(self, ideas):
    present_ids = [iid for idea, iid in self.ideas]
    ideas = [(idea, iid) for (idea, iid) in ideas if not iid in present_ids]
    self.ideas.extend(ideas)

  def merge(self, other_node):
    self.add_ideas(other_node.ideas)

  def sort_children(self):
    self.children = sorted(self.children, key=lambda x: x.label().lower())

  def append_child(self, child_node):
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

  def get_ids(self):
    ids = [i[1] for i in self.ideas]

    for c in self.children:
      ids += c.get_ids()

    return ids

  def get_all_ideas(self):
    ret = self.ideas.copy()

    for c in self.children:
      ret += c.get_all_ideas()

    return ret

root = None
with open("out/_iPod_IdeaTreeNode.pickle", 'rb') as f:
    root = pickle.load(f)

if root is None:
    print("error")
    sys.exit(1)

all_trees = root.children

sampled_trees = random.sample(all_trees, 40)

for i in sampled_trees:
    print('\n'.join(i.as_text()))
    print("\n\n\n\n\n\n\n\n\n\n")
