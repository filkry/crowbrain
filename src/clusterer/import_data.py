#!/usr/bin/python

import csv
import os
import sqlite3 as db
import re
import nltk
import feature_extraction as fe
import naive_bayes as nb
import operator
import math
import sys

sys.setrecursionlimit(10000)

db_name = '/home/fil/Dropbox/clusters.db'
clusters_dir = '/home/fil/Dropbox/crowbrain_share/data/idea-clusters-2013.10.21/'

def create_tables(cursor):
  sql = [
    '''CREATE TABLE IF NOT EXISTS
          ideas (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 idea TEXT NOT NULL,
                 idea_num INTEGER NOT NULL,
                 auto_cluster_num INTEGER NOT NULL,
                 worker_id TEXT NOT NULL,
                 post_date TEXT NOT NULL,
                 num_ideas_requested INTEGER NOT NULL,
                 question_code TEXT NOT NULL)''',
    '''CREATE TABLE IF NOT EXISTS
          used_ideas(idea_id INTEGER REFERENCES ideas(id))''',
    '''CREATE TABLE IF NOT EXISTS
          similarity_scores (idea1 INTEGER NOT NULL REFERENCES ideas(id),
                             idea2 INTEGER NOT NULL REFERENCES ideas(id),
                             score DOUBLE)''',
    '''CREATE TABLE IF NOT EXISTS
          clusters (id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question_code TEXT NOT NULL,
                    label TEXT)''',
    '''CREATE TABLE IF NOT EXISTS
          cluster_hierarchy (child INTEGER NOT NULL REFERENCES clusters(id),
                             parent INTEGER NOT NULL REFERENCES clusters(id))''',
    '''CREATE TABLE IF NOT EXISTS
          idea_clusters (idea_id INTEGER NOT NULL REFERENCES ideas(id),
                         cluster_id INTEGER NOT NULL REFERENCES clusters(id),
                         UNIQUE(idea_id) ON CONFLICT REPLACE)''',
    '''CREATE INDEX IF NOT EXISTS
          ideas_idea_index ON ideas(idea)''',
    '''CREATE INDEX IF NOT EXISTS
          used_ideas_index ON used_ideas(idea_id)''',
    '''CREATE INDEX IF NOT EXISTS
          clusters_cluster_index ON clusters(id)''',
    '''CREATE INDEX IF NOT EXISTS
          cluster_h_cluster_index ON cluster_hierarchy(child)''',
    '''CREATE INDEX IF NOT EXISTS
          idea_clusters_idea_index ON idea_clusters(idea_id)''',
    '''CREATE INDEX IF NOT EXISTS
          idea_clusters_cluster_index ON idea_clusters(cluster_id)''',
    '''CREATE INDEX IF NOT EXISTS
          similarity_score_index_1 ON similarity_scores(idea1)''',
    '''CREATE INDEX IF NOT EXISTS
          similarity_score_index_2 ON similarity_scores(idea2)''',
    '''CREATE INDEX IF NOT EXISTS
          similarity_score_index_3 ON similarity_scores(idea1, idea2)''',
  ]
  for s in sql:
    cursor.execute(s)

def import_file(fname, cursor):
  with open(fname) as fin:
    print "Importing", fname
    reader = csv.reader(fin)
    reader.next() # Read header
    for row in reader:
      cursor.execute("INSERT INTO ideas(auto_cluster_num, idea, idea_num, worker_id, post_date, num_ideas_requested, question_code) VALUES(?, ?, ?, ?, ?, ?, ?)", row)

def get_stems(cursor):
  cursor.execute('SELECT idea FROM ideas')
  regex = re.compile('[^a-z ]')
  stems = set()
  stemmer = nltk.PorterStemmer()
  for row in cursor.fetchall():
    idea = row[0].lower()
    idea = regex.sub('', idea)
    for t in idea.split():
      stem = stemmer.stem(t)
      stems = stems.union(set(stem))
  return stems

def import_clusters(dir_name):
  conn = db.connect(db_name)
  cursor = conn.cursor()
  create_tables(cursor)
  cursor.execute("SELECT COUNT(*) FROM ideas")
  if cursor.fetchone()[0] == 0:
    for f in os.listdir(dir_name):
      if not f.endswith('csv'):
        continue
      import_file(os.path.join(dir_name, f), cursor)
  conn.commit()
  cursor.close()
  conn.close()

def get_features(s):
  tokens = fe.tokenize_lowercase_no_punct(s)
  bag_of_words = fe.feature_list_to_bag_of_words(
        fe.extract_stems(
          fe.merge_feature_lists(fe.extract_hypernyms(tokens),
            fe.merge_feature_lists(fe.extract_synonyms(tokens), fe.extract_identity(tokens)))))
  return bag_of_words

def score_ideas(left_feature_set, right_feature_set, probability_dict):
  left_features = list(left_feature_set)
  right_features = list(right_feature_set)
  # I was using the log of the inverse probability, but it wasn't having enough of an effect
  left = [1.0/probability_dict[t] for t in left_features]
  right = [1.0/probability_dict[t] for t in right_features]
  numerator = 0.0
  for i in range(len(left)):
    if left_features[i] in right_features:
      numerator = numerator + left[i]*left[i]
  denominator = reduce(operator.mul, [math.sqrt(reduce(operator.add, [p*p for p in p_list], 0)) for p_list in [left, right]], 1.0)
  return numerator/denominator


def score_question(cursor, question_code):
  cursor.execute("SELECT id, idea FROM ideas WHERE question_code = ?", (question_code,))
  ideas = cursor.fetchall() # (id, idea) tuples
  features = {} # key is idea id, value is feature set
  feature_counts = {}
  feature_probs = {}
  for idea_id, idea in ideas:
    features[idea_id] = get_features(idea)
    if not feature_counts:
      feature_counts = nb.init_count_dict(list(features[idea_id]))
    else:
      nb.add_counts(feature_counts, nb.init_count_dict(list(features[idea_id])))
  feature_probs = nb.convert_counts_to_probabilities(feature_counts)
  for i in range(len(ideas)):
    for j in range(i, len(ideas)):
      score = score_ideas(features[ideas[i][0]], features[ideas[j][0]], feature_probs)
      values = [ideas[i][0], ideas[j][0], score]
      cursor.execute("INSERT INTO similarity_scores(idea1, idea2, score) VALUES (?,?,?)", values)
      if values[0] != values[1]:
        cursor.execute("INSERT INTO similarity_scores(idea2, idea1, score) VALUES (?,?,?)", values)

def do_scoring():
  conn = db.connect(db_name)
  cursor = conn.cursor()
#  stems = get_stems(cursor)
  cursor.execute("SELECT DISTINCT question_code FROM ideas")
  question_codes = [row[0] for row in cursor.fetchall()]
  for question_code in question_codes:
    score_question(cursor, question_code)
  conn.commit()
  cursor.close()
  conn.close()

import_clusters(clusters_dir)
do_scoring()
