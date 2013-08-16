#!/usr/bin/python

import os
import sys
import csv
import popen2
import operator
import re
import math
import cPickle as pickle
import nltk
import nltk.probability as nltkp
from nltk.corpus import wordnet as wn
import naive_bayes as nb
import networkx as nx
import numpy as np
import random
from collections import defaultdict
from scipy.stats.stats import pearsonr

base_dirs = ['/home/fil/Dropbox/crowbrain_share/experiments/pilot14-2013.08.16',
             '/home/fil/Dropbox/crowbrain_share/experiments/pilot13-2013-07-18',
             '/home/fil/Dropbox/crowbrain_share/experiments/pilot12-2013.07.16',
             '/home/fil/Dropbox/crowbrain_share/experiments/pilot11-2013.07.15',]

fixed_clusters_dir = '/home/fil/Dropbox/crowbrain_share/data/idea-clusters-2013.08.01/'
mike_scores_file = '/home/fil/Dropbox/crowbrain_share/data/originality-scores-2013.08.06/mike-scores.csv'
#base_dir = '/home/mterry/Dropbox/crowbrain_share/experiments/pilot13-2013-07-18'
cache_file = './cache.bin'
alzheimers_file_date = '2013.07.07'

# Recurses and reads in all data found within the base_dir
# Manages references to all of the QuestionSets
# Maintains counts of all stems across all sessions
class FullDataSet:
  def __init__(self, base_dir):
    self.runs = [] # List of BrainstormingRun objects
    self.question_sets = [] # List of QuestionSet objects
    self.question_sets_by_code = {}
    self.corpus_stem_counts = {} # Word stem counts across all data read in
    self.total_stems = 0 # Total number of word stems
    self._read_data(base_dir)
    self.question_sets = self.question_sets_by_code.values()
    self.update_stem_counts()
    for qs in self.question_sets:
      print "Question code:", qs.question_code, "total num responses:", len(qs.answers)
  def _read_data(self, dirs):
    for d in dirs:
      for f in os.listdir(d):
        full_name = os.path.join(d,f)
        if os.path.isdir(full_name):
          self._read_data(full_name)
        else:
          if f == 'answers.csv':
            self._read_file(full_name)
  def _read_file(self, f):
    with open(f) as fin:
      l = fin.readline()
      # If we see the follow, it's an early run we can ignore
      if 'Number of Answers' in l:
        return
      fin.seek(0)
      sep = '|'
      # Guess separator character
      if l.count(',') > l.count('|'):
        sep = ','
      run = BrainstormingRun(f, csv.reader(fin, delimiter=sep))
      self.runs.append(run)
      for answer_set in run.answer_sets:
        if not answer_set.question_code in self.question_sets_by_code:
          self.question_sets_by_code[answer_set.question_code] = QuestionSet(answer_set.question_code)
        question_set = self.question_sets_by_code[answer_set.question_code]
        question_set.answers.extend(answer_set.answers)
      print f
      print "Num responses:", len(run.answer_sets), "question codes:", run.get_unique_question_codes(), "total num ideas:", run.get_total_num_ideas()
#    for x in run.get_worker_ids():
#      print x
  def update_stem_counts(self):
    self.corpus_stem_counts = {}
    for run in self.runs:
      run.update_stem_counts()
      for answer_set in run.answer_sets:
        nb.add_counts(self.corpus_stem_counts, answer_set.answer_set_stem_counts)
    self.total_stems = reduce(operator.add, self.corpus_stem_counts.values(), 0)
    for question_set in self.question_sets:
      question_set.update_stem_counts()
      for answer in question_set.answers:
        stem_probs = []
        for stem in answer.answer_stem_counts:
          response_prob = answer.answer_set.get_stem_probability(stem)
          question_prob = question_set.get_stem_probability(stem)
          corpus_prob = self.get_stem_probability(stem)
          answer.answer_stem_prob[stem] = (response_prob, question_prob, corpus_prob)
          stem_probs.append((stem, question_prob))
        stem_probs.sort(lambda x,y: cmp(x[1],y[1]))
        answer.sorted_stem_probabilities = stem_probs # stem, question_prob
        answer.uniqueness_score = 1.0
        if answer.sorted_stem_probabilities:
          answer.uniqueness_score = reduce(operator.add, [x[1] for x in answer.sorted_stem_probabilities], 0) / len(answer.sorted_stem_probabilities)
  def get_stem_probability(self, stem):
    return 1.0 * self.corpus_stem_counts[stem] / self.total_stems

# Reads in data from a CSV file and stores results in a list of AnswerSet objects (answer_sets)
class BrainstormingRun:
  def __init__(self, filename, reader): # reader is a csv reader
    self.header = reader.next()
    num_cols = len(self.header)
    self.answer_sets = []
    self.filename = filename
    # Get indices of headings to make it easy to read into an AnswerSet
    primary_indices = [self.header.index(x) for x in ['hashed_worker_id', 'question',
                                                      'question_code', 'post_date',
                                                      'screenshot', 'num_answers_requested',
                                                      'answer_num', 'answer',]]
    cur_answer_set = None
    for row in reader:
      if len(row) != num_cols:
        print "Error: row column count != header column count. Ignoring", filename, self.header, row
      else:
        worker_id, question, question_code, post_date, screenshot, num_answers_requested, answer_num, answer = [row[i] for i in primary_indices]
        if alzheimers_file_date in filename and question_code == 'charity':
          question_code = 'alzheimers_charity'
        answer_num = int(answer_num)
        if answer_num == 0:
          cur_answer_set = AnswerSet(worker_id, question, question_code, post_date, screenshot, num_answers_requested)
          self.answer_sets.append(cur_answer_set)
        answer = Answer(cur_answer_set, answer_num, answer)
        cur_answer_set.answers.append(answer)
  def get_worker_ids(self):
    return [x.worker_id for x in self.answer_sets]
  def get_unique_question_codes(self):
    return set([x.question_code for x in self.answer_sets])
  def get_total_num_ideas(self):
    return reduce(operator.add, [len(x.answers) for x in self.answer_sets])
  def update_stem_counts(self):
    for answer_set in self.answer_sets:
      answer_set.update_stem_counts()

# Contains all Answer objects for a particular question
class QuestionSet:
  def __init__(self, question_code):
    self.question_code = question_code
    self.answers = [] # Answer objects
    self.sorted_answer_pairs = [] # Each row is a (similarity_score, Answer, Answer) tuple
                                  # Essentially orders pairs of Answer objects from most to least similar
                                  # Computed through get_sorted_answer_pairs
    self.similarity_matrix = None
    self.question_stem_counts = {} # Counts for each word stem across all Answers
    self.total_stems = 0 # Total number of stems
  def update_stem_counts(self):
    self.question_stem_counts = {}
    for answer in self.answers:
      nb.add_counts(self.question_stem_counts, answer.answer_stem_counts)
    self.total_stems = reduce(operator.add, self.question_stem_counts.values(), 0)
  def get_stem_probability(self, stem):
    return 1.0 * self.question_stem_counts[stem] / self.total_stems
  def print_top_n_grams(self):
    fdists = [nltk.probability.FreqDist() for x in range(3)]
    for answer in self.answers:
      for t in answer.stems:
        fdists[0].inc(t.lower())
      for bigram in answer.bigrams:
        fdists[1].inc(tuple([x.lower() for x in bigram]))
      for trigram in answer.trigrams:
        fdists[2].inc(tuple([x.lower() for x in trigram]))
    for i in range(len(fdists)):
      d = fdists[i]
      print "Frequency for", self.question_code, (i+1)
      d.tabulate(30)

  # Version of the above function that returns all <= n^2 pairs
  def get_sorted_answer_pairs(self):
    if len(self.sorted_answer_pairs) == 0:
      for i, a in enumerate(self.answers[:-1]):
        for b in self.answers[i + 1:]:
          score = a.similarity(b, self)
          self.sorted_answer_pairs.append([score, a, b])

      self.sorted_answer_pairs.sort(lambda x, y: cmp(x[0], y[0]))
      self.sorted_answer_pairs.reverse()

    return self.sorted_answer_pairs

  def get_similarity_matrix(self):
    if self.similarity_matrix is None:
      self.similarity_matrix = np.zeros((len(self.answers), len(self.answers)))

      # for i, a in enumerate(self.answers[:-1]):
      #   self.similarity_matrix[i, i] = 1
      #   for j, b in enumerate(self.answers[i+1:]):
      #     score = a.similarity(b, self)
      #     self.similarity_matrix[i, j] = score
      #     self.similarity_matrix[j, i] = score

      # I seem to be stupid, so do double the work to be safe
      for i, a in enumerate(self.answers):
        for j, b in (enumerate(self.answers)):
          score = a.similarity(b, self)
          self.similarity_matrix[i, j] = score

    return self.similarity_matrix

  def dump_answer_key(self, f):
    with open(f, 'w') as fout:
      csv_out = csv.writer(fout, dialect='excel')
      csv_out.writerow(['id', 'answer', 'worker_id', 'question_code', 'answer_num', 'num_requested'])
      for i, a in enumerate(self.answers):
        a_set = a.answer_set
        csv_out.writerow([i, a.answer, a_set.worker_id, a_set.question_code, a.answer_num, a_set.num_answers_requested])

  def dump_similarity_matrix(self, f):
    M = self.get_similarity_matrix()
    np.savetxt(f, M, delimiter=',')

  def get_ideas_sorted_by_uniqueness(self):
    unique_answers = self.answers[:]
    unique_answers.sort(lambda x,y: cmp(x.uniqueness_score, y.uniqueness_score))
    return unique_answers
  # First creates list of Answer similarity tuples (see get_sorted_answer_pairs), then
  # attempts to order the pairs based on how unique the ideas are (uses average uniqueness for this)
  def get_ideas_sorted_by_similarity_uniqueness(self):
    sorted_ideas = self.get_sorted_answer_pairs() # similarity score, answer 1, answer 2
    for score_set in sorted_ideas:
      score, a1, a2 = score_set
      score_set.append((a1.uniqueness_score + a2.uniqueness_score)/2)
    sorted_ideas.sort(lambda x,y: -cmp(x[-1],y[-1]))
    return sorted_ideas
    
# Represents a set of Answers from a Turker for a specific question
class AnswerSet:
  def __init__(self,
               worker_id,
               question,
               question_code,
               post_date,
               screenshot,
               num_answers_requested):
    self.worker_id = worker_id
    self.question = question
    self.question_code = question_code
    self.post_date = post_date
    self.screenshot = screenshot
    self.num_answers_requested = num_answers_requested
    self.answers = [] # Set of Answer objects
    self.answer_set_stem_counts = {} # Word stem counts totaled over all answers
    self.total_stems = 0 # Total number of word stems across all answers
  def update_stem_counts(self):
    self.answer_set_stem_counts = {}
    for answer in self.answers:
      nb.add_counts(self.answer_set_stem_counts, answer.answer_stem_counts)
    self.total_stems = reduce(operator.add, self.answer_set_stem_counts.values(), 0)
  def get_stem_probability(self, stem):
    return 1.0 * self.answer_set_stem_counts[stem] / self.total_stems

# Represents a single response/idea from a turker. Is part of an AnswerSet
class Answer:
  def __init__(self, answer_set, answer_num, answer):
    self.answer_set = answer_set # The AnswerSet object that this Answer is a part of
    self.answer_num = answer_num
    self.answer = answer
    self.stems, self.bigrams, self.trigrams, self.bag_words = nb.extract_features(answer, True, False, False, True)
    self.answer_stem_counts = {} # Counts for each word stem in the answer. Currently uses bag of words, which includes synonyms and hypernyms
    self.answer_stem_prob = {} # key is stem, value is tuple of probability: (in response set, in question, in whole corpus)
    self.sorted_stem_probabilities = [] # Sorted list of (stem, question_prob) items, calculated later. Used to calculate uniqueness_score
    self.uniqueness_score = 0.0 # Determined once all data is read in
    # Currently using the bag of words (which includes synonyms and hypernyms), rather than the stems
    self.answer_stem_counts = nb.init_token_count_dict(self.bag_words)
  def similarity(self, other_answer, question_set):
    if not self.bag_words or not other_answer.bag_words:
      return 0.0
    left_words = list(self.bag_words)
    right_words = list(other_answer.bag_words)
    # I was using the log of the inverse probability, but it wasn't having enough of an effect
    left = [1.0/question_set.get_stem_probability(t) for t in left_words]
    right = [1.0/question_set.get_stem_probability(t) for t in right_words]
    numerator = 0.0
    for i in range(len(left)):
      if left_words[i] in right_words:
        numerator = numerator + left[i]*left[i]
    denominator = reduce(operator.mul, [math.sqrt(reduce(operator.add, [p*p for p in p_list], 0)) for p_list in [left, right]], 1.0)
    return numerator/denominator
  def __repr__(self):
    return ' '.join([self.answer_set.worker_id, self.answer_set.question_code, str(self.answer_num), self.answer])

def get_full_data_set(base_dirs, cache_file):
  if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
      return pickle.load(f)
  full_data_set = FullDataSet(base_dirs)
  with open(cache_file, 'wb') as f:
    pickle.dump(full_data_set, f)
  return full_data_set  

# The caching can screw things up, so remember to delete it if things don't seem to be updating properly
data = get_full_data_set(base_dirs, cache_file)
#data = FullDataSet(base_dir)

output_dir = sys.argv[1]

question_sets = data.question_sets
for qs in question_sets:
  qs.print_top_n_grams()

  legend = '%s/%s_legend.csv' %(output_dir, qs.question_code)
  similarity = '%s/%s_similarity.csv' %(output_dir, qs.question_code)

  qs.dump_answer_key(legend)
  qs.dump_similarity_matrix(similarity)
