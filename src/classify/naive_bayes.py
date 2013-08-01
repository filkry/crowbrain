#!/usr/bin/python

import os
import sys
import csv
import popen2
import operator
import re
import cPickle as pickle
import nltk
from nltk.corpus import wordnet as wn
import pprint
import math

# Dependencies: nltk, WordNet for nltk, and hunspell (installed locally, only if spell checking)

TAG_COUNTS = 0
TAG_TO_TOKEN_COUNTS = 1
TAG_PROB = 2
TAG_TO_TOKEN_PROB = 3
def create_model():
# First two dictionaries are the count dictionaries, second two are the probability dictionaries
  return [{}, {}, {}, {}]

# Returns (stems, bigrams, trigrams, bag_of_words)
# model not actually used right now
def extract_features_for_model(model, clause, spellcheck):
  return extract_features(clause, spellcheck, False, False)

# tokens is the already extracted feature set, labels is a list of strings
def update_model(model, tokens, labels):
  tag_counts = model[TAG_COUNTS]
  tag_to_token_counts = model[TAG_TO_TOKEN_COUNTS]
  update_counts_from_parsed_tokens(tokens, labels, tag_counts, tag_to_token_counts)
  model[TAG_PROB], model[TAG_TO_TOKEN_PROB] = convert_tag_count_dictionaries_to_probabilities(tag_counts, tag_to_token_counts)

# tokens is the already extracted feature set
# Returns list of (tag, log(prob)) tuples, ordered most to least likely
def label_tokens(model, tokens):
  return get_tag_predictions_from_parsed_tokens(tokens, model[TAG_PROB], model[TAG_TO_TOKEN_PROB])

# Assumes two sets of tokens passed in
def bag_of_words_cos_similarity(l, r):
  if len(l) == 0 or len(r) == 0:
    return 0
  common_set = l.intersection(r)
  num_in_common = len(common_set)
  denom = reduce(operator.mul, [math.sqrt(len(x)) for x in [l,r]], 1.0)
  return num_in_common / denom



# Spell checking stuff: http://blog.quibb.org/2009/04/spell-checking-in-python/
# Following code is from above

# Spell checker for LibreOffice. In some casual tests, I like its output better
class hunspell:
    def __init__(self):
        self.cache = {}
        self._f = popen2.Popen3("hunspell -d en_US")
        self._f.fromchild.readline() #skip the credit line
    def __call__(self, words):
        words = words.split(' ')
        output = []
        for word in words:
            if word in self.cache:
              output.append(self.cache[word])
            else:
              self._f.tochild.write(word+'\n')
              self._f.tochild.flush()
              s = self._f.fromchild.readline().strip().lower()
              self._f.fromchild.readline() #skip the blank line
              if s == "*":
                  output.append(None)
                  self.cache[word] = None
              elif s[0] == '#':
                  output.append("No Suggestions")
                  self.cache[word] = "No Suggestions"
              elif s[0] == '+':
                  pass
              else:
                  try:
                    output.append(s.split(':')[1].strip().split(', '))
                    self.cache[word] = output[-1]
                  except:
                    print "hunspell error on word, s:", word, s
        return output

spellchecker = hunspell()
# Returns list of original tokens with up to three additional spelling suggestions for each token
def spell_check_tokens(tokens):
  return_tokens = []
  for w in tokens:
    results = spellchecker(w)
    to_add = [w]
    if results and results[0]:
      to_add.extend(results[0][:3]) # Add first three suggestions
    return_tokens.extend(list(set(to_add)))
  return return_tokens      

# Returns a single word that is best spelling for given word
def get_best_spelling(w):
  results = spellchecker(w)
  if results and results[0]:
    return results[0][0]
  return w

def clean_clause(clause):
  return clause.replace(" '", "'").replace("n't", "not")

def tokenize_number(t):
  try:
    float(t)
    return '_NUM_'
  except ValueError:
    return t
  
def code_punctuation(t):
  return_set = []
  punct_dict = {
    '!' : '_EXCLAMATION_POINT_',
    '@' : '_AT_SYMBOL_',
    '#' : '_HASH_',
    '$' : '_DOLLAR_SIGN_',
    '%' : '_PERCENT_',
    '^' : '_HAT_',
    '&' : '_AMPERSAND_',
    '*' : '_ASTERISK_',
    '(' : '_OPEN_PAREN_',
    ')' : '_CLOSE_PAREN_',
    '-' : '_MINUS_',
    '_' : '_UNDERSCORE_',
    '+' : '_PLUS_',
    '=' : '_EQUALS_',
    '[' : '_LEFT_ANGLE_BRACKET_',
    ']' : '_RIGHT_ANGLE_BRACKET_',
    '{' : '_LEFT_CURLY_BRACE',
    '}' : '_RIGHT_CURLY_BRACE',
    '\\' : '_BACK_SLASH',
    '|' : '_PIPE_',
    ';' : '_SEMICOLON_',
    ':' : '_COLON_',
    ',' : '_COMMA_',
    '.' : '_PERIOD_',
    '<' : '_LESS_THAN_',
    '>' : '_GREATER_THAN_',
    '/' : '_FORWARD_SLASH_',
    '?' : '_QMARK_',
    ':)' : '_SMILEY_',
    '--' : '_DOUBLE_DASH_',
    '//' : '_DOUBLE_FORWARD_SLASH_',
    '`' : '_BACK_TICK_',
    '~' : '_TILDE_',
    '"' : '_DOUBLE_QUOTE_',
    "'" : '_SINGLE_QUOTE_',
  }
  for p in punct_dict.keys():
    if p in t:
      return_set.append(punct_dict[p])
  if '(' in t and ')' in t:
    return_set.append('_OPEN_CLOSE_PAREN_')
  if "'" in t or '"' in t:
    return_set.append('_QUOTE_')
  return return_set
  
def no_punct_tokenize(s):
#    self.tokens = nltk.word_tokenize(self.answer)
  s = re.sub(r'[^a-zA-Z0-9 ]', '', s)
  return s.split()

def minimal_punct_tokenize(s):
  s = re.sub(r'[^a-zA-Z0-9,.!? ]', '', s)
  return s.split()

def get_synonyms(word, pos_tag=None):
  word = word.lower()
  return filter_synsets_to_words(wn.synsets(word), pos_tag)

def get_hypernyms(word, pos_tag=None):
  word = word.lower()
  return filter_synsets_to_words(reduce(operator.add, [w.hypernyms() for w in wn.synsets(word)], []), pos_tag)
  
def filter_synsets_to_words(synsets, pos_tag):
  if pos_tag:
    pos_tag = wordnet_pos_code(pos_tag)
  final_list = reduce(operator.add, [x.lemma_names for x in synsets if not pos_tag or x.pos == pos_tag], [])
  return list(set(final_list))

# Code snippet from: http://www.ling.helsinki.fi/~gwilcock/Tartu-2011/P2-nltk-2.xhtml
def wordnet_pos_code(tag):
  if tag.startswith('NN'):
    return wn.NOUN
  elif tag.startswith('VB'):
    return wn.VERB
  elif tag.startswith('JJ'):
    return wn.ADJ
  elif tag.startswith('RB'):
    return wn.ADV
  else:
    return ''

# Creates a count dictionary from the tokens passed in
def init_token_count_dict(tokens):
  counts = {}
  for t in tokens:
    if not t in counts:
      counts[t] = 1.0
    else:
      counts[t] = counts[t] + 1.0
  return counts

# Adds counts from src_dict to dest_dict
def add_counts(dest_dict, src_dict):
  for t in src_dict:
    if not t in dest_dict:
      dest_dict[t] = src_dict[t]
    else:
      dest_dict[t] = dest_dict[t] + src_dict[t]

# Assumes a sentence is passed in. The include_bi/trigrams is whether to include those in the bag of words returned
# Filter stopwords is whether to filter out stopwords from the stems and bag of words. Uses NLTK's stopword list
# Bag of words includes synonyms and hypernyms
# Returns (stems, bigrams, trigrams, bag_of_words)
def extract_features(sentence, correct_spelling, include_bigrams, include_trigrams, filter_stopwords=False):
  abstracted_tokens = []
  sentence = sentence.encode('utf-8')
  got_first_token = False
  regex = re.compile('.*[A-Z].*')
  for token in sentence.split():
    abstracted_tokens.extend(code_punctuation(token))
    if got_first_token:
      if regex.match(token):
        abstracted_tokens.append('_CAP_')
    got_first_token = True
  tokens = no_punct_tokenize(sentence)
  if correct_spelling:
    tokens = [get_best_spelling(t) for t in tokens]
  pos_tags = nltk.pos_tag(tokens) # TODO: Validate. This may be thrown off a bit by there being no punctuation (e.g., if there are multiple sentences)
  synonyms = reduce(operator.add, [get_synonyms(t, ptag) for t,ptag in pos_tags], [])
  hypernyms = reduce(operator.add, [get_hypernyms(t, ptag) for t,ptag in pos_tags], [])
  lower_tokens = [w.lower() for w in tokens]
  # Not sure how I feel about replacing a number with an abstract token here, but will do it for now... Could pull out the actual numbers later and add them to the bag of words
  if filter_stopwords:
    lower_tokens = [tokenize_number(t) for t in lower_tokens if not t in nltk.corpus.stopwords.words('english')]
  else:
    lower_tokens = [tokenize_number(t) for t in lower_tokens]
  stemmer = nltk.PorterStemmer()
  stems = [stemmer.stem(t) for t in lower_tokens]
  bigrams = set(nltk.bigrams(stems))
  trigrams = set(nltk.trigrams(stems))
  
  stemmed_everything = list(set(stems + [stemmer.stem(t) for t in (synonyms + hypernyms)]))
  bag_words = set(stemmed_everything + abstracted_tokens)
  if include_bigrams:
    bag_words = bag_words.union(bigrams)
  if include_trigrams:
    bag_words = bag_words.union(trigrams)
  return (stems, bigrams, trigrams, bag_words)

# Updates the counts for a pre-tokenized clause
# Updates the counts for a given clause and its associated tags
def update_counts(clause, tags, tag_counts, tag_to_token_counts, correct_spelling, include_bigrams, include_trigrams):
  if tags:
    clause = clean_clause(clause)
    stems, bigrams, trigrams, bag_of_words = extract_features(clause, correct_spelling, include_bigrams, include_trigrams)
    update_counts_from_parsed_tokens(bag_of_words, tags, tag_counts, tag_to_token_counts)

def update_counts_from_parsed_tokens(tokens, tags, tag_counts, tag_to_token_counts):
  if tags:
    token_counts = init_token_count_dict(tokens)
    for tag in tags:
      if not tag in tag_counts:
        tag_counts[tag] = 1.0
      else:
        tag_counts[tag] = tag_counts[tag] + 1.0
      if not tag in tag_to_token_counts:
        tag_to_token_counts[tag] = {}
      add_counts(tag_to_token_counts[tag], token_counts)

def convert_counts_to_probabilities(count_dict):
  total_counts = reduce(operator.add, count_dict.values(), 0.0)
  prob_dict = {}
  for t in count_dict:
    prob_dict[t] = count_dict[t] / total_counts
  return prob_dict

def convert_tag_count_dictionaries_to_probabilities(tag_counts, tag_to_token_counts):
  tag_probabilities = convert_counts_to_probabilities(tag_counts)
  tag_to_token_probabilities = {}
  total_num_tokens = get_total_num_tag_tokens(tag_to_token_counts)
  for tag in tag_to_token_counts:
    tag_to_token_probabilities[tag] = convert_counts_to_probabilities(tag_to_token_counts[tag])
    tag_to_token_probabilities[tag]['_MISSING_'] = 1.0/total_num_tokens
  return tag_probabilities, tag_to_token_probabilities
  
def get_total_num_tag_tokens(tag_to_token_counts):
  return reduce(operator.add, [reduce(operator.add, tag_to_token_counts[tag].values(), 0.0) for tag in tag_to_token_counts], 0.0)

def get_tag_predictions(clause, tag_probabilities, tag_to_token_probabilities, correct_spelling, include_bigrams, include_trigrams):
  clause = clean_clause(clause)
  stems, bigrams, trigrams, bag_of_words = extract_features(clause, correct_spelling, include_bigrams, include_trigrams)
  return get_tag_predictions_from_parsed_tokens(bag_of_words, tag_probabilities, tag_to_token_probabilities)

def get_tag_predictions_from_parsed_tokens(tokens, tag_probabilities, tag_to_token_probabilities):
  final_probs = {}
  for tag in tag_probabilities:
    prob_sum = 0.0
    for token in tokens:
      if not token in tag_to_token_probabilities[tag]:
        token = '_MISSING_'
      this_prob = math.log(tag_to_token_probabilities[tag][token])
      prob_sum = prob_sum + this_prob
    final_probs[tag] = prob_sum + math.log(tag_probabilities[tag])
  predictions = [(tag, int(final_probs[tag])) for tag in final_probs]
  predictions.sort(lambda l,r: -cmp(l[1],r[1]))
  return predictions

# Convenience function for testing
# Outputs prediction sfor each of the clauses passed in
def predict_clauses(clauses, tag_counts, tag_to_token_counts, correct_spelling, include_bigrams, include_trigrams):
  tag_probabilities, tag_to_token_probabilities = convert_tag_count_dictionaries_to_probabilities(tag_counts, tag_to_token_counts)
  return [get_tag_predictions(clause, tag_probabilities, tag_to_token_probabilities, correct_spelling, include_bigrams, include_trigrams) for clause in clauses]

# Convenience function for testing
# Assumes a list or tuple passed in with the clause as the first item, and
# a list of tags as the second item
# Returns two dictionaries. First is a dictionary mapping tags to the number
# of times those tags have been seen
# Second is a dictionary of dictionaries. Key is a tag, which yields a second
# dictionary that maps a token to the number of times it's been seen for that tag
def get_feature_counts(clauses_and_tags, correct_spelling, include_bigrams, include_trigrams):
  tag_to_token_counts = {}
  tag_counts = {}
  for clause, tags in clauses_and_tags:
    update_counts(clause, tags, tag_counts, tag_to_token_counts, correct_spelling, include_bigrams, include_trigrams)
  return tag_counts, tag_to_token_counts

def run_tests(correct_spelling, include_bigrams, include_trigrams):
  clauses_and_tags = [
    ('Click on teh color button and type in a setting of 128 or do a #FFFF00 color :)', ['do', 'app part',]),
    ('Now click on the paintbrush to start painting', ['do', 'app part']),
    ('If you want, you can color this in with blue', ['optional']),
    ('If this looks too harsh, you can fix it later', ['optional']),
  ]
  clauses = [ct[0] for ct in clauses_and_tags] + [
    "In this tutorial, we'll make your brown eyes blue",
    'Press the button labeled "Save" to save your document',
    "If it doesn't come out OK, you can always undo it",
  ]
  features = [extract_features(clause, correct_spelling, include_bigrams, include_trigrams) for clause in clauses]
  for clause, (stems, bigrams, trigrams, bag_of_words) in zip(clauses, features):
    print "Clause:", clause
    print "Stems:", stems
    print "bigrams:", bigrams
    print "trigrams:", trigrams
    print "bag of words:", bag_of_words
  for i in range(len(features)-1):
    for j in range(i, len(features)):
      l = features[i][-1]
      r = features[j][-1]
      print "Similarity between clause", i, "and", j, ":", bag_of_words_cos_similarity(l,r)
  tag_counts, tag_to_token_counts = get_feature_counts(clauses_and_tags, correct_spelling, include_bigrams, include_trigrams)
  tag_probabilities, tag_to_token_probabilities = convert_tag_count_dictionaries_to_probabilities(tag_counts, tag_to_token_counts)
  print "Tag counts:"
  pprint.pprint(tag_counts)
  print
  print "Tag probabilities:"
  pprint.pprint(tag_probabilities)
  print
  print "Token counts:"
  pprint.pprint(tag_to_token_counts)
  print
  print "Token probabilities:"
  pprint.pprint(tag_to_token_probabilities)
  predictions = predict_clauses(clauses, tag_counts, tag_to_token_counts, correct_spelling, include_bigrams, include_trigrams)
  for clause, prediction_list in zip(clauses, predictions):
    print "Clause:", clause
    print "Predictions:"
    pprint.pprint(prediction_list)

if __name__ == '__main__':
  run_tests(correct_spelling=True, include_bigrams=True, include_trigrams=False)
