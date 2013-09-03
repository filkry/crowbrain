#!/usr/bin/python

# Set of routines for transforming strings into higher-level codes
# and for extracting features

import popen2
import re
import nltk
from nltk.corpus import wordnet as wn
import operator
import math

# Random utility function
# Assumes two sets of features passed in
def bag_of_words_cos_similarity(l, r):
  if len(l) == 0 or len(r) == 0:
    return 0
  common_set = l.intersection(r)
  num_in_common = len(common_set)
  denom = reduce(operator.mul, [math.sqrt(len(x)) for x in [l,r]], 1.0)
  return num_in_common / denom

def sanitize_raw_input(input_string):
# Potentially do this, too:
  #input_string = input_string.replace(" '", "'").replace("sn't", "s not").replace("n't", "not")
  return input_string.encode('utf-8')

def tokenize_no_punct(s):
#    self.tokens = nltk.word_tokenize(self.answer)
  s = re.sub(r'[^a-zA-Z0-9 ]', '', s)
  return s.split()

def tokenize_minimal_punct(s):
  s = re.sub(r'[^a-zA-Z0-9,.!? ]', '', s)
  return s.split()

def tokenize_whitespace(s):
  return s.split()

def tokenize_lowercase_no_punct(s):
  s = s.lower()
  return tokenize_no_punct(s)

# Returns a feature list that just includes the tokens
def extract_identity(tokens):
  return [[t] for t in tokens]

def extract_spelling_suggestions(tokens):
  return create_feature_list(tokens, spellcheck_word)

def extract_best_spelling_suggestions(tokens):
  return create_feature_list(tokens, get_best_spelling)

def extract_number_codes(tokens):
  return create_feature_list(tokens, code_number)

def extract_punct_codes(tokens):
  return create_feature_list(tokens, code_punctuation)

def extract_has_punct(tokens):
  return create_feature_list(tokens, code_has_punctuation)

# Adds codes if words other than first word are capitalized
def extract_later_caps_codes(tokens):
  if len(tokens) > 1:
    return create_feature_list(tokens[0], lambda x:[]) + create_feature_list(tokens[1:], code_caps)
  return create_feature_list(tokens, lambda x:[])

# Returns feature list of synonyms for tokens passed in. If pos_tags is None, calculates them itself
def extract_synonyms(tokens, pos_tags=None):
  return _extract_nyms(tokens, pos_tags, get_synonyms)
  
# Returns feature list of hypernyms for tokens passed in. If pos_tags is None, calculates them itself
def extract_hypernyms(tokens, pos_tags=None):
  return _extract_nyms(tokens, pos_tags, get_hypernyms)

# Extracts syno/hypernyms for every token in the feature list
def _extract_nyms(tokens, pos_tags, nym_fn):
  def _get_string_token(t):
    if type(t) == str or type(t) == unicode:
      return t
    return t[0]
  tokens = [_get_string_token(t) for t in tokens]
  if not pos_tags:
    pos_tags = get_pos_tags(tokens)
  return [nym_fn(t, pos_tag) for t, pos_tag in zip(tokens, pos_tags)]

def extract_lower_case(tokens):
  return create_feature_list(tokens, lambda x: [x.lower()])
  
def extract_no_punct_tokens(tokens):
  regex = re.compile('[^a-zA-Z0-9 ]')
  return create_feature_list(tokens, lambda x: [regex.sub('', x)])
  
def extract_stems(tokens):
  stemmer = nltk.PorterStemmer()
  return create_feature_list(tokens, lambda x: [stemmer.stem(x)])

def extract_function_composition(first, second):
  return lambda x: second(first(x))


# Requires a list of strings
# Returns a list of POS tags corresponding to each token
# TODO: Need to test to see what preprocessing is most appropriate to a phrase before doing the pos tagging
def get_pos_tags(tokens):
  return [tag[1] for tag in nltk.pos_tag(tokens)]

def merge_feature_lists(l, r):
  return [a+b for a,b in zip(l,r)]

def filter_stop_words(tokens):
  return [t for t in tokens if not t.lower() in nltk.corpus.stopwords.words('english')]
  
# Will add in original tokens, if passed in
# Returns a set
def feature_list_to_bag_of_words(feature_list, tokens=None):
  final_list = []
  for feature in feature_list:
    if type(feature) == str or type(feature) == unicode:
      final_list.append(feature)
    else:
      final_list.extend(list(feature_list_to_bag_of_words(feature, tokens)))
  if tokens:
    return set(final_list + tokens)
  return set(final_list)

# For each feature position, flattens all lists to a single set
# TODO: Test
def convert_to_sets_in_feature_positions(feature_list):
  return_list = []
  for t in feature_list:
    if type(t) == str or type(t) == unicode:
      return_list.append(set([t]))
    elif t:
      this_set = reduce(set.union, convert_to_sets_in_feature_positions(t), set())
      return_list.append(this_set)
    else:
      return_list.append(set())
  return return_list

# Assumes a feature list passed in (where each element of the list
# is either a string or a list)
# If the element is a list, it generates bigrams for each possible combination
# Returns a set of unique bigrams
def get_bigrams(feature_list):
  bigrams = []
  for i in range(0, len(feature_list)-1):
    l = feature_list[i]
    r = feature_list[i+1]
    if type(l) != str:
      for token in l:
        bigrams.extend(list(get_bigrams([token, r])))
    elif type(r) != str:
      for token in r:
        bigrams.extend(list(get_bigrams([l, token])))
    else:
      bigrams.append((l,r))
  return set(bigrams)

# Generates bigrams that skip a token, so "a b c" would yield a single bigram of "a c"
def get_skip_one_bigrams(feature_list):
  bigrams = []
  for i in range(0, len(feature_list)-2):
    l = feature_list[i]
    r = feature_list[i+2]
    if type(l) != str:
      for token in l:
        bigrams.extend(list(get_bigrams([token, r])))
    elif type(r) != str:
      for token in r:
        bigrams.extend(list(get_bigrams([l, token])))
    else:
      bigrams.append((l,r))
  return set(bigrams)

# Returns a list of tokens. In the case of a token position having a list,
# uses the first token in the list
# TODO: Untested code
def get_top_level_feature_list(feature_list):
  return_list = []
  for t in feature_list:
    if t:
      return_list.append(t[0])
    else:
      return_list.append([])
  return return_list

# Takes a list of tokens or lists and returns a feature list where each element is
# a list of features derived from the original token. feature_fn must return a list,
# and is passed a single token
def create_feature_list(tokens, feature_fn):
  return_list = []
  for t in tokens:
    entry_list = []
    if type(t) == str or type(t) == unicode:
      entry_list = feature_fn(t)
    else: # Debatable whether I want to recurse on the list of output, or simply peel off the first element (the original token)
      entry_list = create_feature_list(t, feature_fn)
    return_list.append(entry_list)
  return return_list

def code_caps(token, regex=re.compile('.*[A-Z].*')):
  if regex.match(token):
    return ['__CAP__']
  return []

# Returns a code in a list if the token is a number, otherwise []
def code_number(t):
  try:
    float(t)
    return ['__NUM__']
  except ValueError:
    pass
  return []
  
punct_dict = {
  '!' : '__EXCLAMATION_POINT__',
  '@' : '__AT_SYMBOL__',
  '#' : '__HASH__',
  '$' : '__DOLLAR_SIGN__',
  '%' : '__PERCENT__',
  '^' : '__HAT__',
  '&' : '__AMPERSAND__',
  '*' : '__ASTERISK__',
  '(' : '__OPEN_PAREN__',
  ')' : '__CLOSE_PAREN__',
  '-' : '__MINUS__',
  '_' : '__UNDERSCORE__',
  '+' : '__PLUS__',
  '=' : '__EQUALS__',
  '[' : '__LEFT_ANGLE_BRACKET__',
  ']' : '__RIGHT_ANGLE_BRACKET__',
  '{' : '__LEFT_CURLY_BRACE',
  '}' : '__RIGHT_CURLY_BRACE',
  '\\' : '__BACK_SLASH',
  '|' : '__PIPE__',
  ';' : '__SEMICOLON__',
  ':' : '__COLON__',
  ',' : '__COMMA__',
  '.' : '__PERIOD__',
  '<' : '__LESS_THAN__',
  '>' : '__GREATER_THAN__',
  '/' : '__FORWARD_SLASH__',
  '?' : '__QMARK__',
  ':)' : '__SMILEY__',
  ';)' : '__WINKY__',
# Should have a whole separate thing on emoticons
  '--' : '__DOUBLE_DASH__',
  '//' : '__DOUBLE_FORWARD_SLASH__',
  '`' : '__BACK_TICK__',
  '~' : '__TILDE__',
  '"' : '__DOUBLE_QUOTE__',
  "'" : '__SINGLE_QUOTE__',
}

# Returns a list of punctuation codes for all of the punctuation found in the passed in token
def code_punctuation(t):
  return_set = []
  for p in punct_dict.keys():
    if p in t:
      return_set.append(punct_dict[p])
  if '(' in t and ')' in t:
    return_set.append('__OPEN_CLOSE_PAREN__')
  if "'" in t or '"' in t:
    return_set.append('__QUOTE__')
  return return_set
  
def code_has_punctuation(t):
  for p in punct_dict:
    if p in t:
      return ['__PUNCT__']
  return []
  
# Returns a list of synonyms
def get_synonyms(word, pos_tag=None):
  word = word.lower()
  return filter_synsets_to_words(wn.synsets(word), pos_tag)

# Returns a list of hypernyms
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
            elif word:
              self._f.tochild.write(word+'\n')
              self._f.tochild.flush()
              s = self._f.fromchild.readline().strip().lower()
              self._f.fromchild.readline() #skip the blank line
              if not s:
                continue
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

spellchecker = None
# Returns list of suggested spellings for word
def spellcheck_word(word):
  global spellchecker
  if not spellchecker:
    spellchecker = hunspell()
  word = re.sub(r"[^a-zA-Z0-9-']", '', word)
  results = spellchecker(word)
  if results and results[0]:
    return results[0]
  return []

# Returns list with best spelling suggestion in list, or original word
# if no suggestions
# TODO: Make this probabilistic; choose correction most frequently found in some corpus
def get_best_spelling(word):
  suggestions = spellcheck_word(word)
  if suggestions:
    return [suggestions[0]]
  return [word]


if __name__ == "__main__":
  import pprint
  test_sentences = [
    "Teh button you should press is Save",
    "Enter in a value between 100 and 150",
    '''"This," she said, "is why I don't reaally do Mondays."'''
  ]
  tokenization_fns = [
    tokenize_whitespace,
    tokenize_no_punct,
    tokenize_minimal_punct,
  ]
  feature_extraction_fns = [
    extract_spelling_suggestions,
    extract_best_spelling_suggestions,
    extract_number_codes,
    extract_punct_codes,
    extract_later_caps_codes,
    extract_synonyms,
    extract_hypernyms,
    extract_lower_case,
    extract_stems,
    extract_function_composition(extract_best_spelling_suggestions, extract_stems),
  ]
  for s in test_sentences:
    print s
    s = sanitize_raw_input(s)
    print s
    for tokenize_fn in tokenization_fns:
      tokens = tokenize_fn(s)
      print "Tokens:", tokenize_fn
      pprint.pprint(tokens)
      print "POS tags:"
      pos_tags = get_pos_tags(tokens)
      print pos_tags
      for extraction_fn in feature_extraction_fns:
        print "Extraction fn:", extraction_fn
        print(extraction_fn(tokens))
      print "Stop words filtered:"
      print(filter_stop_words(tokens))
      stems = extract_stems(tokens)
      numbers = extract_number_codes(tokens)
      punct = extract_punct_codes(tokens)
      merged = merge_feature_lists(stems,
                 merge_feature_lists(numbers, punct))
      print "Merged stems, numbers, punct"
      print(merged)
      print "bigrams:"
      pprint.pprint(get_bigrams(merged))
      print "bag of words:"
      print feature_list_to_bag_of_words(merged, tokens)
