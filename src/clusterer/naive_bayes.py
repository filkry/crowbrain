#!/usr/bin/python

import operator
import re
import pprint
import math

# Dependencies: nltk, WordNet for nltk, and hunspell (installed locally, only if spell checking)

TAG_COUNTS = 0
TAG_TO_FEATURE_COUNTS = 1
TAG_PROB = 2
TAG_TO_FEATURE_PROB = 3
STALE_PROBABILITIES_FLAG = 4
def create_model():
# First two dictionaries are the count dictionaries, second two are the probability dictionaries
  return [{}, {}, {}, {}, False]

# instance_features are the extracted features for an instance. instance_labels is a list of strings associated with this instance and its features
def update_model(model, instance_features, instance_labels):
  tag_counts = model[TAG_COUNTS]
  tag_to_feature_counts = model[TAG_TO_FEATURE_COUNTS]
  update_counts_from_feature_set(instance_features, instance_labels, tag_counts, tag_to_feature_counts)
  model[STALE_PROBABILITIES_FLAG] = True

# Removes the data represented by this instance's features and labels from the model
def remove_from_model(model, instance_features, instance_labels):
  tag_counts = model[TAG_COUNTS]
  tag_to_feature_counts = model[TAG_TO_FEATURE_COUNTS]
  remove_counts_from_feature_set(instance_features, instance_labels, tag_counts, tag_to_feature_counts)
  model[STALE_PROBABILITIES_FLAG] = True
  
# features are the extracted feature set
# Returns list of (tag, log(prob)) tuples, ordered most to least likely
def label_features(model, features):
  if model[STALE_PROBABILITIES_FLAG]:
    update_model_probabilities_from_counts(model)
  return get_tag_predictions_from_feature_set(features, model[TAG_PROB], model[TAG_TO_FEATURE_PROB])

def print_model_info(model):
  print "NB Model:"
  print "Num different labels:", len(model[TAG_COUNTS])
  print "Labels:", model[TAG_COUNTS].keys()
  print "Total number of label data points:", reduce(operator.add, model[TAG_COUNTS].values(), 0)

def pretty_print_model(model):
  import pprint
  print "Tag counts:"
  pprint.pprint(model[TAG_COUNTS])
  print
  print "Tag probabilities:"
  pprint.pprint(model[TAG_PROB])
  print
  print "Token counts:"
  pprint.pprint(model[TAG_TO_FEATURE_COUNTS])
  print
  print "Token probabilities:"
  pprint.pprint(model[TAG_TO_FEATURE_PROB])

# Creates a count dictionary from the items passed in
def init_count_dict(items):
  counts = {}
  for item in items:
    if not item in counts:
      counts[item] = 1.0
    else:
      counts[item] = counts[item] + 1.0
  return counts

# Adds counts from src_dict to dest_dict
def add_counts(dest_dict, src_dict):
  for t in src_dict:
    if not t in dest_dict:
      dest_dict[t] = src_dict[t]
    else:
      dest_dict[t] = dest_dict[t] + src_dict[t]

# Removes counts in src_dict from dest_dict
def remove_counts(dest_dict, src_dict):
  for t in src_dict:
    if t in dest_dict:
      dest_dict[t] = dest_dict[t] - src_dict[t]
      if dest_dict[t] < 1:
        del dest_dict[t]

def update_counts_from_feature_set(features, tags, tag_counts, tag_to_feature_counts):
  if tags:
    token_counts = init_count_dict(features)
    for tag in tags:
      if not tag in tag_counts:
        tag_counts[tag] = 1.0
      else:
        tag_counts[tag] = tag_counts[tag] + 1.0
      if not tag in tag_to_feature_counts:
        tag_to_feature_counts[tag] = {}
      add_counts(tag_to_feature_counts[tag], token_counts)

def remove_counts_from_feature_set(features, tags, tag_counts, tag_to_feature_counts):
  if tags:
    feature_counts = init_count_dict(features)
    for tag in tags:
      if tag in tag_counts:
        tag_counts[tag] = tag_counts[tag] - 1.0
      if tag in tag_to_feature_counts:
        remove_counts(tag_to_feature_counts[tag], feature_counts)

def convert_counts_to_probabilities(count_dict):
  total_counts = reduce(operator.add, count_dict.values(), 0.0)
  prob_dict = {}
  for t in count_dict:
    prob_dict[t] = count_dict[t] / total_counts
  return prob_dict

def update_model_probabilities_from_counts(model):
  model[TAG_PROB] = convert_counts_to_probabilities(model[TAG_COUNTS])
  tag_to_feature_counts = model[TAG_TO_FEATURE_COUNTS]
  tag_to_feature_probabilities = {}
  total_num_features = get_total_num_tag_features(tag_to_feature_counts)
  for tag in tag_to_feature_counts:
    tag_to_feature_probabilities[tag] = convert_counts_to_probabilities(tag_to_feature_counts[tag])
    tag_to_feature_probabilities[tag]['_MISSING_'] = 1.0/total_num_features
  model[TAG_TO_FEATURE_PROB] = tag_to_feature_probabilities
  model[STALE_PROBABILITIES_FLAG] = False
  
def get_total_num_tag_features(tag_to_feature_counts):
  return reduce(operator.add, [reduce(operator.add, tag_to_feature_counts[tag].values(), 0.0) for tag in tag_to_feature_counts], 0.0)

def get_tag_predictions_from_feature_set(features, tag_probabilities, tag_to_feature_probabilities):
  final_probs = {}
  for tag in tag_probabilities:
    if tag_probabilities[tag] <= 0:
      continue # Can happen if remove data for a tag with only one entry
    prob_sum = 0.0
    for feature in features:
      if not feature in tag_to_feature_probabilities[tag]:
        feature = '_MISSING_'
      this_prob = math.log(tag_to_feature_probabilities[tag][feature])
      prob_sum = prob_sum + this_prob
    final_probs[tag] = prob_sum + math.log(tag_probabilities[tag])
  predictions = [(tag, int(final_probs[tag])) for tag in final_probs]
  predictions.sort(lambda l,r: -cmp(l[1],r[1]))
  return predictions


def run_tests():
  clauses_and_labels = [
    ('Click on teh color button and type in a setting of 128 or do a #FFFF00 color :)', ['do', 'app part',]),
    ('Now click on the paintbrush to start painting', ['do', 'app part']),
    ('If you want, you can color this in with blue', ['optional']),
    ('If this looks too harsh, you can fix it later', ['optional']),
  ]
  clauses = [ct[0] for ct in clauses_and_labels] + [
    "In this tutorial, we'll make your brown eyes blue",
    'Press the button labeled "Save" to save your document',
    "If it doesn't come out OK, you can always undo it",
  ]
  def extract_features(clause):
    import feature_extraction as fe
    
    clause = fe.sanitize_raw_input(clause)
    # This will strip out some potentially useful punct.
#    tokens = fe.tokenize_minimal_punct(clause)
    tokens = fe.tokenize_whitespace(clause)
    spell_correct_tokens = fe.extract_best_spelling_suggestions(tokens)
    synonyms = fe.merge_feature_lists(fe.extract_synonyms(tokens), fe.extract_synonyms(spell_correct_tokens))
    hypernyms = fe.merge_feature_lists(fe.extract_hypernyms(tokens), fe.extract_hypernyms(spell_correct_tokens))
    punct_tokens = fe.extract_punct_codes(tokens)
    number_tokens = fe.extract_number_codes(tokens)
    caps_tokens = fe.extract_later_caps_codes(tokens)
    lower_tokens = fe.extract_lower_case(tokens)
    stem_tokens = fe.extract_stems(lower_tokens)
    bigram_feature_list = fe.merge_feature_lists(stem_tokens,
                            fe.merge_feature_lists(punct_tokens,
                              fe.merge_feature_lists(number_tokens, caps_tokens)))
    bigrams = fe.get_bigrams(bigram_feature_list)
    bag_of_words = fe.feature_list_to_bag_of_words(fe.merge_feature_lists(bigram_feature_list,
                                                    fe.merge_feature_lists(synonyms,
                                                      fe.merge_feature_lists(hypernyms, lower_tokens))))
    return bag_of_words.union(bigrams)

  clause_to_features = {}
  for clause in clauses:
    clause_to_features[clause] = extract_features(clause)
  for clause in clause_to_features:
    print "Clause:", clause
    print "Features:", clause_to_features[clause]
  model = create_model()
  for clause, labels in clauses_and_labels:
    update_model(model, clause_to_features[clause], labels)
  pretty_print_model(model)
  for clause in clause_to_features:
    print "Clause:", clause
    print "Predictions:"
    features = clause_to_features[clause]
    pprint.pprint(label_features(model, features))
    if clause in [x[0] for x in clauses_and_labels]:
      index = [x[0] for x in clauses_and_labels].index(clause)
      labels = clauses_and_labels[index][1]
      print "Removing data for clause"
      remove_from_model(model, features, labels)
#      pretty_print_model(model)
      print "New prediction:"
      pprint.pprint(label_features(model, features))
      print "Adding again, and redoing prediction:"
      update_model(model, features, labels)
      pprint.pprint(label_features(model, features))
#      print "Model after adding back in:"
#      pretty_print_model(model)

if __name__ == '__main__':
  run_tests()
