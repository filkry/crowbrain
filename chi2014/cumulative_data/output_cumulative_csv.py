#!/usr/bin/python

import json
import glob

files = glob.glob("*.json")

for f in files:
  prefix = f[:-5]
  print prefix

  with open(f) as fin:
    rows = json.loads(fin.read())

  with open(prefix + '.csv', 'w') as fout:
    header = ['response_id', 'id', 'condition', 'response_num', 'num_cum_ideas', 'num_cum_categories', 'run_id', 'num_responses_requested', 'user_id']
    fout.write('\t'.join(header) + '\n')
    for row in rows:
      tuple_id = row[0]
      num_responses_requested = tuple_id[-2]
      run_id = tuple_id[:3]
      user_id = tuple_id[0]
      fout.write('\t'.join([str(x) for x in row]) + '\t' + str(run_id) + '\t' + str(num_responses_requested) + '\t' + user_id + '\n')
