#!/usr/bin/python

import json

with open('cumulative_ideas_categories.json') as fin:
  rows = json.loads(fin.read())

with open('cumulative_data.csv', 'w') as fout:
  header = ['tuple_id', 'id', 'condition', 'response_num', 'num_cum_ideas', 'num_cum_categories']
  fout.write('\t'.join(header) + '\n')
  for row in rows:
    fout.write('\t'.join([str(x) for x in row]) + '\n')
