import pandas as pd
import numpy as np
import re, pystan, format_data, modeling
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

def filter_today(df):
    df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
def tree_question_boxplots(cdf, values_fn, ylabel): 
    fig = plt.figure()
    ax = fig.add_subplot(111)

    data = []
    qcs = set(cdf['question_code'])
    for qc in qcs:
        qcdf = cdf[cdf['question_code'] == qc]
        data.append(values_fn(qcdf))

    ax.boxplot(data)
    ax.set_ylim(0, max(max(d) for d in data) + 1)
    ax.set_xticks(range(1, len(qcs)+1))
    ax.set_xticklabels(list(qcs))
    ax.set_ylabel(ylabel)
    ax.set_xlabel('question')

    plt.show()

def extract_tree_depths(cdf):
    trees = cdf.groupby(['subtree_root'])
    return [max(tree['depth_in_subtree']) for name, tree in trees]

def extract_tree_breadths(cdf):
    not_leaf_df = cdf[cdf['is_leaf'] == 0]
    return not_leaf_df['num_children']

if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    tree_question_boxplots(cdf, extract_tree_depths, 'tree depth')
    tree_question_boxplots(cdf, extract_tree_breadths, 'non-leaf node degree') 

