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

def gen_uniques_counts(adf, field):
    adf = adf.sort(columns=['submit_datetime', 'answer_num'], ascending=[1, 1])
    uniques = set()
    
    counts = []
    for thing in adf[field]:
        uniques.add(thing)
        counts.append(len(uniques))
    return counts

def idea_rate_plot(df):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    styles = ['-', '--', '-.', ':']
    qcs = set(cdf['question_code'])
    max_instances = 0
    for i, qc in enumerate(qcs):
        qcdf = df[df['question_code'] == qc]
        uniques = gen_uniques_counts(qcdf, 'idea')
        xs = range(len(uniques))
        ax.plot(xs, uniques, styles[i] + 'k', label=qc)
        max_instances = max(max_instances, len(uniques))

    ax.legend()
    ax.set_xlim(0, max_instances)
    ax.set_ylim(0, max_instances)
    ax.plot(range(max_instances), range(max_instances), '-k', alpha=0.3)
    ax.set_xlabel('number of instances received')
    ax.set_ylabel('number of unique ideas')

    plt.show()

def gen_counts_table(df):
    res = ['\\begin{table}', '\\centering', '\\begin{tabular}{| r | l l l |}',
            '\\hline \\textbf{question} & number of instances & number of nodes & number of trees \\\\ \\hline',]
    for qc in set(df['question_code']):
        qcdf = df[df['question_code'] == qc]
        n_instances = len(qcdf)
        n_nodes = len(set(qcdf['idea']))
        n_trees = len(set(qcdf['subtree_root']))
        res.append('%s & %i & %i (%0.2f) & %i (%0.2f) \\\\' % (qc, n_instances,
            n_nodes, n_nodes / n_instances, n_trees, n_trees/n_instances))
    res.extend(['\\hline', '\\end{tabular}',
        '\\caption{Descriptive statistics for size of idea forests}',
        '\\label{tab:forest_descriptive_statistics}', '\\end{table}'])

    with open('tex/forests_counts_table.tex', 'w') as f:
        print('\n'.join(res), file=f)

def gen_riffs_table(df):
    res = ['\\begin{table}', '\\centering', '\\begin{tabular}{| r | l l l |}',
            '\\hline \\textbf{question} & riffs & source & chain \\\\ \\hline',]
    for qc in set(df['question_code']):
        qcdf = df[df['question_code'] == qc]
        n_riffs = len(qcdf[(qcdf['is_midmix'] == 1) | (qcdf['is_outmix'] == 1)])
        n_source = len(qcdf[qcdf['is_inmix'] == 1])
        n_chain = len(qcdf[qcdf['is_midmix'] == 1])

        res.append('%s & %i & %i & %i \\\\' % (qc, n_riffs, n_source, n_chain))
    res.extend(['\\hline', '\\end{tabular}',
        '\\caption[Descriptive statistics of brainstorming runs]{Run descriptive stats. Each value is the median number of instances with the given characteristic, where counts are normalized by the number of instances given in the run}',
        '\\label{tab:run_descriptive_stats}', '\\end{table}'])

    with open('tex/forests_riffs_table.tex', 'w') as f:
        print('\n'.join(res), file=f)



if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    gen_riffs_table(df)
    gen_counts_table(df)

    idea_rate_plot(df)
    tree_question_boxplots(cdf, extract_tree_depths, 'tree depth')
    tree_question_boxplots(cdf, extract_tree_breadths, 'non-leaf node degree') 
