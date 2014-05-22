import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, os
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

model_string = """
}
"""

def plot_geometric_series(ax, a, r, max_x):
    vals = [a * pow(r, i) for i in range(max_x)]
    ax.plot(range(max_x), vals, color='k')

def plot_log_log_zipf(df):
    freqs = idea_freqs(df)
    ranks = range(1, len(freqs) + 1)

    ys = np.log(freqs)
    xs = np.log(ranks)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys)
    ax.set_xlabel('log(rank)')
    ax.set_ylabel('log(frequency count)')
    
    plt.show()

def estimate_r(series):
    rs = []
    for prev_s, s in zip(series[:-1], series[1:]):
        rs.append(s / prev_s)

    #print(rs[:100])
    return np.mean(rs)

def idea_freqs(df):
    icounts = defaultdict(int)
    ideas = df['idea']
    for i in ideas:
        icounts[i] += 1

    freqs = sorted(icounts.values(), reverse=True)
    return freqs


def plot_empirical_p_idea(df):
    freqs = idea_freqs(df)
    vals = [v/len(vals) for v in freqs] # normalize to sum to 1

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.bar(range(len(vals)), vals, color='r', alpha=0.25)

    r = estimate_r(vals)
    print("Estimated r=%f" % r)
    #plot_geometric_series(ax, 0.05, 0.95, 1200)
    plot_geometric_series(ax, 0.05, r, 1200)

    plt.show()


def filter_today(df):
    df = df[(df['question_code'] == 'iPod')]# | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
import sys
if __name__ == '__main__':
    print(os.path.basename(__file__))
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    #plot_empirical_p_idea(df)
    plot_log_log_zipf(df)
