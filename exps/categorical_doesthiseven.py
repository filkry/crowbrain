import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, os
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

model_string = """
data {
    int N; // number of instances
    int K; // number of categories
    int<lower=1, upper=K> cat[N]; // which category each instance is
}

parameters {
    simplex[K] theta; // category prevalence
}

model {
    for (i in 1:N) {
        cat[i] ~ categorical(theta);
    }
}
"""

def model_predict():
    raise NotImplementedError

def categorical_series(series):
    next_val = 1
    seen = dict()
    
    s = pd.Series(index=series.index, dtype='int64')
    for ix in series.index:
        v = series[ix]
        if v not in seen:
            seen[v] = next_val
            next_val += 1
        s[ix] = seen[v]

    return s

def gen_data(df, rmdf, clusters_df, idea_forest):
    N = len(df)
    R = len(rmdf)
    K = len(set(df['subtree_root']))

    adf = df.copy()
    cat = categorical_series(adf['subtree_root'])

    adf['category_oneid'] = cat

    dat = {
            'N': N,
            'K': K,
            'cat': [int(c) for c in cat],
    }

    return adf, dat


def view_fit():
    raise NotImplementedError

def plot_fit(param_walks, dat):
    print(param_walks[0]['theta'].shape)
    print(len(set(dat['cat'])))

    empirical_proportions = dict()
    for cat in range(max(dat['cat'])):
        count = sum([1 if c == cat else 0 for c in dat['cat']])
        empirical_proportions[cat] = count / len(dat['cat'])

    posterior_thetas = dict()
    for cat in range(max(dat['cat'])):
        posterior_thetas[cat] = np.mean(param_walks[0]['theta'][:,cat])

    for cat in range(max(dat['cat'])):
        print(empirical_proportions[cat] - posterior_thetas[cat])


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

    n_iter = 1500
    n_chains = 3

    annotated_df, dat = gen_data(df, rmdf, None, None) 
    param_walks = modeling.compile_and_fit(model_string, dat, n_iter, n_chains,
            manual_convergence_check = False)

    plot_fit(param_walks, dat)
