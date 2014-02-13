import pandas as pd
import scipy as sp
import numpy as np
import csv, os, re, itertools, random, math, json, pystan, format_data, pickle
import scipy.stats as stats
import networkx as nx
from imp import reload
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict


model_string = """
data {
    int N; // number of instances
    int novel[N]; // whether there was a novel idea at this point
    int x[N]; // ordinal position of the instance in its condition
}

parameters {
    real <upper=0> rate;
}

model {
    real mu[N];
    real theta;
    int t;
    for (i in 1:N) {
        theta <- exp(rate * x[i]);
        novel[i] ~ bernoulli(theta);
    }
}
"""

def compute_model(max_x, rate):
    ys = [0]
    for x in range(1, max_x):
        ys.append(ys[-1] + math.exp(rate * x))
    return ys

def gen_uniques_counts(adf, field):
    adf = adf.sort(columns=['submit_datetime', 'answer_num'], ascending=[1, 1])
    uniques = set()
    
    counts = []
    for thing in adf[field]:
        uniques.add(thing)
        counts.append(len(uniques))
    return counts

def gen_data(df, field):
    dat = defaultdict(list)

    for nr in set(df['num_requested']):
        nrdf = df[df['num_requested'] == nr]
        for qc in set(df['question_code']):
            qcdf = nrdf[nrdf['question_code'] == qc]
            uniques_counts = gen_uniques_counts(qcdf, field)
            temp = [0] + uniques_counts
            uniques_diffs = np.diff(temp)
            
            dat['novel'].extend(uniques_diffs)
            dat['x'].extend(range(len(uniques_diffs)))
                            
    return {'M': len(set(dat['t'])),
            'x': dat['x'],
            'novel': dat['novel'],
            'N': len(dat['x'])}

def view_fit(df, field, fit):
    la = fit.extract(permuted=True)
    rate = mystats.mean_and_hpd(la['rate'], 0.95)

    plot_rate_model(rate, df, field)

def plot_rate_model(rate, df, field):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("number of instances received")
    ax.set_ylabel("number of unique instances")

    max_x = max(len(adf) for n, adf in df.groupby(['num_requested']))
    xs = range(max_x)

    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_x)

    # plot the hpd area
    bottom_ys = compute_model(max_x, rate[1])
    top_ys = compute_model(max_x, rate[2])
    ax.fill_between(xs, bottom_ys, top_ys, color='g', alpha=0.25)

    # plot the line for each condition
    for name, adf in df.groupby(['num_requested']):
        ys = gen_uniques_counts(adf, field)
        ax.plot(xs[:len(ys)], ys, '-', color='k')

    # plot the model line
    ys = compute_model(max_x, rate[0])
    ax.plot(xs[:len(ys)], ys, '--', color='k')

    plt.show()

# TODO: this could be done with passed parameters
def filter_today(df):
    df = df[df['question_code'] == 'iPod']
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    #df, rmdf, clusters_df, cluster_forests = format_data.mk_redundant(idf, cfs)

    dat = gen_data(idf, 'idea')

    model_file = 'cache/rate_counting_model_stan'
    if os.path.isfile(model_file): 
        print("Warning: loading model from file.")
        model = pickle.load(open(model_file, 'rb'))
    else:
        model = pystan.StanModel(model_code=model_string)
        pickle.dump(model, open(model_file, 'wb'))

    fit = model.sampling(data=dat, iter=1000, chains=3)

    view_fit(idf, 'idea', fit)

