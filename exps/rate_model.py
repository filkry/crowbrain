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


rate_model_string = """
data {
    int N; // number of instances
    real y[N]; // the number of ideas or categories receieved up to and including instance n
    int x[N]; // ordinal position of the instance in its condition
}

parameters {
    real<lower=0, upper=1> rate;
    real<lower=0> y_scale;
    real<lower=0> sigma;
}

model {
    real mu[N];
    for (i in 1:N) {
        mu[i] <- y_scale * pow(x[i], rate);
        y[i] ~ normal(mu[i], sigma);
    }
}
"""

def compute_rate_model(x, y_scale, rate):
    return y_scale * (x ** rate)

def gen_uniques_counts(adf, field):
    adf = adf.sort(columns=['submit_datetime', 'answer_num'], ascending=[1, 1])
    uniques = set()
    
    counts = []
    for thing in adf[field]:
        uniques.add(thing)
        counts.append(len(uniques))
    return counts

def gen_rate_model_data(df, field):
    dat = defaultdict(list)

    for nr in set(df['num_requested']):
        nrdf = df[df['num_requested'] == nr]
        for qc in set(df['question_code']):
            qcdf = nrdf[nrdf['question_code'] == qc]
            uniques_counts = gen_uniques_counts(qcdf, field)
            dat['y'].extend(uniques_counts)
            dat['x'].extend(range(1, len(uniques_counts) + 1))
                            
    assert(len(dat['x']) == len(dat['y']))

    return {'x': dat['x'],
            'y': dat['y'],
            'N': len(dat['x'])}

def view_rate_model_fit(df, field, fit):
    la = fit.extract(permuted=True)
    rates = la['rate']
    y_scale = np.mean( la['y_scale'])
    
    left, right = mystats.hpd(rates, 0.95)
    rate_mean = np.mean(rates)

    plot_rate_model(y_scale, (left, right), rate_mean, df, field)

def plot_rate_model(y_scale, rate_hpd, rate_mean, df, field):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("number of instances received")
    ax.set_ylabel("number of unique instances")

    max_x = max(len(adf) for n, adf in df.groupby(['num_requested']))
    xs = range(max_x)

    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_x)

    # plot the hpd area
    bottom_ys = [compute_rate_model(x, y_scale, rate_hpd[0]) for x in xs]
    top_ys = [compute_rate_model(x, y_scale, rate_hpd[1]) for x in xs]
    ax.fill_between(xs, bottom_ys, top_ys, color='g', alpha=0.25)

    # plot the line for each condition
    for name, adf in df.groupby(['num_requested']):
        ys = gen_uniques_counts(adf, field)
        ax.plot(xs[:len(ys)], ys, '-', color='k')

    # plot the model line
    ys = [compute_rate_model(x, y_scale, rate_mean) for x in xs]
    ax.plot(xs[:len(ys)], ys, '--', color='k')

    plt.show()
    
def hyp_test_rate_exclude_one(df, field, cache_key):
    dat = gen_exp_model_nocond_data(df, field)
    
    def testfunc(fits):
        la = fits[0].extract(permuted=True)
        rates = la['rate']
        left, right = mystats.hpd(rates, 0.95)
        return 1 > right
    
    fits, success = mystats.stan_hyp_test([dat], exp_model_nocond_string, testfunc, cache_key)
    return dat, fits, success

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

    dat = gen_rate_model_data(idf, 'idea')

    model_file = 'cache/rate_model_stan'
    if os.path.isfile(model_file): 
        model = pickle.load(open(model_file, 'rb'))
    else:
        model = pystan.StanModel(model_code=rate_model_string)
        pickle.dump(model, open(model_file, 'wb'))

    fit = model.sampling(data=dat, iter=500, chains=1)

    view_rate_model_fit(idf, 'idea', fit)

