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
    int<lower=0> N; // number of instances
    int<lower=0,upper=1> success[N]; // whether there was a novel idea at this point
    int<lower=0,upper=N> x[N]; // ordinal position of the instance in its condition
    int<lower=0,upper=N> M; // number of turkers
    int<lower=1,upper=M> t[N]; // which turker this is
}

parameters {
    real <lower=-10, upper=0> rate[M];
}

model {
    for (i in 1:N) {
        success[i] ~ bernoulli( exp(rate[t[i]] *  x[i]) );
    }
}
"""

def compute_model(turkers, turker_rates):
    ys = [0]
    for t in turkers:
        rate = turker_rates[t]
        ys.append(ys[-1] + math.exp(rate * x))
    return ys

def gen_uniques_counts(adf, field):
    adf = adf.sort(columns=['submit_datetime', 'answer_num'], ascending=[1, 1])
    uniques = set()
    
    counts = []
    turkers = []
    for i in adf.index:
        thing = adf[field][i]
        uniques.add(thing)
        counts.append(len(uniques))
        turkers.append(int(adf['worker_int'][i]))
    return counts, turkers

def gen_data(df, field):
    dat = defaultdict(list)

    for nr in set(df['num_requested']):
        nrdf = df[df['num_requested'] == nr]
        for qc in set(df['question_code']):
            qcdf = nrdf[nrdf['question_code'] == qc]
            uniques_counts, turkers = gen_uniques_counts(qcdf, field)
            temp = [0] + uniques_counts
            uniques_diffs = np.diff(temp)
            
            dat['success'].extend(uniques_diffs)
            dat['x'].extend(range(len(uniques_diffs)))
            dat['t'].extend(turkers)

    # Assertion based on potential error identified by Bob Carpenter on the
    # Stan mailing list (success should be 1 at t=0 with P = 1)
    for x, s in zip(dat['x'], dat['success']):
        assert(not (x == 0 and s == 0))
                            
    return {'M': len(set(dat['t'])),
            'x': [int(x) for x in dat['x']],
            'success': [int(s) for s in dat['success']],
            'N': len(dat['x']),
            't': [int(t) for t in dat['t']]}

def view_fit(df, field, fit):
    la = fit.extract(permuted=True)
    rates = [mystats.mean_and_hpd(la['rate'][:,i], 0.95)
            for i in range(la['rate'].shape[1])]
    print(rates)

    #plot_rate_model(rates, df, field)

def plot_rate_model(rates, df, field):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("number of instances received")
    ax.set_ylabel("number of unique instances")

    max_x = max(len(adf) for n, adf in df.groupby(['num_requested']))
    xs = range(max_x)

    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_x)

    # plot the hpd area
    #bottom_ys = compute_model(max_x, rate[1])
    #top_ys = compute_model(max_x, rate[2])
    #ax.fill_between(xs, bottom_ys, top_ys, color='g', alpha=0.25)

    # plot the line and model line for each condition
    for name, adf in df.groupby(['num_requested']):
        ys, turkers = gen_uniques_counts(adf, field)
        ax.plot(xs[:len(ys)], ys, '-', color='k')

        ys = compute_model(max_x, rate[0])
        #ax.plot(xs[:len(ys)], ys, '--', color='k')

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

    cat = pd.Categorical(idf['worker_id'])
    idf['worker_int'] = pd.Series(cat.labels, index=idf.index)
    idf['worker_int'] += 1

    dat = gen_data(idf, 'idea')
    json.dump(dat, open('cache/rate_counting_types_model_stan_dat.json', 'w'))

    model_file = 'cache/rate_counting_types_model_stan'
    if os.path.isfile(model_file): 
        print("Warning: loading model from file.")
        model = pickle.load(open(model_file, 'rb'))
    else:
        model = pystan.StanModel(model_code=model_string)
        pickle.dump(model, open(model_file, 'wb'))

    fit = model.sampling(data=dat, iter=1000, chains=3)

    view_fit(idf, 'idea', fit)

