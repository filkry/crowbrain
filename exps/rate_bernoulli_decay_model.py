import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, math
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
    real <lower=0, upper=1> min_rate;
}

model {
    real mu[N];
    real theta;
    int t;
    for (i in 1:N) {
        theta <- min_rate + exp(rate * x[i]) * (1-min_rate);
        novel[i] ~ bernoulli(theta);
    }
}
"""

def model_integral_predict(max_x, rate, min_rate):
    ys = [0]
    for x in range(1, max_x):
        ys.append(ys[-1] + math.exp(rate * x) + min_rate)
    return ys

def gen_uniques_counts(adf, field):
    adf = adf.sort(columns=['submit_datetime', 'answer_num'], ascending=[1, 1])
    uniques = set()
    
    counts = []
    for thing in adf[field]:
        uniques.add(thing)
        counts.append(len(uniques))
    return counts

def gen_model_data(df, rmdf, cdf, ifs):
    field = 'idea'
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

def view_fit(df, field, la):
    rate = mystats.mean_and_hpd(la['rate'], 0.95)
    min_rate = mystats.mean_and_hpd(la['min_rate'], 0.95)

    plot_model(rate, min_rate, df, field)

def plot_model(rate, min_rate, df, field):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("number of instances received")
    ax.set_ylabel("number of unique ideas")

    max_x = max(len(adf) for n, adf in df.groupby(['num_requested']))
    xs = range(max_x)

    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_x)

    # plot the hpd area
    bottom_ys = model_integral_predict(max_x, rate[1], min_rate[1])
    top_ys = model_integral_predict(max_x, rate[2], min_rate[2])
    ax.fill_between(xs, bottom_ys, top_ys, color='g', alpha=0.25)

    # plot the line for each condition
    for name, adf in df.groupby(['num_requested']):
        ys = gen_uniques_counts(adf, field)
        ax.plot(xs[:len(ys)], ys, '-', color='k')

    # plot the model line
    ys = model_integral_predict(max_x, rate[0], min_rate[0])
    ax.plot(xs[:len(ys)], ys, '--', color='k')

    plt.show()

# TODO: this could be done with passed parameters
def filter_today(df):
    df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, ifs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(ifs, idf)

    n_iter = 1500
    n_chains = 3

    dat = gen_model_data(df, rmdf, cdf, ifs)
    param_walks = modeling.compile_and_fit(model_string, dat, n_iter, n_chains)

    view_fit(df, 'idea', param_walks[0])

