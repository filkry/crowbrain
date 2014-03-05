import pandas as pd
import numpy as np
import re, pystan, format_data, modeling
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

model_string = """
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

def model_predict(x, y_scale, rate):
    return y_scale * (x ** rate)

def gen_uniques_counts(adf, field):
    adf = adf.sort(columns=['submit_datetime', 'answer_num'], ascending=[1, 1])
    uniques = set()
    
    counts = []
    for thing in adf[field]:
        uniques.add(thing)
        counts.append(len(uniques))
    return counts

def gen_model_data(df, rmdf, clusters_df, idea_forest):
    field = 'idea' # TODO?
    dat = defaultdict(list)

    for nr in set(df['num_requested']):
        nrdf = df[df['num_requested'] == nr]
        for qc in set(df['question_code']):
            qcdf = nrdf[nrdf['question_code'] == qc]
            uniques_counts = gen_uniques_counts(qcdf, field)
            dat['y'].extend(uniques_counts)
            dat['x'].extend(range(1, len(uniques_counts) + 1))
                            
    assert(len(dat['x']) == len(dat['y']))

    # Converting to tuple for immutability so the structure is hashable
    return {'x': tuple(dat['x']),
            'y': tuple(dat['y']),
            'N': len(dat['x'])}

def view_model_fit(df, field, la):
    rates = la['rate']
    y_scale = np.mean( la['y_scale'])
    
    rate = mystats.mean_and_hpd(rates, 0.95)

    print("Mean rate:", rate[0])
    print("Mean y_scale:", y_scale)
    print("Mean sigma:", np.mean(la['sigma']))

    plot_model(y_scale, (rate[1], rate[2]), rate[0], df, field)

def plot_model(y_scale, rate_hpd, rate_mean, df, field):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("number of instances received")
    ax.set_ylabel("number of unique ideas")

    max_x = max(len(adf) for n, adf in df.groupby(['num_requested']))
    xs = range(max_x)

    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_x)

    # plot the hpd area
    bottom_ys = [model_predict(x, y_scale, rate_hpd[0]) for x in xs]
    top_ys = [model_predict(x, y_scale, rate_hpd[1]) for x in xs]
    ax.fill_between(xs, bottom_ys, top_ys, color='g', alpha=0.25)

    # plot the line for each condition
    for name, adf in df.groupby(['num_requested']):
        ys = gen_uniques_counts(adf, field)
        ax.plot(xs[:len(ys)], ys, '-', color='k')

    # plot the model line
    ys = [model_predict(x, y_scale, rate_mean) for x in xs]
    ax.plot(xs[:len(ys)], ys, '--', color='k')

    # plot the 1:1 line
    ys = [x for x in xs]
    ax.plot(xs, ys, '--', color='k', alpha=0.5)

    plt.show()
    
def hyp_test_exclude_one(params):
    rates = params['rate']
    left, right = mystats.hpd(rates, 0.95)
    return 1 > right

def filter_today(df):
    df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    n_iter = 1500
    n_chains = 3

    dat = gen_model_data(df, None, None, None) 
    param_walks = modeling.compile_and_fit(model_string, dat, n_iter, n_chains)

    view_model_fit(idf, 'idea', param_walks[0])

    sim_passes = modeling.simulate_error_hypothesis(10, model_string, n_iter, n_chains,
            gen_model_data, hyp_test_exclude_one, cfs, df)
    print("Exclude one hypothesis held in %i/10 cases" % sim_passes)
