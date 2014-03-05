import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, math
import matplotlib.pyplot as plt
import stats_fns as mystats
from functools import reduce
from collections import defaultdict, OrderedDict


model_string = """
data {
    int <lower=1> M; // number of participants
    int <lower=M> N; // number of instances
    int <lower=0, upper=1> novel[N]; // whether there was a novel idea at this point
    int <lower=0> x[N]; // ordinal position of the instance in its condition
    int <lower=1, upper=M> participant[N]; // which participant provided response
}

parameters {
    real <lower=-100, upper=0> rate[M];
    real <lower=0, upper=1> min_rate;

    real <lower=-100, upper=0> hyper_rate_mu;
    real <lower=0, upper=10> hyper_rate_sigma;
}

model {
    real theta;

    for (i in 1:M) {
        rate[i] ~ normal(hyper_rate_mu, hyper_rate_sigma);
    }

    for (i in 1:N) {
        theta <- min_rate + exp(rate[participant[i]] * x[i]) * (1-min_rate);
        increment_log_prob(bernoulli_log(novel[i], theta));
    }
}
"""

def model_integral_predict_p_decay(which_worker, worker_decay, min_rate):
    ys = [0]
    for x, worker in enumerate(which_worker):
        ys.append(ys[-1] + (1-min_rate) * math.exp(worker_decay[worker - 1] * x) + min_rate)
    return ys

def model_integral_predict_fixed_decay(max_x, decay_rate, min_rate):
    ys = [0]
    for x in range(1, max_x):
        ys.append(ys[-1] + (1-min_rate) * math.exp(decay_rate * x) + min_rate)
    return ys

def gen_uniques_counts(adf, field):
    adf = adf.sort(columns=['submit_datetime', 'answer_num'], ascending=[1, 1])
    uniques = set()
    
    counts = []
    for thing in adf[field]:
        uniques.add(thing)
        counts.append(len(uniques))
    return counts

def get_worker_ints(df, next_worker_int, worker_ints):
    """
    Generate a list of ints to represent workers from their worker IDs
    There are many smarter ways to do this, but this was the fastest for
    me to write
    """
    ret = []
    for worker_id in df['worker_id']:
        if not worker_id in worker_ints:
            worker_ints[worker_id] = next_worker_int
            next_worker_int += 1
        ret.append(worker_ints[worker_id])
    return next_worker_int, worker_ints, ret

def gen_model_data(df, rmdf, cdf, ifs):
    field = 'idea'
    dat = defaultdict(list)

    cur_worker_int = 1
    worker_ints = dict()

    for nr in set(df['num_requested']):
        nrdf = df[df['num_requested'] == nr]
        for qc in set(df['question_code']):
            qcdf = nrdf[nrdf['question_code'] == qc]
            uniques_counts = gen_uniques_counts(qcdf, field)

            temp = [0] + uniques_counts
            uniques_diffs = np.diff(temp)

            dat['novel'].extend(uniques_diffs)
            dat['x'].extend(range(len(uniques_diffs)))

            cur_worker_int, worker_ints, df_wints = get_worker_ints(qcdf, cur_worker_int, worker_ints)
            dat['participant'].extend(df_wints)

    return {
            'x': dat['x'],
            'novel': dat['novel'],
            'participant': dat['participant'],
            'M': cur_worker_int - 1,
            'N': len(dat['x'])}

def view_fit(df, field, la, dat):
    rate_array = la['rate']
    p_rates = [rate_array[:, i] for i in range(rate_array.shape[1])]
    sorted_rates = sorted(p_rates, key=np.mean)

    min_rate = np.mean(la['min_rate'])

    prediction_low_decay = model_integral_predict_fixed_decay(100, np.mean(sorted_rates[-1]), min_rate)
    prediction_high_decay = model_integral_predict_fixed_decay(100, np.mean(sorted_rates[0]), min_rate)
    print("Number of ideas generated with low decay rate:", prediction_low_decay[-1])
    print("Number of ideas generated with high decay rate:", prediction_high_decay[-1])

    plot_rate_posteriors(p_rates)

    #plot_cumulative_model(df, dat, p_rates, mystats.mean_and_hpd(la['min_rate'], 0.95), field)

def plot_cumulative_model(df, dat, p_rates, min_rate_hpd, field):
    # TODO: Keeping this around because it may be fixed later, but the
    # plot here has no meaning for now. The cumulative prediction
    # doesn't account for which condition we are in (it just appends all
    # participants)
    # TODO: assure p_rates comes in order
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("number of instances received")
    ax.set_ylabel("number of unique ideas")

    p_rate_hpds = [mystats.mean_and_hpd(p_rate, 0.95) for p_rate in p_rates]
    p_rate_mean = [hpd[0] for hpd in p_rate_hpds]
    p_rate_low = [hpd[1] for hpd in p_rate_hpds]
    p_rate_high = [hpd[2] for hpd in p_rate_hpds]

    xs = range(len(dat['participant']) + 1)

    # plot the hpd area
    bottom_ys = model_integral_predict_p_decay(dat['participant'],
            p_rate_low, min_rate_hpd[1])
    top_ys = model_integral_predict_p_decay(dat['participant'],
            p_rate_high, min_rate_hpd[2])
    ax.fill_between(xs, bottom_ys, top_ys, color='g', alpha=0.25)

    # plot the line for each condition
    for name, adf in df.groupby(['question_code', 'num_requested']):
        ys = gen_uniques_counts(adf, field)
        ax.plot(range(len(ys)), ys, '-', color='k')

    # plot the fit line
    fit_ys = model_integral_predict_p_decay(dat['participant'],
            p_rate_mean, min_rate_hpd[0])

    # plot the 1:1 line
    ys = [x for x in xs]
    ax.plot(xs, ys, '--', color='k', alpha=0.5)

    plt.show()

def n_tiles(l, n):
    """ Returns l split into n chunks
    """
    size = len(l) / n;
    for i in range(n):
        yield l[n * i : n * i+1]
        
def add_hpd_bar(ax, left, right, y, linewidth=2, edge_height = 200):
    heh = edge_height / 2
    ax.plot([left, right], [y, y], color='k', linewidth=linewidth)
    ax.plot([left, left], [y + heh, y - heh], color='k', linewidth=linewidth)
    ax.plot([right, right], [y + heh, y - heh], color='k', linewidth=linewidth)

def plot_rate_posteriors(p_rates):
    sorted_rates = sorted(p_rates, key=np.mean)
    #quartiles = n_tiles(sorted_rates, 4)
    #quartiles_flat = [reduce(lambda x, y: x.extend(y), quartile)
    #        for quartile in quartiles]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #for i, quartile in enumerate(quartiles_flat):
    #    ax.hist(quartiles_flat, bins=20, alpha=0.5, label="%ith quartile" % (i+1))

    # Plot highest and lowest rates
    ax.hist(sorted_rates[0], bins=20, alpha=0.75, label="fastest decay")
    ax.hist(sorted_rates[-1], bins=20, alpha=0.75, label="slowest decay")

    fast_hpd = mystats.mean_and_hpd(sorted_rates[0], 0.95)
    slow_hpd = mystats.mean_and_hpd(sorted_rates[-1], 0.95)
    add_hpd_bar(ax, fast_hpd[1], fast_hpd[2], 500)
    add_hpd_bar(ax, slow_hpd[1], slow_hpd[2], 500)

    ax.set_xlabel("rate parameter value")
    ax.set_ylabel("number of samples")
    ax.set_xlim((-0.015, 0))
    ax.legend()
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
    df = df[df['num_received'] >= 50]

    n_iter = 10000
    n_chains = 3

    dat = gen_model_data(df, rmdf, cdf, ifs)
    param_walks = modeling.compile_and_fit(model_string, dat, n_iter, n_chains)

    view_fit(df, 'idea', param_walks[0], dat)

