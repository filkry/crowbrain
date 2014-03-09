import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, math
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

model_string = """
data {
    int N;
    int y[N];
}

parameters {
    real<lower=1> mu;
    real<lower=1> sigma;
}

model {
    for (i in 1:N) {
        y[i] ~ lognormal(mu, sigma);
    }
}
"""

def gen_data(df, rmdf, cdf, ifs):
    sdf = df[~df['time_spent'].isnull()] # Some runs had glitchy time data; drop them
    runs = sdf.groupby(['worker_id', 'num_requested', 'accept_datetime'])
    changing_times = []
    within_times = []
    
    for name, run in runs:
        sdf = run.sort('answer_num', ascending=True)
        s_cats = sdf['subtree_root']
        s_times = sdf['time_spent']
        pairs = list(zip(s_cats[:-1], s_cats[1:], s_times[1:]))
        changing_times += [int(time) for cat1, cat2, time in pairs if cat1 != cat2]
        within_times += [int(time) for cat1, cat2, time in pairs if cat1 == cat2]

    changing_dat = { 'N': len(changing_times), 'y': changing_times }
    within_dat = { 'N': len(within_times), 'y': within_times }

    return [changing_dat, within_dat]
    
    dat = defaultdict(list)

def add_hpd_bar(ax, left, right, y, linewidth=2, edge_height = 50):
    heh = edge_height / 2
    ax.plot([left, right], [y, y], color='k', linewidth=linewidth)
    ax.plot([left, left], [y + heh, y - heh], color='k', linewidth=linewidth)
    ax.plot([right, right], [y + heh, y - heh], color='k', linewidth=linewidth)

def plot_fit(posterior, btwn_dat, wthn_dat):
    plt.rc('font', **{'sans-serif' : 'Arial',
                                   'family' : 'sans-serif'})

    fig = plt.figure()

    # plot time distributions
    btwn_times = [y for y in btwn_dat['y']]
    wthn_times = [y for y in wthn_dat['y']]

    ax = fig.add_subplot(2, 1, 1)
    ax.set_xlim(0, 200000)

    ax.set_title('time spent')
    btwn_bins = max(btwn_times) / 10000
    wthn_bins = max(wthn_times) / 10000
    ax.hist(btwn_times, label='between categories', bins = btwn_bins)
    ax.hist(wthn_times, label='within categories', bins= wthn_bins)
    ax.set_xlabel('time spent on response (ms)')
    ax.set_ylabel('number of responses')
    ax.legend()

    # Plot the posteriors with HDIs
    between_la, within_la = posterior
    ax = fig.add_subplot(2, 1, 2)
    ax.hist(between_la['mu'], label='between categories')
    ax.hist(within_la['mu'], label='within categories')
    ax.set_xlabel(u'μ')
    ax.set_ylabel('samples')
    ax.set_title(u'posterior μs')

    between_mu_hpd = mystats.mean_and_hpd(between_la['mu'], 0.95)
    within_mu_hpd = mystats.mean_and_hpd(within_la['mu'], 0.95)
    add_hpd_bar(ax, between_mu_hpd[1], between_mu_hpd[2], 100)
    add_hpd_bar(ax, within_mu_hpd[1], within_mu_hpd[2], 100)
    
    plt.show()

   
def hyp_test_between_greater(las, df, rmdf, cdf, ifs):
    between_la, within_la = las
    between_mu = mystats.mean_and_hpd(between_la['mu'], 0.95)
    within_mu = mystats.mean_and_hpd(within_la['mu'], 0.95)

    return min(between_mu) > max(within_mu)


def filter_today(df):
    df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df

def gen_posterior_fn(n_iter, n_chains):
    def posterior_fn(df, rmdf, cdf, ifs):
        btwn_dat, wthn_dat = gen_data(df, rmdf, cdf, ifs)
        btwn_walks = modeling.compile_and_fit(model_string, btwn_dat,
                n_iter, n_chains)
        wthn_walks = modeling.compile_and_fit(model_string, wthn_dat,
                n_iter, n_chains)

        return btwn_walks[0], wthn_walks[0]

    return posterior_fn
    
 
if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    btwn_dat, wthn_dat = gen_data(df, rmdf, cdf, cfs)

    posterior_fn = gen_posterior_fn(n_iter = 1500, n_chains = 3)
    posterior = posterior_fn(df, rmdf, cdf, cfs)
    print("Time between greater than time within:",
            hyp_test_between_greater(posterior,
                df, rmdf, cdf, cfs))

    plot_fit(posterior, btwn_dat, wthn_dat)
    #view_model_fit(idf, 'idea', param_walks[0])

    sim_passes = modeling.simulate_error_hypothesis_general(10,
        posterior_fn, hyp_test_between_greater, idf, cfs)
    print("Between greater hypothesis held in %i/10 cases" % sim_passes)

