import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, os
import matplotlib.pyplot as plt
import stats_fns as mystats
import random
from collections import defaultdict, OrderedDict
from scipy.stats import pareto, zipf

model_string = """
data {
    int <lower=0> N;
    real <lower=1> rrank[N];
}

parameters {
    real <lower=0> alpha;
}

model {
    alpha ~ exponential(0.5); // prior on alpha

    rrank ~ pareto(1, alpha);
}
"""

def gen_dat(freqs):
    rank = []
    for i, f in enumerate(freqs):
        rank.extend([i+1] * f)

    return {'rrank': rank,
            'N': len(rank)}

def gen_uniques_counts(series):
    uniques = set()
    
    counts = []
    for thing in series:
        uniques.add(thing)
        counts.append(len(uniques))
    return counts

def plot_real_cum(ax, df, field, **kwargs):
    adf = df.sort(columns=['submit_datetime', 'answer_num'], ascending=[1, 1])
    series = adf[field]
    counts = gen_uniques_counts(series)
    xs = range(len(counts))
    ax.plot(xs, counts, **kwargs)

def plot_sim_cum(ax, rnge, sample_fn, **kwargs):
    series = [sample_fn() for x in rnge]
    counts = gen_uniques_counts(series)
    ax.plot(rnge, counts, **kwargs)

def plot_dist(ax, max_x, dist_fn):
    xs = range(1, max_x)
    ys = [dist_fn(x) for x in xs]
    ax.plot(xs, ys, color='k')

def plot_freqs_against_dist(ax, freqs, dist_fn):
    total_freq = sum(freqs)
    norm_freqs = [f / total_freq for f in freqs]

    max_x = len(norm_freqs) + 1
    ax.bar(range(1, max_x), norm_freqs, color='r', alpha=0.25)
    plot_dist(ax, max_x, dist_fn)
    
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

def idea_freqs(df):
    icounts = defaultdict(int)
    ideas = df['idea']
    for i in ideas:
        icounts[i] += 1

    freqs = sorted(icounts.values(), reverse=True)
    return freqs

def plot_empirical_p_idea(ax, df):
    freqs = idea_freqs(df)

    total_freq = sum(freqs)
    norm_freqs = [f / total_freq for f in freqs]

    max_x = len(norm_freqs) + 1
    ax.bar(range(1, max_x), norm_freqs, color='r', alpha=0.25)

def plot_freqs_dists_btwn_nconds(df):
    fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    for i, nr in enumerate(set(df.num_requested)):
        sdf = df[df['num_requested'] == nr]
        ax = axes[i % 2, i % 3]
        ax.set_title(nr)

        freqs = idea_freqs(sdf)
        pdf, sample_fn = fit_pdf_and_sampling_fn(sdf)
        plot_freqs_against_dist(ax, freqs, pdf)

    fig.tight_layout()
    plt.show()

def filter_match_data_size(df):
    nr_conds = set(df['num_requested'])
    
    keep_indices = []
    
    for nr in nr_conds:
        sub_df = df[df['num_requested'] == nr]
        
        questions = sub_df.groupby('question_code')
        least_runs = min(len(qdf.groupby(['worker_id', 'accept_datetime', 'submit_datetime']))
                            for name, qdf in questions)
        
        for qc in set(sub_df['question_code']):
            ssub_df = sub_df[sub_df['question_code'] == qc]
            runs = list(ssub_df.groupby(['worker_id', 'accept_datetime', 'submit_datetime']))
            runs = random.sample(runs, least_runs)
            
            for ix, (name, run) in enumerate(runs):
                keep_indices.extend(run.index)
                
    new_df = df.select(lambda x: x in keep_indices)
    
    return new_df

def fit_pdf_and_sampling_fn(df):
    freqs = idea_freqs(df)
    dat = gen_dat(freqs)
    param_walks = modeling.compile_and_fit(model_string, dat, 3000, 3)
    post_alpha_param = mystats.mean_and_hpd(param_walks[0]['alpha'])

    return (lambda x: pareto.pdf(x, post_alpha_param[0]),
            lambda: int(pareto.rvs(post_alpha_param[0])))

def plot_sim_cum_cross_nconds(df):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i,nr in enumerate(set(df.num_requested)):
        sdf = df[df['num_requested'] == nr]
        pdf, sample_fn = fit_pdf_and_sampling_fn(sdf)

        plot_real_cum(ax, sdf, 'idea', color=colors[i])
        plot_sim_cum(ax, range(len(sdf)), sample_fn,
                color=colors[i], linestyle='--', label=str(nr))

    ax.legend(loc='upper left')

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
    #plot_log_log_zipf(df)

    # Plot a distribution fit againt the empirical frequencies
    #plot_freqs_against_dist(ax, freqs, pareto_pdf)

    # Plot cumulative number of ideas of both empirical data and a fit distribution
    #plot_real_cum(ax, df, 'idea', color='k')
    #for sim_color in ['r', 'b', 'g']:
    #    plot_sim_cum(ax, range(3500),
    #            lambda: int(pareto.rvs(post_alpha_param[0])),
    #            color=sim_color)
    #ax.set_xlabel('number of ideas sampled')
    #ax.set_ylabel('number of unique ideas')

    #plot_freqs_dists_btwn_nconds(df)

    plot_sim_cum_cross_nconds(df)
    plt.show()

