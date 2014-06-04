import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, math, os
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict
import rate_bernoulli_decay_model as bdm

def anal_string(expected_difference, small_rate_post,
        large_rate_post, small_min_post, large_min_post):
    anal_string="""The model was fit to the 2 groups, each of which included 3 conditions for number of responses requested. The fit model is shown in Figure~\\ref{fig:bernoulli_decay_btwn_nr}. Solid lines represent actual data, dotted lines the fit for the expected quantity of unique ideas generated, and the shaded regions the HDI for the expected number of responses. The non-overlapping shading demonstrates that when fewer ideas were requested, the expected quantity of unique ideas is significantly lower. At 500 ideas gathered, this manifests in a difference of %i expected unique ideas.

    The posterior values for $decay$ were %0.4f (HDI %0.4f-%0.4f) for the smaller requests condition and %0.4f (HDI %0.4f-%0.4f) for the larger. The posterior values for $min\\_rate$ were %0.2f (HDI %0.2f-%0.2f) for the smaller requests condition and %0.2f (HDI %0.2f-%0.2f) for the larger."""

    return anal_string % (expected_difference,
            small_rate_post[0], small_rate_post[1], small_rate_post[2],
            large_rate_post[0], large_rate_post[1], large_rate_post[2],
            small_min_post[0], small_min_post[1], small_min_post[2],
            large_min_post[0], large_min_post[1], large_min_post[2])

def t_chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield tuple(l[i:i+n])

def post_params_btwn_merged_nr(df, n_iter, n_chains, n_groups):
    nrs = sorted(list(set(df['num_requested'])))
    nrgrps = list(t_chunks(nrs, int(len(nrs) / n_groups)))
    dats = dict()
    res = dict()
    for nrgrp in nrgrps:
        print("Fitting group %s" % str(nrgrp))
        nrdf = df[df['num_requested'].isin(list(nrgrp))]
        dat = bdm.gen_model_data(nrdf, None, None, None)
        dats[nrgrp] = dat
        param_walks = modeling.compile_and_fit(bdm.model_string, dat, n_iter, n_chains)
        res[nrgrp] = param_walks

    return nrgrps, dats, res

def post_params_btwn_nr(df, n_iter, n_chains):
    nrs = list(set(df['num_requested']))
    dats = dict()
    res = dict()
    for nr in nrs:
        nrdf = df[df['num_requested'] == nr]
        dat = bdm.gen_model_data(nrdf, None, None, None)
        dats[nr] = dat
        param_walks = modeling.compile_and_fit(bdm.model_string, dat, n_iter, n_chains)
        res[nr] = param_walks

    return nrs, dats, res

def plot_fit_and_data_btwn_nr(ax, nrs, dats, fits):
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    assert(len(colors) >= len(nrs))
    for nr, color in zip(nrs, colors):
        dat = dats[nr]
        fit = fits[nr]

        # plot the real data first
        bdm.plot_data_cum(ax, dat, color = color)

        # then plot the predicted line and hpd
        rate = mystats.mean_and_hpd(fit[0]['rate'])
        min_rate = mystats.mean_and_hpd(fit[0]['min_rate'])
        bdm.plot_line_and_hpd(ax, rate, min_rate, max(dat['x']),
                '--', color=color, label=str(nr))

    ax.set_ylabel('number of unique ideas')
    ax.set_xlabel('number of instances collected')
    ax.set_ylim(bottom=0)
    ax.legend()


def filter_today(df):
    #df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    print('\033[1m' + os.path.basename(__file__) + '\033[0m')

    processed_data_folder = os.path.expanduser('~/enc_projects/crowbrain/processed_data')
    idf, ifs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(ifs, idf)

    n_iter = 3000
    n_chains = 3

    
    #for qc in set(df['question_code']):
        #qcdf = df[df['question_code'] == qc]
        #nrs, dats, res = post_params_btwn_nr(qcdf, n_iter, n_chains)
        #nrs, dats, res = post_params_btwn_merged_nr(qcdf, n_iter, n_chains, 2)

    nrs, dats, res = post_params_btwn_merged_nr(df, n_iter, n_chains, 2)
    decay1 = mystats.mean_and_hpd(res[nrs[0]][0]['rate'])
    decay2 = mystats.mean_and_hpd(res[nrs[1]][0]['rate'])
    min_rate1 = mystats.mean_and_hpd(res[nrs[0]][0]['min_rate'])
    min_rate2 = mystats.mean_and_hpd(res[nrs[1]][0]['min_rate'])

    smalld, larged = (decay1, decay2) if decay1[0] < decay2[0] else (decay2, decay1)
    smallmr, largemr = (min_rate1, min_rate2) if min_rate1[0] < min_rate2[0] else (min_rate2, min_rate1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
        #ax.set_title(qc)
    plot_fit_and_data_btwn_nr(ax, nrs, dats, res)
    fig.savefig('figures/bernoulli_decay_btwn_nr', dpi=600)

    with open('tex/bernoulli_decay_btwn_nr_anal.txt', 'w') as f:
        print(anal_string(bdm.expected_difference(decay1, min_rate1,
            decay2, min_rate2, 500), smalld,
            larged, smallmr, largemr),
            file=f)
 
