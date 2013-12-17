# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

## All of this is GPL (https://www.gnu.org/licenses/gpl.html)

# <codecell>

import pandas as pd
import scipy as sp
import numpy as np
import csv, os
import re
import scipy.stats as stats
import networkx as nx
#import cPickle as pickle
import itertools
from imp import reload
import random
import math
import matplotlib.pyplot as plt
import json

from collections import defaultdict, OrderedDict

# <markdowncell>

# # Introduction
# 
# The plan is to read in a bunch of fields from our various data sources and then do some stats on them.
# 
# Note that the first four fields together provide a unique way of identifying the answer (this could be better).
# 
# ## Transform data
# 
# For analysis, we may want to throw out some of it or manipulate things in some way. The following code block does this.

# <codecell>

pystan_test_mode = False # When this variable is on, we do single chains and few iterations, to go fast
pystan_fit_cache = dict()

error_df = None

# <codecell>

def all_pairings(s):
    return ((s1, s2) for i, s1 in enumerate(s[:-1])
                     for s2 in s[i+1:])

# <codecell>

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

def filter_repeats(df):
    new_df = df[df['is_repeat_worker'] == 0]
    return new_df

# A stupid function to just combine some filters. Oh, for haskell operators...
def filter_today(df):
    df = df[df['question_code'] == 'iPod']
    df = filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
    

# <codecell>

# Import data

import format_data
reload(format_data)

processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
df, rmdf, clusters_df, cluster_forests = format_data.mk_redundant(idf, cfs)

print(len(idf), len(rmdf))
# # Analysis functions

# <codecell>


nr_conds = list(set(df['num_requested']))
nr_conds = sorted(nr_conds)
print(nr_conds)

qc_conds = list(set(df['question_code']))
print(qc_conds)

matplotlib_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

qc_colors = {None: matplotlib_colors[0]}
for i, code in enumerate(qc_conds):
        qc_colors[code] = matplotlib_colors[i+1]
        print(qc_colors)

        nr_colors = {None: matplotlib_colors[0]}
        for i, nr in enumerate(nr_conds):
                nr_colors[nr] = matplotlib_colors[i+1]
                print(nr_colors)

                cond_colors = {'question_code': qc_colors,
                                        'num_requested': nr_colors }

import subprocess

_hdi_stats_cache = {}
# Use this one instead
def get_hdi(a, b, confidence_interval):
  if (a,b,confidence_interval) in _hdi_stats_cache:
    return _hdi_stats_cache[(a,b,confidence_interval)]
  #print "get_hdi:", a, b, confidence_interval
  if a == 1 or b == 1:
    if a == 1 and b == 1:
      lower_bound = (1.0-confidence_interval) / 2
      upper_bound = 1.0 - lower_bound
    else:
      #print "special qbeta:", a, b, confidence_interval
      results = subprocess.check_output(["R", "-q", "-e", 'qbeta(%f, 1, %d)' % (confidence_interval, max(a,b))])
      results = str(results)
      results = results.split('\\n')[1]
      upper_bound = float(results[results.index(' ')+1:])
      if a < b:
        lower_bound = 0
      else:
        lower_bound = 1.0 - upper_bound
        upper_bound = 1.0
  else:
    results = subprocess.check_output(["R", "-q", "-e", 
        'get_hdi = function(a, b, level=0.95) { density_diff = function(lower, a, b) { p_lower = pbeta(lower, a, b); p_upper = min(1.0, p_lower + level); upper = qbeta(p_upper, a, b); return(dbeta(lower, a, b) - dbeta(upper, a, b)); }; lower = uniroot(density_diff, c(0, qbeta(1.0-level, a, b)), a=a, b=b)$root; upper = qbeta(pbeta(lower, a, b) + level, a, b); return(c(lower, upper)); }; get_hdi(%d, %d, %f)' % (a, b, confidence_interval)])
    results = str(results)
    results = results.split('\\n')[1]
    results = results[results.index(' ')+1:]
    lower_bound, upper_bound = [float(x) for x in results.split(' ')]
  _hdi_stats_cache[(a,b,confidence_interval)] = (lower_bound, upper_bound)
  return lower_bound, upper_bound

def calculate_posterior(num_successes, total_num):
  a_prior = 1
  b_prior = 1
  a_post = a_prior + num_successes
  b_post = b_prior + (total_num - num_successes)
  posterior_mean = a_post / float(total_num + a_prior + b_prior)
  posterior_variance = posterior_mean * (1.0-posterior_mean) / (1.0 + a_prior + b_prior + total_num)
#  old_lower_bound, old_upper_bound = run_qbeta(a_post, b_post, 0.95)
  lower_bound, upper_bound = get_hdi(a_post, b_post, 0.95)
  return posterior_mean, lower_bound, upper_bound#, old_lower_bound, old_upper_bound


# HPD calculation from biopy, copied under GPL. Using this purely
# because it is an existing implementation of a credible interval,
# and I don't know why I would choose HDI, etc over this
def hpd(data, level) :
  """ The Highest Posterior Density (credible) interval of data at level level.

  :param data: sequence of real values
  :param level: (0 < level < 1)
  """ 
  
  d = list(data)
  d.sort()

  nData = len(data)
  nIn = int(round(level * nData))
  if nIn < 2 :
    raise RuntimeError("not enough data")
  
  i = 0
  r = d[i+nIn-1] - d[i]
  for k in range(len(d) - (nIn - 1)) :
    rk = d[k+nIn-1] - d[k]
    if rk < r :
      r = rk
      i = k

  assert 0 <= i <= i+nIn-1 < len(d)
  
  return (d[i], d[i+nIn-1])

# <markdowncell>

# # Modelling
# 
# Now that Pystan exists, we can do modelling right in the notebook! Dreamy!
# 
# Note: when running this notebook the first time, it's recommended to "run all" above here to get the data set up and make all the plots of "cheap" descriptive stats. Below, run blocks under headings as-needed, as each is a hypothesis test that relies on MCMC methods, which can be slow.

# <markdowncell>

# ## Model rate of idea/category generation
# 
# Take cumulative counts of these in each condition.

# <codecell>

import pystan

exp_model_string = """
data {
    int N; // number of instances
    real y[N]; // the number of ideas or categories receieved up to and including instance n
    int x[N]; // ordinal position of the instance in its condition
    int num_cond; // the number of conditions
    int cond_group[N]; // which condition
}

parameters {
    real<lower=0, upper=1> b_cond[num_cond];
    real<lower=0> y_scale;
    real<lower=0> sigma;
}

model {
    real mu[N];
    for (i in 1:N) {
        mu[i] <- y_scale * pow(x[i], b_cond[cond_group[i]]);
        y[i] ~ normal(mu[i], sigma);
    }
}
"""

def gen_exp_model_data(df, field):
    dat = defaultdict(list)

    for condition_num, nr in enumerate(nr_conds):
        nrdf = df[df['num_requested'] == nr]
        for qc in cluster_forests.keys():
            qcdf = nrdf[nrdf['question_code'] == qc]
            qcdf = qcdf.sort(columns=['submit_datetime', 'answer_num'], ascending=[1, 1])
            uniques = set()
            
            for order, thing in enumerate(qcdf[field]):
                uniques.add(thing)
                
                dat['x'].append(order + 1)
                dat['y'].append(len(uniques))
                dat['cond_group'].append(condition_num + 1)
                
    dat['N'] = len(dat['x'])
    dat['num_cond'] = len(nr_conds)
    
    return dat

def add_hpd_bar(ax, left, right, label, index):
    h = 200 + 100 * index
    ax.plot([left, right], [h, h], color='k')
    ax.plot([left, left], [h + 50, h - 50], color='k')
    ax.plot([right, right], [h + 50, h - 50], color='k')

def exp_model_density_regions(dat, fit):
    la = fit.extract(permuted=True)
    b_conds = la['b_cond']
    
    regions = []
    for i in range(1, dat['num_cond'] + 1):
        b_cond_i_samples = b_conds[:, i - 1]
        m_bci = mean(b_cond_i_samples)
        
        regions.append(hpd(b_cond_i_samples, 0.95))
        
    return regions
    
def view_exp_model_fit(dat, fit):
    la = fit.extract(permuted=True)
    b_conds = la['b_cond']
    y_scales = la['y_scale']
    
    m_y_scale = np.mean(y_scales)
    
    # plot learned functions
    figure(figsize=(10,10))
    ax = subplot(1,1,1)
    xlim(0, 1000)
    ylim(0, 1000)
    legend()
    for i in range(1, dat['num_cond'] + 1):
        b_cond_i_samples = b_conds[:, i - 1]
        m_bci = mean(b_cond_i_samples)
        
        X = np.linspace(0, max(dat['x']), 500, endpoint=True)
        Y = [m_y_scale * pow(x, m_bci) for x in X]
        ax.plot(X, Y, label='condition %d' % i)
        #hist(b_cond_i_samples
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    
    # plot parameters with HDI
    f = figure(figsize=(10,5))
    ax = f.add_subplot(1,1,1)
    density_regions = exp_model_density_regions(dat, fit)
    for i in range(1, dat['num_cond'] + 1):
        left, right = density_regions[i - 1]
        
        ax.hist(b_cond_i_samples, bins=20,
                label='condition %d' % i, alpha=0.2)
        add_hpd_bar(ax, left, right, 'condition %d hpd' % i, i)
        
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    f.show()

# <codecell>

def hyp_test(dats, model_string, testfunc, cache_key, df = None):
    n_chain = 2 if pystan_test_mode else 3 
    n_saved_steps = 1000 if pystan_test_mode else 10000
    
    fits = []
    for i, dat in enumerate(dats):
        #print(dat)
        ck = cache_key + str(i)
        if ck not in pystan_fit_cache:
            try:
                fit = pystan.stan(model_code=model_string,
                              data=dat,
                              iter=math.ceil(n_saved_steps/n_chain), chains=n_chain)
                # stop thrashing hard drive
                #pystan_fit_cache[ck] = fit
                fits.append(fit)
            except:
                print("Exception")
                dat2 = {'N': dat['N'],
                        'y': [float(y) for y in dat['y']]}
                with open('ipython_output/exception_dat.json', 'w') as f:
                    f.write(json.dumps(dat2))
                if not df is None:
                    with open('ipython_output/exception_ideas.json', 'w') as f:
                        f.write(json.dumps([int(i) for i in df['idea']]))
                    with open('ipython_output/exception_cats.json', 'w') as f:
                        f.write(json.dumps([int(i) for i in df['subtree_root']]))
                #raise
        else:
            fits.append(pystan_fit_cache[ck])
        
    success = testfunc(fits) if len(fits) == len(dats) else 'error' 
    return fits, success


def hyp_test_rate_exclude_one(df, field, cache_key):
    dat = gen_exp_model_data(df, field)
    
    def testfunc(fits):
        density_regions = exp_model_density_regions(dat, fits[0])
        return all([1 > right for left, right in density_regions])
    
    fits, success = hyp_test([dat], exp_model_string, testfunc, cache_key)
    return dat, fits, success

import pystan
from random import shuffle

exp_model_nocond_string = """
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

def gen_exp_model_nocond_data(df, field):
    dat = defaultdict(list)

    for condition_num, nr in enumerate(nr_conds):
        nrdf = df[df['num_requested'] == nr]
        for qc in cluster_forests.keys():
            qcdf = nrdf[nrdf['question_code'] == qc]
            qcdf = qcdf.sort(columns=['submit_datetime', 'answer_num'], ascending=[1, 1])
            uniques = set()
            
            for order, thing in enumerate(qcdf[field]):
                uniques.add(thing)
                
                dat['x'].append(order + 1)
                dat['y'].append(len(uniques))
                
    dat['N'] = len(dat['x'])
    
    return dat

def gen_exp_model_nocond_k_fold_data(field, k = 10):
    runs = df.groupby(['worker_id', 'num_requested'])
    
    run_dfs = [rdf for name, rdf in runs]
    
    shuffle(run_dfs)
    
    dats = []
    for i in range(k):
        train_df = pd.concat([rdf for j, rdf in enumerate(run_dfs) if j % k != i])
        test_df = pd.concat([rdf for j, rdf in enumerate(run_dfs) if j % k == i])
        
        train_dat = gen_exp_model_nocond_data(field, train_df)
        test_dat = gen_exp_model_nocond_data(field, test_df)
        #print(len(train_dat['x']), len(test_dat['x']))
    
        dats.append((train_dat, test_dat))
        
    return dats

def view_exp_model_nocond_fit(dat, fit):
    la = fit.extract(permuted=True)
    rates = la['rate']
    
    #print(rates)
    left, right = hpd(rates, 0.95)
    print(left, right)
    
def hyp_test_rate_nocond_exclude_one(df, field, cache_key):
    dat = gen_exp_model_nocond_data(df, field)
    
    def testfunc(fits):
        la = fits[0].extract(permuted=True)
        rates = la['rate']
        left, right = hpd(rates, 0.95)
        return 1 > right
    
    fits, success = hyp_test([dat], exp_model_nocond_string, testfunc, cache_key)
    return dat, fits, success

# <markdowncell>

# ## Early Common Analysis
# 
# One hypothesis in the paper is that there are a set of common, general ideas that make up the first several responses of each session. To test this, we take a set of the 5% of most common ideas. Then we test the probability that an idea in the first 5 instances of a run is in the common set, and the probability an idea in any later section of the run is in the common set.
# 
# We only train these parameters on runs that have 10 or more responses, so no run can contribute to either parameter without contributing to the other.

# <codecell>

def hyp_test_early_common(df, clusters_df):
    temp_df = clusters_df.sort('num_instances_under', ascending=False)
    num_top_5_pc = math.ceil(len(temp_df) * 0.05)
    top_5_pc = temp_df.iloc[:num_top_5_pc]
    top_5_pc_idea = set(top_5_pc['idea'])
    
    ten_or_more = df[df['num_received'] >= 10]
    first_five = ten_or_more[ten_or_more['answer_num'] < 5]
    rest = ten_or_more[ten_or_more['answer_num'] >= 5]
    
    first_five_successes = len([idea for idea in first_five['idea'] if idea in top_5_pc_idea])
    rest_successes = len([idea for idea in rest['idea'] if idea in top_5_pc_idea])
    
    first_five_post = calculate_posterior(first_five_successes, len(first_five))
    rest_post = calculate_posterior(rest_successes, len(rest))
    
    return first_five_post, rest_post, min(first_five_post) > max(rest_post)

# <markdowncell>

# ## Originality before/after 20

# <codecell>

beta_model = """
data {
    int N;
    real y[N];
}

parameters {
    real<lower=0> alpha;
    real<lower=0> beta;
}

model {
    for (i in 1:N) {
        y[i] ~ beta(alpha, beta);
    }
}
"""


def split_20_beta_model_dat(df, field):
    left_df = df[df['answer_num'] < 20]
    right_df = df[df['answer_num'] >= 20]
    
    left_dat = { 'N': len(left_df),
                 'y': left_df[field] }
    right_dat = { 'N': len(right_df),
                  'y': right_df[field] }


    return left_dat, right_dat

def view_split_20_beta_model_fit(left_fit, right_fit):
    for name, fit in [('LEFT', left_fit), ('RIGHT', right_fit)]:
        la = fit.extract(permuted=True)
        alpha = la['alpha']
        beta = la['beta']
        
        print(name)
        hl, hr = hpd(alpha, 0.95)
        print("alpha:", hl, hr)
        hl, hr = hpd(beta, 0.95)
        print("beta:", hl, hr)
        
def hyp_test_split_20_beta_model(df, field, cache_key):
    dats = list(split_20_beta_model_dat(df, field))
    
    def testfunc(fits):
        prev_alpha = None
        prev_beta = None
        for name, fit in [('LEFT', fits[0]), ('RIGHT', fits[1])]:
            la = fit.extract(permuted=True)
            alpha = la['alpha']
            beta = la['beta']
            
            if prev_alpha is None:
                prev_alpha = hpd(alpha, 0.95)
                prev_beta = hpd(beta, 0.95)
            else:
                alpha = hpd(alpha, 0.95)
                beta = hpd(beta, 0.95)
                if max(prev_alpha) < min(alpha) or max(alpha) < min(prev_alpha):
                    return True
                elif max(prev_beta) < min(beta) or max(beta) < min(prev_beta):
                    return True
        return False
    
    fits, success = hyp_test(dats, beta_model, testfunc, cache_key, df)
    
    return dats, fits, success

# <markdowncell>

# ### idea o-score
# <markdowncell>

# ## Probability of remaining in same category
# 
# From the SIAM model, we hypothesize that an idea is more likely to be followed by another idea in the same category than by random change.

# <codecell>

def hyp_test_prob_in_category(df):
    runs = df.groupby(['worker_id', 'num_requested', 'accept_datetime'])
    
    total_transitions = 0
    within_transitions = 0
    
    for name, run in runs:
        sorted_ideas = run.sort('answer_num', ascending=True)['subtree_root']
        pairs = list(zip(sorted_ideas[:-1], sorted_ideas[1:]))
        total_transitions += len(pairs)
        within_transitions += sum(1 for a, b in pairs if a == b)
    
    post = calculate_posterior(within_transitions, total_transitions)
    success = min(post) > max(df['idea_probability'])
    
    return post, success

# <markdowncell>

# ## Time spent changing categories
# 
# Also from SIAM, we expect time spent changing categories to be greater than time spent within categories. Both seem to fit a roughly log-normal distribution.

# <codecell>

lognormal_time_model = """
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

def lognormal_time_model_dat(df):
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
    

def view_lognormal_time_model_fit(change_fit, within_fit):
    for name, fit in [('CHANGE', change_fit), ('WITHIN', within_fit)]:
        la = fit.extract(permuted=True)
        mu = la['mu']
        sigma = la['sigma']
        
        print(name)
        hl, hr = hpd(mu, 0.95)
        print("mu:", hl, hr)
        hl, hr = hpd(sigma, 0.95)
        print("sigma:", hl, hr)
        
def hyp_test_lognormal_time_model(df, cache_key):
    dats = lognormal_time_model_dat(df)
    
    def testfunc(fits):
        prev_mu = None
        prev_sigma = None
        for name, fit in [('changing', fits[0]), ('within', fits[1])]:
            la = fit.extract(permuted=True)
            mu = la['mu']
            
            if prev_mu is None:
                prev_mu = hpd(mu, 0.95)
            else:
                mu = hpd(mu, 0.95)
                if max(mu) < min(prev_mu):
                    return True
        return False
    
    fits, success = hyp_test(dats, lognormal_time_model, testfunc, cache_key)
    
    return dats, fits, success

def bin_sequence(seq, bin_val, num_bins):
    bin_vals = [bin_val(s) for s in seq]
    max_val = max(bin_vals)
    min_val = min(bin_vals)
    
    bins = [[] for i in range(num_bins)]
    
    for s in seq:
        val = bin_val(s)
        # min fixes inclusive upper bound
        b = min(num_bins - 1, math.floor(num_bins * float(val - min_val) / float(max_val - min_val)))
        bins[b].append(s)
        
    return bins

# <markdowncell>

# ### Read and process results
# 
# I hand-input the results from the printed surveys. Here we're going to input them and get a sense of what the surface of the error space looks like (as a fn of bin).

# <codecell>

import csv
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D

def get_results():
    survey_results = "%s/validity_survey/data.csv" % processed_data_folder
    with open(survey_results, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        all_results = [row for row in reader if row[0] != 'judge' and row[5] != '?']
        
        return all_results
    
def cull_repeat_pairings(results):
    seen = set()
    for res in results:
        judge, n1id, n2id, bin1, bin2, rel, _, _, _, _ = res
        if (n1id, n2id) not in seen:
            seen.add((n1id, n2id))
            yield int(judge), int(n1id), int(n2id), int(bin1), int(bin2), int(rel)

def ancestors(f, nid):
    preds = f.predecessors(nid)
    return [ancestors(f, p) for p in preds] + preds
            
def get_node_relationship(n1id, n2id):
    f = cluster_forests['iPod']

    n1 = f.node[n1id]
    n2 = f.node[n2id]
    
    if n1['subtree_root'] != n2['subtree_root']:
        return 3
    elif n1id in ancestors(f, n2id):
        return 1
    elif n2id in ancestors(f, n1id):
        return 2
    else:
        return 4

def test_violation(guess, judge_rel, violation_type = "all"):
    fail = guess != int(judge_rel)
    if violation_type == 'all':
        return fail
    elif violation_type == 'parent_child':
        return fail and (judge_rel == 1 or judge_rel == 2 or guess == 1 or guess == 2)
    elif violation_type == 'artificial_parent':
        return fail and (judge_rel == 4 or guess == 4)
    elif violation_type == 'single_node_per_idea':
        return fail and (judge_rel == 5)
    
def gen_grid_bernoullis(results, violation_type = "all"):
    grid = defaultdict(list)
    bern_grid = dict()
    
    for judge, n1id, n2id, bin1, bin2, rel in results:
        guess = get_node_relationship(n1id, n2id)
        grid[(bin1, bin2)].append(int(test_violation(guess, rel, violation_type)))
    
    for k in grid.keys():
        trials = grid[k]
        p = calculate_posterior(sum(trials), len(trials))
        bern_grid[k] = (p[0], p[1], p[2], sum(trials) / len(trials))
        
    return bern_grid

def bern_grid_sym(bg, key1, key2):
    if (key1, key2) in bg:
        return bg[(key1, key2)]
    else:
        return bg[(key2, key1)]

def plot_bern_grid(bern_grid, title):
    # mean is 0, empirical mean is 3
    X, Y, Z = zip(*[(k[0], k[1], bern_grid[k][3]) for k in bern_grid])
    
    X2d, Y2d = np.meshgrid(np.arange(0, 5, 1), np.arange(0, 5, 1))
    Z2d = np.array([bern_grid_sym(bern_grid, x, y)[3] for x, y in zip(np.ravel(X2d), np.ravel(Y2d))])
    Z2d = Z2d.reshape(X2d.shape)
    
    plt.ion()
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.invert_xaxis()
    ax.plot_surface(X2d, Y2d, Z2d, shade=True, rstride=1, cstride=1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("bin 1")
    ax.set_ylabel("bin 2")
    ax.set_zlabel("error proportion")
    ax.set_title(title)
    
    # headmap
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    A = np.zeros((5,5))
    for x, y, z in zip(X, Y, Z):
        A[x, y] = z
        A[y, x] = z
    
    ax.imshow(A, interpolation='bilinear')
    ax.invert_yaxis()
    ax.set_xlabel("bin 1")
    ax.set_ylabel("bin 2")
    ax.set_title(title)
    
all_results = get_results()
culled = list(cull_repeat_pairings(all_results))
bern_grid = gen_grid_bernoullis(culled)
#plot_bern_grid(bern_grid, 'all')

pc_bern_grid = gen_grid_bernoullis(culled, 'parent_child')
#plot_bern_grid(pc_bern_grid, 'parent_child')

ap_bern_grid = gen_grid_bernoullis(culled, 'artificial_parent')
#plot_bern_grid(ap_bern_grid, 'artificial_parent')

sn_bern_grid = gen_grid_bernoullis(culled, 'single_node_per_idea')
#plot_bern_grid(sn_bern_grid, 'single_node_per_idea')

# <markdowncell>

# ## Simulation of error impact
# 
# Based on error rates identified above, permute the tree structure and see how that impacts our hypotheses.
# 
# **Note**: iPod only right now

# <codecell>

import random, sys

def get_node_bin(bins, node):
    for i, binn in enumerate(bins):
        if node in binn:
            return i
    return "ERROR" + str(node)

def chain_between(forest, ancestor, descendent):
    cur = descendent
    chain = [cur]
    while(cur != ancestor and forest.predecessors(cur)[0] != ancestor):
        cur = forest.predecessors(cur)[0]
        chain.append(cur)
    return chain

def get_root(forest, node):
    cur = node
    while(len(forest.predecessors(cur)) > 0):
        cur = forest.predecessors(cur)[0]
    return cur

def introduce_parent_child_error(new_forest, n1, n2):
    if n1 in nx.descendants(new_forest, n2) or n2 in nx.descendants(new_forest, n1):
        anc, desc = (n1, n2) if n2 in nx.descendants(new_forest, n1) else (n2, n1)
        chain = chain_between(new_forest, anc, desc)
        bad_link = random.sample(chain, 1)[0]
        new_forest.remove_edge(new_forest.predecessors(bad_link)[0], bad_link)
    else:
        pa, chi = (n1, n2) if random.random() > 0.5 else (n2, n1)
        if len(new_forest.predecessors(chi)) > 0:
            new_forest.remove_edge(new_forest.predecessors(chi)[0], chi)
        new_forest.add_edge(pa, chi)

def introduce_artificial_error(new_forest, n1, n2):
    n1root = get_root(new_forest, n1)
    if n2 in nx.descendants(new_forest, n1root):
        err_node = n1 if n2 == n1root else (n2 if n1 == n1root or random.random() > 0.5 else n1)
        chain = chain_between(new_forest, n1root, err_node)
        bad_link = random.sample(chain, 1)[0]
        new_forest.remove_edge(new_forest.predecessors(bad_link)[0], bad_link)
    else: # if no artificial node, introduce one
        n2root = get_root(new_forest, n2)
        new_node = max(new_forest.nodes()) + 1
        new_forest.add_node(new_node)
        new_forest.node[new_node]['label'] = 'artificial node (simulated error)'
        new_forest.add_edge(new_node, n1root)
        new_forest.add_edge(new_node, n2root)
        
def introduce_single_node_error(new_forest, instance_df, n1, n2):
    keep, lose = (n1, n2) if random.random() > 0.5 else (n2, n1)
    keep, lose = (lose, keep) if lose in nx.ancestors(new_forest, keep) else (keep, lose)
    
    for n in new_forest.successors(lose):
        new_forest.add_edge(keep, n)
    new_forest.remove_node(lose)
    
    new_idea = instance_df['idea']

    for idx in new_idea.index:
        if new_idea[idx] == lose:
            new_idea[idx] = keep

    instance_df['idea'] = new_idea
    
    return lose

def one_normalize(alist):
    total = sum(alist)
    return [float(v) / total for v in alist]


# This is really ugly, I should use more generalizable distribution/
# marginalization code, but fast > good sigh
def bern_grid_marginalize_bin2(bern_grid, bins):
    total_len_bins = sum(len(b) for b in bins)
    p_bins = [float(len(b)) / total_len_bins for b in bins]

    new_bern = dict()

    bin1s = sorted(list(set(b1 for b1, b2 in bern_grid.keys())))
    bin2s = sorted(list(set(b2 for b1, b2 in bern_grid.keys())))

    for bin1 in bin1s:
        sum_parts = []
        for bin2 in bin2s:
            bern_mean = bern_grid_sym(bern_grid, bin1, bin2)[0]
            sum_parts.append(bern_mean * p_bins[bin2])
        new_bern[bin1] = (sum(sum_parts), one_normalize(sum_parts))

    return new_bern

def get_error_node2(err_bern, n1bin, bins, exclude_nodes):
    x = random.random()
    for binno, rate in enumerate(err_bern[n1bin][1]):
        if x <= rate:
            sample_set = set(bins[binno]).difference(set(exclude_nodes))
            assert(len(sample_set) > 0)
            n2 = random.sample(sample_set, 1)[0]
            return n2
        x -= rate

    assert(False)

def simulate_error_node(instance_df, cluster_forest, bins, parent_child_bern_grid,
                         artificial_bern_grid, single_node_bern_grid):
    new_forest = cluster_forest.copy()
    new_idf = instance_df.copy()
    
    real_nodes = [n for n in cluster_forest.nodes() if len(instance_df[instance_df['idea'] == n]) > 0]
    print("Number of nodes", len(real_nodes))
    
    err_berns = [bern_grid_marginalize_bin2(parent_child_bern_grid, bins),
            bern_grid_marginalize_bin2(artificial_bern_grid, bins),
            bern_grid_marginalize_bin2(single_node_bern_grid, bins)]

    lost_nodes = []
    
    for j, n1 in enumerate(real_nodes):
        print("Simulating error on forest %i/%i" % (j, len(real_nodes)), end='\r')
        
        if n1 in lost_nodes:
            continue
        
        n1bin = get_node_bin(bins, n1)
        
        err_rates = [eb[n1bin][0] for eb in err_berns]
        x = random.random()

        actual_err = None
        n2 = None
        for i, (err, rate) in enumerate(zip(err_berns, err_rates)):
            if x <= rate:
                actual_err = i
                n2 = get_error_node2(err, n1bin, bins, lost_nodes)
                break
            x -= rate
            
        if actual_err == 0:
            introduce_parent_child_error(new_forest, n1, n2)
            last = "Parent"
        elif actual_err == 1:
            introduce_artificial_error(new_forest, n1, n2)
            last = "artificial"
        elif actual_err == 2:
            lost_nodes.append(introduce_single_node_error(new_forest, new_idf, n1, n2))
            last = 'single'

    print("Total nodes merged:", len(lost_nodes))

    return new_forest, new_idf


def simulate_error_pair(instance_df, cluster_forest, bins, parent_child_bern_grid,
                         artificial_bern_grid, single_node_bern_grid):
    
    new_forest = cluster_forest.copy()
    new_idf = instance_df.copy()
    
    real_nodes = [n for n in cluster_forest.nodes() if len(instance_df[instance_df['idea'] == n]) > 0]
    print("Number of nodes", len(real_nodes))
    node_pairings = list(all_pairings(real_nodes))
    
    lost_nodes = []
    
    last = None
    
    for j, (n1, n2) in enumerate(node_pairings):
        print("Simulating error on forest %i/%i" % (j, len(node_pairings)), end='\r')
        
        if n1 in lost_nodes or n2 in lost_nodes:
            continue
        
        n1bin = get_node_bin(bins, n1)
        n2bin = get_node_bin(bins, n2)
        
        err_order = [parent_child_bern_grid, artificial_bern_grid, single_node_bern_grid]
        err_rates = [bern_grid_sym(err, n1bin, n2bin)[0] for err in err_order]

        x = random.random()

        actual_err = None
        for err, rate in zip(err_order, err_rates):
            if x <= rate:
                actual_err = err
                break
            x -= rate
            
        if actual_err == parent_child_bern_grid:
            introduce_parent_child_error(new_forest, n1, n2)
            last = "Parent"
        elif actual_err == artificial_bern_grid:
            introduce_artificial_error(new_forest, n1, n2)
            last = "artificial"
        elif actual_err == single_node_bern_grid:
            lost_nodes.append(introduce_single_node_error(new_forest, new_idf, n1, n2))
            last = 'single'

    print("Total nodes merged:", len(lost_nodes))

    return new_forest, new_idf

qc_cdf = clusters_df[(clusters_df['question_code'] == 'iPod') & (clusters_df['num_instances'] > 0)]
nodes = list(zip(list(qc_cdf['idea']), list(qc_cdf['num_instances'])))
bins = bin_sequence(nodes, lambda x: x[1], 5)
bins = [[n[0] for n in binn] for binn in bins]

# <codecell>


def test_all_chi14_hypotheses(df, cdf, suffix):
    successes = []

    dat, fits, success = hyp_test_rate_nocond_exclude_one(df, 'idea', 'idea rate nocond simulate error ' + suffix)
    successes.append(success)

    dat, fits, success = hyp_test_rate_nocond_exclude_one(df, 'subtree_root', 'cat rate nocond simulate error ' + suffix)
    successes.append(success)
    
    urk = hyp_test_early_common(df, cdf)
    successes.append(urk[-1])

    dat, fits, success = hyp_test_split_20_beta_model(df, 'idea_oscore', 'idea oscore split 20simulate error ' + suffix)
    successes.append(success)

    dat, fits, success = hyp_test_split_20_beta_model(df, 'subtree_oscore', 'cat oscore split 20simulate error ' + suffix)
    successes.append(success)
    
    post, success = hyp_test_prob_in_category(df)
    successes.append(success)

    dat, fits, success = hyp_test_lognormal_time_model(df, 'test_lognormal_time_model simulate error ' + suffix)
    successes.append(success)
    
    return successes
    

# <codecell>

#def do():
num_permuted_trees = 10

with open('ipython_output/simulate_error.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['simulation', 'idea_rate', 'cat_rate', 'early_common',
        'idea_split_20', 'cat_split_20', 'within_prob', 'time_changing'])
    for i in range(num_permuted_trees):
        err_forest, err_idf = simulate_error_node(idf[idf['question_code'] == 'iPod'], cfs['iPod'],
                bins, pc_bern_grid, ap_bern_grid, sn_bern_grid)
        assert(nx.is_directed_acyclic_graph(err_forest))
        
        err_df, err_rmdf, err_clusters_df, err_cluster_forests = format_data.mk_redundant(err_idf, {'iPod': err_forest})

        successes = test_all_chi14_hypotheses(err_df, err_clusters_df, str(i))
        writer.writerow([i] + successes)

