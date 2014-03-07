import pandas as pd
import numpy as np
import re, pystan, format_data, modeling
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

def gen_data(df, rmdf, cdf, ifs):
    runs = df.groupby(['worker_id', 'num_requested', 'accept_datetime'])
    
    total_transitions = 0
    within_transitions = 0
    
    for name, run in runs:
        sorted_ideas = run.sort('answer_num', ascending=True)['subtree_root']
        pairs = list(zip(sorted_ideas[:-1], sorted_ideas[1:]))
        total_transitions += len(pairs)
        within_transitions += sum(1 for a, b in pairs if a == b)
    
    return within_transitions, total_transitions

def get_posterior(within_transitions, total_transitions):
    return mystats.beta_bernoulli_posterior(within_transitions, total_transitions)

def view_model_fit(posterior, cluster_df):
    print(posterior)
    print("More than random chance:",
            hyp_test_greater_chance(posterior, None, None, cluster_df, None))

def hyp_test_greater_chance(posterior, df, rmdf, cdf, ifs):
    roots = cdf[cdf['is_root'] == 1]
    p_consec = sum(pow(p, 2) for p in roots['subtree_probability'])
    p_consec /= len(set(roots['question_code']))

    return posterior[1] > p_consec 

def filter_today(df):
    df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    bidf, bcfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(bcfs, bidf)

    posterior_fn = lambda df, rmdf, cf, ifs: get_posterior(*gen_data(df, rmdf, cf, ifs))
    view_model_fit(posterior_fn(df, rmdf, cdf, cfs), cdf)
    
    sim_passes = modeling.simulate_error_hypothesis_general(10, posterior_fn,
        hyp_test_greater_chance, bidf, cfs)
    print("Greater than chance hypothesis held in %i/10 cases" % sim_passes)
