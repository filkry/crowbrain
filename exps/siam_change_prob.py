import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, os
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

def anal_string(theta_post, p_same):
    anal_string = """Fitting this model with an uniform beta prior (using an analytical solution rather than a sampling approach), the posterior mean for $\\theta$ is %0.2f (HDI %0.2f-%0.2f). 
For the brainstorming corpus across all questions, $p(x_i = x_{i+1}) = %0.2f$. This is well below the lower bound of the $\\theta$ HDI, allowing the rejection of the null hypothesis that category-following is no more likely than would be explained by random chance. This is consistent with the findings of Nijstad and Stroebe, and supports the concept that individuals work within categories of connected ideas and do not generate uniformly random ideas. This finding supports the idea that the SIAM cognitive model of alternative image activation and idea generation phases applies in microtask marketplaces as well.
    """

    return anal_string % (theta_post[0], theta_post[1], theta_post[2],
        p_same)

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
    #df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    print(os.path.basename(__file__))

    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    bidf, bcfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(bcfs, bidf)

    posterior_fn = lambda df, rmdf, cf, ifs: get_posterior(*gen_data(df, rmdf, cf, ifs))
    theta_post = posterior_fn(df, rmdf, cdf, cfs)
    view_model_fit(theta_post, cdf)
    
    sim_passes = modeling.simulate_error_hypothesis_general(10, posterior_fn,
        hyp_test_greater_chance, bidf, cfs)
    print("Greater than chance hypothesis held in %i/10 cases" % sim_passes)
    #sim_passes = 100

    roots = cdf[cdf['is_root'] == 1]
    p_consec = sum(pow(p, 2) for p in roots['subtree_probability'])
    p_consec /= len(set(roots['question_code']))

    with open('tex/siam_change_anal.tex', 'w') as f:
        print(anal_string(theta_post, p_consec, sim_passes), file=f)
