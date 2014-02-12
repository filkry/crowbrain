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

originality_log_model_string = """
data {
    int N;
    real y[N];
}

parameters {
    real<lower=0,upper=1> phi;
    real<lower=0.1> lambda;
}

transformed parameters {
    real<lower=0> alpha;
    real<lower=0> beta;
    alpha <- lambda * phi;
    beta <- lambda * (1 - phi);
}

model {
    phi ~ beta(1,1);
    lambda ~ pareto(0.1,1.5);
    for (i in 1:N) {
        log(y[i]) ~ beta(alpha, beta);
    }
}
"""
def gen_originality_model_dat(df, field):
    left_df = df[df['answer_num'] < 20]
    right_df = df[df['answer_num'] >= 20]
    
    left_dat = { 'N': len(left_df),
                 'y': left_df[field] }
    right_dat = { 'N': len(right_df),
                  'y': right_df[field] }


    return left_dat, right_dat

def view_originality_model_fit(left_fit, right_fit, df, field):
    for name, fit in [('LEFT', left_fit), ('RIGHT', right_fit)]:
        la = fit.extract(permuted=True)
        alpha = la['alpha']
        beta = la['beta']
        
        print(name)
        hl, hr = hpd(alpha, 0.95)
        print("alpha:", hl, hr)
        hl, hr = hpd(beta, 0.95)
        print("beta:", hl, hr)
 
def plot_originality_model(df, field):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("idea_oscore")
    ax.set_ylabel("number of instances")

    min_x = min(df[field])
    #xs = range(min_)
    #ax.set_xlim(0, max_x)
    #ax.set_ylim(0, max_x)

    # plot the hpd area
    #bottom_ys = [compute_rate_model(x, y_scale, rate_hpd[0]) for x in xs]
    #top_ys = [compute_rate_model(x, y_scale, rate_hpd[1]) for x in xs]
    #ax.fill_between(xs, bottom_ys, top_ys, color='g', alpha=0.25)

    # plot the histogram 
    #ranks = sp.stats.rankdata(df[field], method='average')
    #ax.hist(np.log10(ranks), log=True)

    jitter_range = 0.05 * (max(df[field]) - min(df[field]))
    jittered_field = [v + random.uniform(0, jitter_range) - 0.5 * jitter_range
            for v in df[field]]

    ax.hist(df[field])

    # plot the model line
    #ys = [compute_rate_model(x, y_scale, rate_mean) for x in xs]
    #ax.plot(xs[:len(ys)], ys, '--', color='k')

    plt.show()

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

# TODO: this could be done with passed parameters
def filter_today(df):
    df = df[df['question_code'] == 'iPod']
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, clusters_df, cluster_forests = format_data.mk_redundant(idf, cfs)

    left_dat, right_dat = gen_originality_model_dat(idf, 'idea')

    model_file = 'cache/split_originality_log_model_stan'
    if os.path.isfile(model_file): 
        model = pickle.load(open(model_file, 'rb'))
    else:
        model = pystan.StanModel(model_code=originality_log_model_string)
        pickle.dump(model, open(model_file, 'wb'))

    #fit = model.sampling(data=left_dat, iter=500, chains=1)

    #view_originality_model_fit(idf, 'idea', fit)
    plot_originality_model(df, 'idea_oscore')

