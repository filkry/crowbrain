import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, math
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

model_string = """
data {
    // shared data
    int N; // number of instances
    int x[N]; // ordinal position of the instance in its condition

    // bernoulli outcome variable
    int novel[N]; // whether there was a novel idea at this point
    
    // exponential outcome variable
    //real y[N]; // the number of ideas or categories receieved up to and including instance n
}

parameters {
    // bernoulli parameters
    real <upper=0> b_rate;
    real <lower=0, upper=1> b_min_rate;

    // exponential parameters
    //real<lower=0, upper=1> e_rate;
    //real<lower=0, upper=1> e_y_scale;
    //real<lower=0> e_sigma;

    // mixture parameter
    //real<lower=0, upper=1> lambda;
}

model {
    for (i in 1:N) {
        real b_theta;
        //real e_mu;
        
        real b_lp;
        //real e_lp;

        b_theta <- b_min_rate + exp(b_rate * x[i]) * (1 - b_min_rate);
        b_lp <- bernoulli_log(novel[i], b_theta);

        //e_mu  <- e_y_scale * pow(x[i], e_rate);
        //e_lp <- normal_log(y[i], e_mu, e_sigma);

        //increment_log_prob(log_sum_exp(log(lambda) + b_lp, log1m(lambda) + e_lp));
        increment_log_prob(b_lp);
    }
}
"""

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
            dat['y'].extend(uniques_counts)
                            
    return {
            'x': dat['x'],
            'y': dat['y'],
            'novel': dat['novel'],
            'N': len(dat['x'])}

def view_fit(df, field, la):
    lmbda = mystats.mean_and_hpd(la['lambda'], 0,95)
    print(lmbda)

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

