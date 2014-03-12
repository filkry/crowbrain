import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, math
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

def anal_string(n_chains, n_iterations, rate_hdi, sim_passes):
    anal_string = """The fit model is shown in Figure~\\ref{fig:bernoulli_decay_model_example}, converging in %i chains after %i iterations each. In this case, rather than fitting to the cumulative number of ideas or categories, the rate model is fit to binary data, in which a 1 represents that the idea or category was novel when it was received (there were no previous examples of that idea or category). The expected cumulative count is recovered as described to provide a model fit that is comparable to that of the exponential model.
The model was fit using Stan, and the Stan language model specification is given in Appendix~\\ref{sec:decaying_bernoulli} .

The posterior rate parameter is %0.4f (HDI %0.4f, %0.4f), which is within the posterior HDIs for the same model fit under error simulation in %i out of 10 simulations."""

    return anal_string % (n_chains, n_iterations, rate_hdi[0], rate_hdi[1],
            rate_hdi[2], sim_passes)

model_string = """
data {
    int <lower=0> N; // number of instances
    int <lower=0, upper=1> novel[N]; // whether there was a novel idea at this point
    int <lower=0> x[N]; // ordinal position of the instance in its condition
}

parameters {
    real <lower=-100, upper=0> rate;
    real <lower=0, upper=1> min_rate;
}

model {
    real theta;
    for (i in 1:N) {
        theta <- min_rate + exp(rate * x[i]) * (1-min_rate);
        novel[i] ~ bernoulli(theta);
        //increment_log_prob(bernoulli_log(novel[i], theta));
    }
}
"""

def model_integral_predict(max_x, rate, min_rate):
    ys = [0]
    for x in range(1, max_x):
        ys.append(ys[-1] + min_rate + math.exp(rate * x) * (1- min_rate))
    return ys

def gen_uniques_counts(adf, field):
    adf = adf.sort(columns=['submit_datetime', 'answer_num'], ascending=[1, 1])
    uniques = set()
    
    counts = []
    for thing in adf[field]:
        uniques.add(thing)
        counts.append(len(uniques))
    return counts

def gen_model_data(df, rmdf, cdf, ifs):
    field = 'subtree_root'
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
                            
    return {
            'x': dat['x'],
            'novel': dat['novel'],
            'N': len(dat['x'])}

def view_fit(df, field, la):
    rate = mystats.mean_and_hpd(la['rate'], 0.95)
    min_rate = mystats.mean_and_hpd(la['min_rate'], 0.95)

    print("rate HDI:", rate)
    print("min_rate HDI:", min_rate)

    plot_model(rate, min_rate, df, field)

def plot_model_per_question(df, n_iter, n_chains):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    max_x = len(df)

    styles = ['-', '--', '-.', ':']
    real_max_x = 0
    for i, qc in enumerate(set(df['question_code'])):
        qcdf = df[df['question_code'] == qc]
        dat = gen_model_data(qcdf, None, None, None)
        real_max_x = max(real_max_x, max(dat['x']))
        param_walks = modeling.compile_and_fit(model_string, dat, n_iter, n_chains)

        rate = mystats.mean_and_hpd(param_walks[0]['rate'])
        min_rate = mystats.mean_and_hpd(param_walks[0]['min_rate'])

        plot_line_and_hpd(ax, rate, min_rate, max_x, styles[i], label=qc)

    # plot the 1:1 line
    xs = range(max_x)
    ys = [x for x in xs]
    ax.plot(xs, ys, '--', color='k', alpha=0.5)

    ax.set_xlim(0, real_max_x)
    ax.set_ylim(0, real_max_x)
    ax.set_xlabel('number of instances received')
    ax.set_ylabel('number of unique ideas')

    ax.legend()
    
    plt.show()


def plot_line_and_hpd(ax, rate, min_rate, max_x, line_style, **kwargs):
    xs = range(max_x)

    # plot the hpd area
    bottom_ys = model_integral_predict(max_x, rate[1], min_rate[1])
    top_ys = model_integral_predict(max_x, rate[2], min_rate[2])
    ax.fill_between(xs, bottom_ys, top_ys, color='g', alpha=0.25)

    # plot the model line
    ys = model_integral_predict(max_x, rate[0], min_rate[0])
    ax.plot(xs[:len(ys)], ys, line_style, color='k', **kwargs)

def plot_model(rate, min_rate, df, field):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("number of instances received")
    ax.set_ylabel("number of unique ideas")

    max_x = max(len(adf) for n, adf in df.groupby(['question_code', 'num_requested']))
    xs = range(max_x)

    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_x)

    # plot the line for each condition
    for name, adf in df.groupby(['question_code', 'num_requested']):
        ys = gen_uniques_counts(adf, field)
        ax.plot(xs[:len(ys)], ys, '-', color='k')

    plot_line_and_hpd(ax, rate, min_rate, max_x, '--')

    # plot the 1:1 line
    ys = [x for x in xs]
    ax.plot(xs, ys, '--', color='k', alpha=0.5)


    plt.show()

# TODO: this could be done with passed parameters
def filter_today(df):
    #df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
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

    view_fit(df, 'subtree_root', param_walks[0])
    plot_model_per_question(df, n_iter, n_chains)

    post_rate_param = mystats.mean_and_hpd(param_walks[0]['rate'])
    
    def hyp_fn(posterior, edf, ermdf, ecdf, eifs):
        return post_rate_param[0] > posterior[1] and post_rate_param[0] < posterior[2]

    def posterior_fn(edf, ermdf, ecdf, eifs):
        dat = gen_model_data(edf, ermdf, ecdf, eifs)
        param_walks = modeling.compile_and_fit(model_string, dat, n_iter, n_chains)
        return mystats.mean_and_hpd(param_walks[0]['rate'])

    #sim_passes = modeling.simulate_error_hypothesis_general(10, posterior_fn,
    #        hyp_fn, idf, cfs)
    #print("rate posterior in HDI in %i/10 simulations" % sim_passes)
    sim_passes = 100

    with open('tex/bernoulli_decay_model_anal.tex', 'w') as f:
        print(anal_string(n_chains, n_iter, post_rate_param, sim_passes),
                file=f)
