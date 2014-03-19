import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, os
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

def analysis_tex(n_chains, n_iterations, sim_passes, rate_posterior):
    analysis_latex = """The model fit was fit to the campaign quantities conditioned on question asked $\\times$ number of responses requested. The result is given in Figure~\\ref{fig:exponential_model_example}. The model converged in all of %i chains after %i iterations. The real data campaigns, which were used to fit the model, are shown as solid black lines. These values were generated for each campaign by ordering all instances first by HIT submission time and then by order in the brainstorming run, and then taking the cumulative \\emph{idea} count at each point.
The dotted line represents the fit model, the shaded region around which is the 95\\%% credible interval for the fit. Finally, a line representing linear growth is given along the diagonal. From this visualization, it is clear that the credible interval does not include the diagonal line, and the hypothesis stating that the growth rate is linear is rejected.
The mean for the rate parameter is %0.2f, (HDI %0.2f-%0.2f), confirming the visual evidence. Thus, the rate of idea generation over time appears to decrease exponentially as a function of the order in which the instance was received. 

The rejection of the linear hypothesis is also rejected under the error simulation process in %i out of 10 simulations.
"""

    return analysis_latex % (n_chains, n_iterations, sim_passes,
            rate_posterior[0], rate_posterior[1], rate_posterior[2])

model_string = """
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

def model_predict(x, y_scale, rate):
    return y_scale * (x ** rate)

def gen_uniques_counts(adf, field):
    adf = adf.sort(columns=['submit_datetime', 'answer_num'], ascending=[1, 1])
    uniques = set()
    
    counts = []
    for thing in adf[field]:
        uniques.add(thing)
        counts.append(len(uniques))
    return counts

def gen_model_data(df, rmdf, clusters_df, idea_forest):
    field = 'idea' # TODO?
    dat = defaultdict(list)

    for nr in set(df['num_requested']):
        nrdf = df[df['num_requested'] == nr]
        for qc in set(df['question_code']):
            qcdf = nrdf[nrdf['question_code'] == qc]
            uniques_counts = gen_uniques_counts(qcdf, field)
            dat['y'].extend(uniques_counts)
            dat['x'].extend(range(1, len(uniques_counts) + 1))
                            
    assert(len(dat['x']) == len(dat['y']))

    # Converting to tuple for immutability so the structure is hashable
    return {'x': tuple(dat['x']),
            'y': tuple(dat['y']),
            'N': len(dat['x'])}

def view_model_fit(df, field, la):
    rates = la['rate']
    
    y_scale = mystats.mean_and_hpd( la['y_scale'], 0.95)
    rate = mystats.mean_and_hpd(rates, 0.95)

    print("rate:", rate)
    print("y_scale:", y_scale)
    print("mean sigma:", np.mean(la['sigma']))

    plot_model(y_scale, (rate[1], rate[2]), rate[0], df, field)

def plot_model(y_scale_hpd, rate_hpd, rate_mean, df, field):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.set_xlabel("number of instances received")
    ax.set_ylabel("number of unique ideas")

    max_x = max(len(adf) for n, adf in df.groupby(['question_code', 'num_requested']))
    xs = range(max_x)

    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_x)

    # plot the hpd area
    bottom_ys = [model_predict(x, y_scale_hpd[1], rate_hpd[0]) for x in xs]
    top_ys = [model_predict(x, y_scale_hpd[2], rate_hpd[1]) for x in xs]
    ax.fill_between(xs, bottom_ys, top_ys, color='g', alpha=0.25)

    # plot the line for each condition
    for name, adf in df.groupby(['question_code', 'num_requested']):
        ys = gen_uniques_counts(adf, field)
        ax.plot(xs[:len(ys)], ys, '-', color='k')

    # plot the model line
    ys = [model_predict(x, y_scale_hpd[0], rate_mean) for x in xs]
    ax.plot(xs[:len(ys)], ys, '--', color='k')

    # plot the 1:1 line
    ys = [x for x in xs]
    ax.plot(xs, ys, '--', color='k', alpha=0.5)

    #plt.show()
    fig.savefig('figures/exponential_model_example', dpi=600)
    
def hyp_test_exclude_one(params):
    rates = params['rate']
    left, right = mystats.hpd(rates, 0.95)
    return 1 > right

def filter_today(df):
    #df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
import sys
if __name__ == '__main__':
    print(os.path.basename(__file__))
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)

    sys.exit(0)

    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    n_iter = 1500
    n_chains = 3

    dat = gen_model_data(df, None, None, None) 
    param_walks = modeling.compile_and_fit(model_string, dat, n_iter, n_chains)

    view_model_fit(idf, 'idea', param_walks[0])

    rate_post = mystats.mean_and_hpd(param_walks[0]['rate'])

    #sim_passes = modeling.simulate_error_hypothesis(10, model_string, n_iter, n_chains,
    #        gen_model_data, hyp_test_exclude_one, cfs, idf)
    #print("Exclude one hypothesis held in %i/10 cases" % sim_passes)
    sim_passes = 100 

    with open('tex/exp_model_anal.tex', 'w') as f:
        print(analysis_tex(n_chains, n_iter, sim_passes, rate_post), file=f)


