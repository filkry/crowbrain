import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, math, os
import matplotlib.pyplot as plt
import stats_fns as mystats
from functools import reduce
from collections import defaultdict, OrderedDict

def anal_string(min_received, n_chains, n_iter, predictions):
    anal_string="""The model was fit only to participants who generated %i or greater ideas, as fewer responses were insufficient for the model to achieve convergence. The fit model is shown in Figure~\\ref{fig:bernoulli_decay_participant_model_example}, for a subset of campaigns conditioned on question asked $\\times$$ number of responses requested. This was done because a single fit for all campaigns has no meaning since each campaign has different participants. The model converged in %i chains of %i iterations.
The model was fit using Stan, and the Stan language model specification is given in Appendix~\\ref{sec:decaying_bernoulli_part}.

In this model, the fit line is non-continuous (but still contiguous) - different participants ``bump'' or ``flatten'' the rate of idea generation as they contribute. While this model is less general - it is not expected to always receive participants with a similar distribution of decay parameters - by examining the posterior distributions of rate parameters, judgments can be made as to the distribution of ``quality'' brainstormers. Figure~\\ref{fig:bernoulli_decay_participant_posteriors} plots the posterior distributions over the decay parameters for the most and least productive participants in each question condition.

As can be seen by the non-overlapping HPDs, the most productive participant has their rate of idea generation decay significantly less than the least productive participant. This means that variations in individual ability account for a significant portion of the variation in the number of ideas produced. In this case, the most productive participant would produce an expected %i more novel ideas in a solo run of 100 instances than the least productive participant.
This gap widens further to %i additional novel ideas out of 100 when the same participants are contributing to a cumulative brainstorming pool that has already received 500 instances.
These large differences in quantity of ideas generated provide motivation for the future exploration of interventions in the crowd brainstorming space."""

    return anal_string % (min_received, n_chains, n_iter,
            predictions[0] - predictions[1], predictions[2] - predictions[3])

model_string = """
data {
    int <lower=1> M; // number of participants
    int <lower=M> N; // number of instances
    int <lower=0, upper=1> novel[N]; // whether there was a novel idea at this point
    int <lower=0> x[N]; // ordinal position of the instance in its condition
    int <lower=1, upper=M> participant[N]; // which participant provided response
}

parameters {
    real <lower=-10, upper=0> rate[M];
    real <lower=0, upper=1> min_rate;

    real <lower=-10, upper=0> hyper_rate_mu;
    real <lower=0, upper=5> hyper_rate_sigma;
}

model {
    real theta;

    for (i in 1:M) {
        rate[i] ~ normal(hyper_rate_mu, hyper_rate_sigma);
    }

    for (i in 1:N) {
        theta <- min_rate + exp(rate[participant[i]] * x[i]) * (1-min_rate);
        increment_log_prob(bernoulli_log(novel[i], theta));
    }
}
"""

def model_integral_predict_p_decay(which_worker, worker_decay, min_rate):
    ys = []
    for x, worker in enumerate(which_worker):
        if(len(ys) == 0):
            ys.append((1-min_rate) * math.exp(worker_decay[worker - 1] * x) + min_rate)
        else:
            ys.append(ys[-1] + (1-min_rate) * math.exp(worker_decay[worker - 1] * x) + min_rate)
    return ys

def model_integral_predict_fixed_decay(num, decay_rate, min_rate, start_x = 1):
    ys = []
    for x in range(start_x, num + start_x):
        if(len(ys) == 0):
            ys.append((1-min_rate) * math.exp(decay_rate * x) + min_rate)
        else:
            ys.append(ys[-1] + (1-min_rate) * math.exp(decay_rate * x) + min_rate)
    return ys

def gen_uniques_counts(adf, field):
    adf = adf.sort(columns=['submit_datetime', 'answer_num'], ascending=[1, 1])
    uniques = set()
    
    counts = []
    for thing in adf[field]:
        uniques.add(thing)
        counts.append(len(uniques))
    return counts

def get_worker_ints(df, next_worker_int, worker_ints):
    """
    Generate a list of ints to represent workers from their worker IDs
    There are many smarter ways to do this, but this was the fastest for
    me to write
    """
    ret = []
    for worker_id in df['worker_id']:
        if not worker_id in worker_ints:
            worker_ints[worker_id] = next_worker_int
            next_worker_int += 1
        ret.append(worker_ints[worker_id])
    return next_worker_int, worker_ints, ret

def gen_model_data(df, rmdf, cdf, ifs):
    field = 'idea'
    dat = defaultdict(list)
    qcs = []

    cur_worker_int = 1
    worker_ints = dict()

    for nr in sorted(list(set(df['num_requested']))):
        nrdf = df[df['num_requested'] == nr]
        for qc in sorted(list(set(df['question_code']))):
            qcdf = nrdf[nrdf['question_code'] == qc]
            uniques_counts = gen_uniques_counts(qcdf, field)

            temp = [0] + uniques_counts
            uniques_diffs = np.diff(temp)

            dat['novel'].extend(uniques_diffs)
            dat['x'].extend(range(len(uniques_diffs)))
            qcs.extend([qc for i in range(len(uniques_diffs))])

            cur_worker_int, worker_ints, df_wints = get_worker_ints(qcdf, cur_worker_int, worker_ints)
            dat['participant'].extend(df_wints)

    return {
            'x': dat['x'],
            'novel': dat['novel'],
            'participant': dat['participant'],
            'M': cur_worker_int - 1,
            'N': len(dat['x'])}, qcs

def view_fit(df, field, la, dat, qcs):
    rate_array = la['rate']
    p_rates = [rate_array[:, i] for i in range(rate_array.shape[1])]

    sorted_rates = sorted(p_rates, key=np.mean)

    min_rate = np.mean(la['min_rate'])

    prediction_low_decay = model_integral_predict_fixed_decay(100, np.mean(sorted_rates[-1]), min_rate)
    prediction_high_decay = model_integral_predict_fixed_decay(100, np.mean(sorted_rates[0]), min_rate)
    print("Number of ideas generated with low decay rate:", prediction_low_decay[-1])
    print("Number of ideas generated with high decay rate:", prediction_high_decay[-1])

    prediction_low_decay_500 = model_integral_predict_fixed_decay(100, np.mean(sorted_rates[-1]), min_rate, 500)
    prediction_high_decay_500 = model_integral_predict_fixed_decay(100, np.mean(sorted_rates[0]), min_rate, 500)
    print("Number of ideas generated with low decay rate (from 500):", prediction_low_decay[-1])
    print("Number of ideas generated with high decay rate (from 500):", prediction_high_decay[-1])

    plot_rate_posteriors(p_rates, qcs, dat['participant'])

    plot_cumulative_model(df, dat, p_rates, mystats.mean_and_hpd(la['min_rate'], 0.95), field)

    return (prediction_low_decay[-1], prediction_high_decay[-1],
            prediction_low_decay_500[-1], prediction_high_decay_500[-1])

def cumulative_list(l):
    total = 0
    for x in l:
        total += x
        yield total

def index_starting_at(l, target, start):
    if target not in l[start:]:
        return None
    return l[start:].index(target) + start

def plot_cumulative_model(df, dat, p_rates, min_rate_hpd, field):
    fig = plt.figure(figsize=(8,10))

    condition_index = 0 
    # for each of the first 6 combinations
    #print(dat['x'][1000:1100])
    for i in range(1, 7):
        ax = fig.add_subplot(3, 2, i)

        end_of_condition = index_starting_at(dat['x'], 0, condition_index + 1)
        if end_of_condition is None:
            end_of_condition = len(dat['x'])
        xs = dat['x'][condition_index:end_of_condition]
        novels = dat['novel'][condition_index:end_of_condition]
        ys = list(cumulative_list(novels))
        parts = dat['participant'][condition_index:end_of_condition]
            
        ax.set_xlabel("number of instances received")
        ax.set_ylabel("number of unique ideas")

        p_rate_hpds = [mystats.mean_and_hpd(p_rate, 0.95) for p_rate in p_rates]
        p_rate_mean = [hpd[0] for hpd in p_rate_hpds]
        p_rate_low = [hpd[1] for hpd in p_rate_hpds]
        p_rate_high = [hpd[2] for hpd in p_rate_hpds]

        # plot the hpd area
        bottom_ys = model_integral_predict_p_decay(parts,
                p_rate_low, min_rate_hpd[1])
        top_ys = model_integral_predict_p_decay(parts,
                p_rate_high, min_rate_hpd[2])
        ax.fill_between(xs, bottom_ys, top_ys, color='g', alpha=0.25)

        # plot the line for each condition
        ax.plot(xs, ys, '-', color='k')

        # plot the fit line
        fit_ys = model_integral_predict_p_decay(parts,
                p_rate_mean, min_rate_hpd[0])
        ax.plot(xs[:len(ys)], fit_ys, '--', color='k')

        # plot the 1:1 line
        ys = [x for x in xs]
        ax.plot(xs, ys, '--', color='k', alpha=0.5)

        condition_index = end_of_condition

    #plt.show()
    fig.savefig('figures/bernoulli_decay_participant_model_example', dpi=600)

def n_tiles(l, n):
    """ Returns l split into n chunks
    """
    size = len(l) / n;
    for i in range(n):
        yield l[n * i : n * i+1]
        
def add_hpd_bar(ax, left, right, y, linewidth=2, edge_height = 200):
    heh = edge_height / 2
    ax.plot([left, right], [y, y], color='k', linewidth=linewidth)
    ax.plot([left, left], [y + heh, y - heh], color='k', linewidth=linewidth)
    ax.plot([right, right], [y + heh, y - heh], color='k', linewidth=linewidth)

def plot_rate_posteriors(p_rates, qcs, parts):
    fig = plt.figure(figsize=(8,10))
    fig.subplots_adjust(hspace=.5)

    first_ax = None
    for i, qc in enumerate(set(qcs)):
        if first_ax is None:
            ax = fig.add_subplot(4, 1, i+1)
            first_ax = ax
        else:
            ax = fig.add_subplot(4, 1, i+1, sharex=first_ax, sharey=first_ax)
            #ax.set_xticklabels([])

        qc_parts = set()
        for pqc, part in zip(qcs, parts):
            if pqc == qc:
                qc_parts.add(part)

        qcp_rates = [pr for i, pr in enumerate(p_rates) if i in qc_parts]
        
        sorted_rates = sorted(qcp_rates, key=np.mean)

        # Plot highest and lowest rates
        ax.hist(sorted_rates[0], bins=10, alpha=0.75,
                label="fastest decay")
        ax.hist(sorted_rates[-1], bins=10, alpha=0.75,
                label="slowest decay")

        fast_hpd = mystats.mean_and_hpd(sorted_rates[0], 0.95)
        slow_hpd = mystats.mean_and_hpd(sorted_rates[-1], 0.95)
        add_hpd_bar(ax, fast_hpd[1], fast_hpd[2], 500)
        add_hpd_bar(ax, slow_hpd[1], slow_hpd[2], 500)

        ax.set_xlabel("rate parameter value")
        ax.set_ylabel("number of samples")
    #ax.set_xlim((-0.015, 0))
        ax.legend(loc=2)
        ax.set_title(qc)

    #plt.show()
    fig.savefig('figures/bernoulli_decay_participant_posteriors', dpi=600)

# TODO: this could be done with passed parameters
def filter_today(df):
    #df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    #df = df[(df['question_code'] == 'forgot_name')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    print('\033[1m' +  os.path.basename(__file__) + '\033[0m')

    n_iter = 3000
    n_chains = 3
    min_received = 50

    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, ifs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(ifs, idf)
    df = df[df['num_received'] >= min_received]

    dat, qcs = gen_model_data(df, rmdf, cdf, ifs)
    param_walks = modeling.compile_and_fit(model_string, dat, n_iter, n_chains)

    predictions = view_fit(df, 'idea', param_walks[0], dat, qcs)

    with open('tex/bernoulli_decay_participant_anal.tex', 'w') as f:
        print(anal_string(min_received, n_chains, n_iter, predictions), file=f)
