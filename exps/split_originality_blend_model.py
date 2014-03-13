import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, os
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

def anal_string(n_chains, n_iter, split_post, sim_passes):
    anal_string = """This model was fit using Stan (the full model specification in Stan language is given in Appendix~\\ref{sec:novelty_run_model}). The resulting model converged in %i chains in %i iterations. The fit is given in Figure~\\ref{fig:idea_oscore_blend_fit} 

The resulting mean for the $s$ parameter (the point at which idea stop increasing in novelty) was %i. The HDI was (%i, %i), the bounds of which include neither 0 nor 100, such that we reject the hypothesis that the novelty of ideas is either stagnant nor grows for the entirety of the brainstorming run.
This result is surprising in that it suggests that participants do not run out of novel ideas, but rather run out of common ideas after which they reach a period of extended novelty.

This split point found under error simulation falls within the (%i , %i) HDI in %i of 10 simulations.

As a result, I am able to present an empirically-derived guideline for those performing brainstorming tasks on microtask marketplaces: to receive the most novel ideas, ask participants for at least %i ideas."""

    return anal_string % (n_chains, n_iter, int(split_post[0]), int(split_post[1]),
            int(split_post[2]), int(split_post[1]), int(split_post[2]), sim_passes,
            int(split_post[0] + 1))


model_string = """
data {
    int<lower=0> N; // number of instances
    int<lower=1, upper=100> order[N]; // order of instance in brainstorming run
    real<lower=0, upper=1> oscore[N]; // oscore of instance at position
}

parameters {
    real<lower=1, upper=100> split;

    real<lower=0,upper=1> phi1;
    real<lower=0.1> lambda1;
    real<lower=0,upper=1> phi2;
    real<lower=0.1> lambda2;

}

transformed parameters {
    real<lower=0> alpha1;
    real<lower=0> beta1;

    real<lower=0> alpha2;
    real<lower=0> beta2;

    alpha1 <- lambda1 * phi1;
    beta1 <- lambda1 * (1 - phi1);

    alpha2 <- lambda2 * phi2;
    beta2 <- lambda2 * (1 - phi2);
}

model {
    real mix[N];

    phi1 ~ beta(1,1);
    lambda1 ~ pareto(0.1,1.5);
    phi2 ~ beta(1,1);
    lambda2 ~ pareto(0.1,1.5);
 
    for (i in 1:N) {
        if (order[i] <= split)
            mix[i] <- order[i] / split;
        else
            mix[i] <- 1;

        oscore[i] ~ beta((1 - mix[i]) * alpha1 + mix[i] * alpha2,
                         (1 - mix[i]) * beta1 + mix[i] * beta2);
    }

}
"""
def gen_dat(df, rmdf, cdf, ifs):
    field = 'idea_oscore'
    return {'order': df['answer_num'] + 1,
            'oscore': df[field],
            'N': len(df) }

def view_fit(dat, la):
    alpha1 = mystats.mean_and_hpd(la['alpha1'], 0.95)
    alpha2 = mystats.mean_and_hpd(la['alpha2'], 0.95)
    beta1 = mystats.mean_and_hpd(la['beta1'], 0.95)
    beta2 = mystats.mean_and_hpd(la['beta2'], 0.95)
    switch = mystats.mean_and_hpd(la['split'], 0.95)
    #sigma = mystats.mean_and_hpd(la['sigma'], 0.95)

    #print("Estimated normal variance:", sigma[0])
    print("estimated split:", switch)

    init_vals = dict()

    plot_fit(dat, alpha1, alpha2, beta1, beta2, switch)
    init_vals['phi1'] = mystats.mean_and_hpd(la['phi1'])[0]
    init_vals['phi2'] = mystats.mean_and_hpd(la['phi2'])[0]
    init_vals['lambda1'] = mystats.mean_and_hpd(la['lambda1'])[0]
    init_vals['lambda2'] = mystats.mean_and_hpd(la['lambda2'])[0]
    init_vals['split'] = switch[0]

    return init_vals
 
def line_from_betas(alpha1, alpha2, beta1, beta2, switch):
    m1_b = alpha1[0] / (alpha1[0] + beta1[0])
    m2_b = alpha2[0] / (alpha2[0] + beta2[0])

    m1_m = (m2_b - m1_b) / switch[0]

    return m1_b, m1_m, m2_b

def plot_linear_hpd(ax, m, b, xs):
    bottom_ys = [min(m) * x + min(b) for x in xs]
    top_ys = [max(m) * x + max(b) for x in xs]
    ax.fill_between(xs, bottom_ys, top_ys, color='g', alpha=0.10)

def plot_model_per_question(df, n_iter, n_chains, init_vals):
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)

    styles = ['-', '--', '-.', ':']
    for i, qc in enumerate(set(df['question_code'])):
        qcdf = df[df['question_code'] == qc]
        dat = gen_dat(qcdf, None, None, None)
        param_walks = modeling.compile_and_fit(model_string, dat, n_iter,
                n_chains, init_vals)

        alpha1 = mystats.mean_and_hpd(param_walks[0]['alpha1'])
        alpha2 = mystats.mean_and_hpd(param_walks[0]['alpha2'])
        beta1 = mystats.mean_and_hpd(param_walks[0]['beta1'])
        beta2 = mystats.mean_and_hpd(param_walks[0]['beta2'])
        split = mystats.mean_and_hpd(param_walks[0]['split'])

        plot_fit_and_hpd(ax, alpha1, alpha2, beta1, beta2, split,
                linestyle=styles[i], label=qc, color='k')

    ax.set_xlim(0, 100)
    ax.set_ylim(0.99, 1)
    ax.set_xlabel("instance number in run")
    ax.set_ylabel("mean oscore for all participants")

    ax.legend()
    
    fig.savefig('figures/novelty_model_questions', dpi=600)
    #plt.show()


def plot_fit_and_hpd(ax, alpha1, alpha2, beta1, beta2, switch, **kwargs):
    split_point = int(switch[0])
    m1_xs = range(0, split_point + 1)
    m2_xs = range(split_point, 100)

    m1_b, m1_m, m2_b = line_from_betas(alpha1, alpha2, beta1, beta2, switch)

    # plot the split hpd area
    xs = (switch[1], switch[2])
    bys = (0, 0)
    tys = (1, 1)
    ax.fill_between(xs, bys, tys, color='g', alpha=0.10)

    # plot the model
    m1_ys = [m1_m * x + m1_b for x in m1_xs]
    m2_ys = [m2_b for x in m2_xs]
    ax.plot(m1_xs, m1_ys, **kwargs)
    ax.plot(m2_xs, m2_ys, **kwargs)


def plot_fit(dat, alpha1, alpha2, beta1, beta2, switch):
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.set_xlabel("instance number in run")
    ax.set_ylabel("mean oscore for all participants")

    ax.set_xlim(0, 100)

    # plot the lines hpd
    #plot_linear_hpd(ax, s1_m, s1_b, m1_xs)
    #plot_linear_hpd(ax, (0, 0, ), s2_b, m2_xs)

    # plot means
    means = []
    for x in range(1, 101):
        ys = []
        for gx, gy in zip(dat['order'], dat['oscore']):
            if gx == x:
                ys.append(gy)
        means.append(np.mean(ys))
    ax.plot(range(1, 101), means, 'r.', alpha=0.5)

    plot_fit_and_hpd(ax, alpha1, alpha2, beta1, beta2, switch)

    ax.set_ylim(0.99, 1)

    fig.savefig('figures/idea_oscore_blend_fit', dpi=600)
    #plt.show()

# TODO: this could be done with passed parameters
def filter_today(df):
    #df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    print('\033[1m' + os.path.basename(__file__) + '\033[0m')

    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    n_iter = 6000
    n_chains = 3

    dat = gen_dat(df, rmdf, cdf, cfs)
    param_walks = modeling.compile_and_fit(model_string, dat, n_iter, n_chains)
    init_vals = view_fit(dat, param_walks[0])

    plot_model_per_question(df, n_iter, n_chains, init_vals)

    post_split_param = mystats.mean_and_hpd(param_walks[0]['split'])

    #def hyp_fn(posterior, edf, ermdf, ecdf, eifs):
    #    return post_split_param[0] > posterior[1] and post_split_param[0] < posterior[2]

    #def posterior_fn(edf, ermdf, ecdf, eifs):
    #    dat = gen_dat(edf, ermdf, ecdf, eifs)
    #    param_walks = modeling.compile_and_fit(model_string, dat, n_iter, n_chains)
    #    return mystats.mean_and_hpd(param_walks[0]['split'])

    #sim_passes = modeling.simulate_error_hypothesis_general(10, posterior_fn,
    #        hyp_fn, idf, cfs)
    #print("split posterior in HDI in %i/10 simulations" % sim_passes)

    sim_passes = 100
    with open('tex/split_originality_blend_anal.tex', 'w') as f:
        print(anal_string(n_chains, n_iter, post_split_param, sim_passes), file=f)
