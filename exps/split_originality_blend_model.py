import pandas as pd
import numpy as np
import re, pystan, format_data, modeling
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

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

    plot_fit(dat, alpha1, alpha2, beta1, beta2, switch)
 
def line_from_betas(alpha1, alpha2, beta1, beta2, switch):
    m1_b = alpha1[0] / (alpha1[0] + beta1[0])
    m2_b = alpha2[0] / (alpha2[0] + beta2[0])

    m1_m = (m2_b - m1_b) / switch[0]

    return m1_b, m1_m, m2_b

def plot_linear_hpd(ax, m, b, xs):
    bottom_ys = [min(m) * x + min(b) for x in xs]
    top_ys = [max(m) * x + max(b) for x in xs]
    ax.fill_between(xs, bottom_ys, top_ys, color='g', alpha=0.10)

def plot_model_per_question(df, n_iter, n_chains):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    styles = ['-', '--', '-.', ':']
    for i, qc in enumerate(set(df['question_code'])):
        qcdf = df[df['question_code'] == qc]
        dat = gen_dat(qcdf, None, None, None)
        param_walks = modeling.compile_and_fit(model_string, dat, n_iter, n_chains)

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
    
    plt.show()


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
    fig = plt.figure(figsize=(10, 10))
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

    plt.show()

# TODO: this could be done with passed parameters
def filter_today(df):
    df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    n_iter = 3000
    n_chains = 3

    dat = gen_dat(df, rmdf, cdf, cfs)
    param_walks = modeling.compile_and_fit(model_string, dat, n_iter, n_chains)
    view_fit(dat, param_walks[0])

    plot_model_per_question(df, n_iter, n_chains)

    post_split_param = mystats.mean_and_hpd(param_walks[0]['split'])[0]

    def hyp_fn(posterior, edf, ermdf, ecdf, eifs):
        return post_split_param > posterior[1] and post_split_param < posterior[2]

    def posterior_fn(edf, ermdf, ecdf, eifs):
        dat = gen_dat(edf, ermdf, ecdf, eifs)
        param_walks = modeling.compile_and_fit(model_string, dat, n_iter, n_chains)
        return mystats.mean_and_hpd(param_walks[0]['split'])

    sim_passes = modeling.simulate_error_hypothesis_general(10, posterior_fn,
            hyp_fn, idf, cfs)
    print("split posterior in HDI in %i/10 simulations" % sim_passes)
