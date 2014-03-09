import pandas as pd
import numpy as np
import re, pystan, format_data, modeling, simulate_error_tree
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

def filter_today(df):
    df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df

def node_binning_plot(cdf, qc, n_bins = 5):
    qcdf = cdf[cdf['question_code'] == qc]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('node 1 size')
    ax.set_ylabel('node 2 size')

    # Draw shaded bins
    max_size = max(qcdf['num_instances'])
    bin_size = max_size / n_bins
    bin_starts = [bin_size * i for i in range(n_bins)]
    bin_ends = [bin_size * i for i in range(1, n_bins + 1)]
    for bin1 in range(n_bins):
        for bin2 in range(bin1 + 1):
            ax.fill_between(
                    [bin_starts[bin1], bin_ends[bin1]],
                    [bin_starts[bin2], bin_starts[bin2]],
                    [bin_ends[bin2], bin_ends[bin2]],
                    color='k',
                    alpha=0.17)
            

    # Draw bin lines
    for sep in bin_ends[:-1]:
        ax.plot([0, max_size], [sep, sep], 'k-', linewidth=2)
        ax.plot([sep, sep], [0, max_size], 'k-', linewidth=2)
    ax.set_xlim(0, max_size)
    ax.set_ylim(0, max_size)

    # Draw node pairings
    pairs = [(x, y) for x in qcdf['num_instances']
                    for y in qcdf['num_instances']]
    pxs, pys = zip(*pairs)
    ax.plot(pxs, pys, '+')

    plt.show()


if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    node_binning_plot(cdf, 'iPod')
    

