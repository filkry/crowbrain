import pandas as pd
import numpy as np
import re, pystan, format_data, modeling
import matplotlib.pyplot as plt
import stats_fns as mystats
from collections import defaultdict, OrderedDict

def filter_today(df):
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    example = df[(df['answer'] == 'part out') | (df['answer'] == 'writing')]
    st_roots = set(example['subtree_root'])

    print(st_roots)

    
