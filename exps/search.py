import pandas as pd
import numpy as np
import sys
import format_data, modeling, os
import matplotlib.pyplot as plt

def filter_today(df):
    #df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    #df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    print(os.path.basename(__file__))
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    #df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    s_str = sys.argv[1]
    print("Searching for", s_str)

    for ix in idf.index:
        answer = idf.answer[ix]
        wn = idf.worker_num[ix]

        if s_str in answer:
            print(wn, ':', answer)

