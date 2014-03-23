import pandas as pd
import numpy as np
import format_data, modeling, os
import matplotlib.pyplot as plt

def responses_of_interest():
    return [
        ['have them as a resource on public transportation'],
        ['use in abstract art'],
        ['get it fixed and'],
        ['building material', 'brick'],
        ['extract usage data', 'extract texts'],
        ['Jukebox music selector'],
        ['We could use the hard-drives inside'],
        ['I suppose you could just grind them down into a sand'],
        ['decorate shoes'],
        ['use old parts to make a new device'],
        ['melt down old parts', 'see what old parts'],
        ['throw it in the ocean'],
        ['doorstop'],
        ['Remove glass screen to make something'],
        ['give to non-profit that can benefit'],
    ]

def find_wns(adf, responses):
    for response_list in responses:
        candidate_wns = set(adf['worker_num'])

        for n in response_list[0:]:
            match = adf[adf['answer'].str.contains(n)]
            candidate_wns = candidate_wns & set(match['worker_num'])

        if(len(candidate_wns) < 1):
            print(response_list, len(candidate_wns))
        assert(len(candidate_wns) > 0)
        yield candidate_wns.pop()

def print_resps_wns(resps, wns):
    for resp, wn in zip(resps, wns):
        resp_str = ', '.join(resp)
        print("{resp:60} : {worker_num:d}".format(
            resp = resp_str,
            worker_num = int(wn)))

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

    resps = responses_of_interest()
    print_resps_wns(resps, find_wns(idf, resps))


