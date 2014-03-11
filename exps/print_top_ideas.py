import format_data, modeling

def filter_today(df):
    df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    scdf = cdf.sort('num_instances', ascending=False)
    top_3_ideas = scdf['idea'].iloc[:3]

    for idea in top_3_ideas:
        print("idea", idea)
        idf = df[df['idea'] == idea]
        print(idf['answer'].iloc[:10])
