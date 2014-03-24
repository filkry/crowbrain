import nlp
import pandas as pd
import nltk
from nltk.corpus import wordnet as wn
import re, format_data, os, sys
import random
from modeling import get_redundant_data, read_or_gen_cache, hash_instance_df, hash_string
from multiprocessing import Pool


def synset_identifier(synset):
    match = re.match("Synset\('(.+)'\)", str(synset))
    return match.group(1)

def bag_synsets(answer):

    def real_compute():
        at, tokens = nlp.tokenize_sentence(answer, True)
        pos_tags = nltk.pos_tag(tokens)
        synsets = []
        for w, t in pos_tags:
            pos_code = nlp.wordnet_pos_code(t)
            if not pos_code == '':
                syns = wn.synsets(w, pos_code)
            else:
                syns = wn.synsets(w)
            synsets.append([synset_identifier(syn) for syn in syns])
        return synsets

    return read_or_gen_cache(hash_string(answer) + '.answersynset',
            real_compute)

def all_synsets_in_corpus(adf):

    def real_compute():
        all_synsets = set()
        total = len(adf)
        for i, answer in enumerate(adf['answer']):
            print("Computing synsets for all answers... %i/%i" % (i, total),
                    end='\r')
            a_ss = bag_synsets(answer)
            for word_ss in a_ss:
                for syn in word_ss:
                    all_synsets.add(syn)
        return all_synsets

    return read_or_gen_cache(hash_instance_df(adf) + '.dfallsynsets',
            real_compute)

def answer_distance(a1, a2):

    def real_compute():
        a1ss = bag_synsets(a1)
        a2ss = bag_synsets(a2)

        total = 0
        word_pair_score = 0
        for a1_word_ss in a1ss:
            for a2_word_ss in a2ss:
                all_pairings = [(a1syn, a2syn) for a1syn in a1_word_ss
                                               for a2syn in a2_word_ss]
                all_pairings = [(wn.synset(a1syn), wn.synset(a2syn))
                        for a1syn, a2syn in all_pairings]

                # filter things that can't connect because they are different types
                all_pairings = [(ss1, ss2) for ss1, ss2 in all_pairings
                        if ss2.pos() == ss1.pos()]

                if len(all_pairings) == 0:
                    continue
                total += 1

                # get the distance for all pairings and take the mean
                sum_score = 0.0
                for ss1, ss2 in all_pairings:
                    path_sim = ss1.path_similarity(ss2)
                    if path_sim is not None:
                        sum_score += path_sim
                word_pair_score += (sum_score / len(all_pairings))

        if total == 0:
            return 0

        return word_pair_score / total 

    return read_or_gen_cache(hash_string(a1) + hash_string(a2) + '.wordnetsimilarity',
            real_compute)

def run_wn_sim_scores(rdf):
    assert(len(set(rdf['worker_id'])) == 1)

    def real_compute():
        scores = []
        answers = list(rdf['answer'])
        #pair_count = 0
        for i, a1 in enumerate(answers[:-1]):
            for a2 in answers[i+1:]:
                #print("Computing answer pairing %i" % pair_count, end='\r')
                #pair_count += 1
                scores.append(answer_distance(a1, a2))
        return scores

    worker_id = rdf['worker_id'].iloc[0]
    qc = rdf['question_code'].iloc[0]
    hsh = '%s_%s.wordnetworkersimscores' % (worker_id, qc)
    return read_or_gen_cache(hsh, real_compute)

def runs_wn_sim_scored(rdfs):
    for i, rdf in enumerate(rdfs):
        print("Computing run wordnet similarity scores for worker %i/%i" % (i, len(rdfs)))
        run_wn_sim_scores(rdf)

def filter_today(df):
    #df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]

if __name__ == '__main__':
    print(os.path.basename(__file__))

    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = get_redundant_data(cfs, idf)

    all_syns = all_synsets_in_corpus(df)
    print("Number of synonyms in dataset:", len(all_syns))

    # precompute all wordnet similarity scores
    all_runs = []
    for name, run in df.groupby(['worker_id', 'num_requested', 'question_code']):
        all_runs.append(run)
    #random.shuffle(all_runs)

    n_processes = 8
    #run_chunks = list(chunks(all_runs, int(len(all_runs)/ n_processes)))

    p = Pool(n_processes)
    #p.map(runs_wn_sim_scored, run_chunks)

    num_tasks = len(all_runs)
    for i, _ in enumerate(p.imap_unordered(run_wn_sim_scores, all_runs), 1):
        sys.stderr.write('\rdone {0:%}'.format(i/num_tasks))
    

