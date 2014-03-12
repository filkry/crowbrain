import random
import format_data
import modeling
import math

survey_group_text_cache = dict()

question_texts = {
        'turk': """Mechanical Turk currently lacks a dedicated mobile app for performing HITs on
smartphones (iPhone, Androids, etc.) or tablets (e.g., the iPad).
                    
Brainstorm N features for a mobile app to Mechanical Turk that would improve
the worker's experience when performing HITs on mobile devices. Be as specific
as possible in your responses.""",
        'charity': """The Electronic Frontier Foundation (EFF) is a nonprofit whose goal is to protect
individual rights with respect to digital and online technologies. For example, the EFF has initiated a
lawsuit against the US government to limit the degree to which the US surveils its citizens via secret
NSA programs. If you are unfamiliar with the EFF and its goals, read about it on its website
(https://www.eff.org) or via other online sources (such as Wikipedia).

Brainstorm N new ways the EFF can raise funds and simultaneously increase awareness. Your ideas must be
different from their current methods, which include donation pages, merchandise, web badges and banners,
affiliate programs with Amazon and eBay, and donating things such as airmiles, cars, or stocks. See the
full list of their current methods here: https://www.eff.org/helpout. Be as specific as possible in your
responses. """,
        'iPod': """Many people have old iPods or MP3 players that they no longer use. Please brainstorm
N uses for old iPods/MP3 players. Assume that the devices' batteries no longer work, though they can be
powered via external power sources. Also be aware that devices may not have displays. Be as specific as
possible in your descriptions.""",
        'forgot_name': """Imagine you are in a social setting and you have forgotten the name of
somebody you know. Brainstorm N ways you could learn their name without directly asking them. Be
as specific as possible in your descriptions.""",
}


def choose(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))
    

def bin_sequence(seq, bin_val, num_bins):
    bin_vals = [bin_val(s) for s in seq]
    max_val = max(bin_vals)
    min_val = min(bin_vals)
    
    bins = [[] for i in range(num_bins)]
    
    for s in seq:
        val = bin_val(s)
        # min fixes inclusive upper bound
        b = min(num_bins - 1, math.floor(num_bins * float(val - min_val) / float(max_val - min_val)))
        bins[b].append(s)
        
    return bins

def sample_bin_pairings(bins, num_samples):
    samples = []
    
    for i, bn1 in enumerate(bins):
        for j, bn2 in enumerate(bins[:i+1]):
            combs = None
            if i != j:
                combs = [(i, j, e1, e2) for e1 in bn1 for e2 in bn2]
            else: # don't pair nodes with themselves
                combs = [(i, j, e1, e2) for k, e1 in enumerate(bn1[:-1])
                                  for e2 in bn1[k+1:]]
            
            bin_samples = random.sample(combs, min(len(combs), num_samples))
            print(i, j, len(bn1), len(bn2), len(bin_samples))
            samples.extend(bin_samples)
            
    return samples

def gen_survey_group_text(bin1, bin2, n1id, n2id):
    if (n1id, n2id) in survey_group_text_cache:
        return survey_group_text_cache[(n1id, n2id)]
    
    text = []
    text.append("\nGroup 1 (%d):" % n1id)
    
    n1df = df[df['idea'] == n1id]
    n2df = df[df['idea'] == n2id]
    
    assert(len(n1df) > 0)
    assert(len(n2df) > 0)
    
    n1_ideas = random.sample(list(n1df['answer']), min(3, len(n1df)))
    n2_ideas = random.sample(list(n2df['answer']), min(3, len(n2df)))
    
    for idea in n1_ideas:
        text.append('[ ] ' + idea)
        
    text.append("\nGroup 2 (%d):" % n2id)
    
    for idea in n2_ideas:
        text.append('[ ] ' + idea)
        
    survey_group_text_cache[(n1id, n2id)] = text
        
    return text

def gen_survey_questions(pairings, qc):
    text = []
    text.append("""For each of the following questions, you will be presented two groups
                   of three ideas each. Each idea was given in response to this brainstorming
                   task:
                   """)

    text.append(question_texts[qc])
                                      
    text.append("""
                   First, for each group, put a small X beside any idea that is not the same idea
                   as the others (with allowances for rephrasing). If none of the ideas are the same,
                   mark them all with Xs.
                   
                   Then, mark one of the 5
                   options for relationships between group 1 and group 2 of ideas.""")
    for i, pairing in enumerate(pairings):
        bin1, bin2, (n1id, _), (n2id, _) = pairing
    
        text.append("\n\n====================\nQuestion %d (%d x %d)\n====================" % (i+1, bin1, bin2))
        
        text.extend(gen_survey_group_text(bin1, bin2, n1id, n2id))
            
        text.append("\nRelationship between groups:")
        text.append('[ ] group 1 ideas are generalizations of group 2 ideas')
        text.append('[ ] group 2 ideas are generalizations of group 1 ideas')
        text.append('[ ] group 1 and group 2 are unrelated')
        text.append('[ ] group 1 and 2 are related, but not the same, and neither is a generalization of the other')
        text.append('[ ] group 1 and group 2 are the same')
        
    return '\n'.join(text)

def filter_today(df):
    #df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df

if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    for qc in cfs.keys():
        qc_cdf = cdf[(cdf['question_code'] == qc) & (cdf['num_instances'] > 0)]
        
        nodes = list(zip(list(qc_cdf['idea']), list(qc_cdf['num_instances'])))
        bins = bin_sequence(nodes, lambda x: x[1], 5)
        pairings = sample_bin_pairings(bins, 10)
        random.shuffle(pairings)
        
        num_fixed_pairings = 10
        fixed_pairings = pairings[:num_fixed_pairings]
        num_judges = 3
        for j in range(num_judges):
            with open('ipython_output/validity_survey_judge_%s_%i.txt' % (qc, j), 'w') as f:
                rest_pairings = [p for i, p in enumerate(pairings[num_fixed_pairings:]) if i % num_judges == j]
                judge_pairings = fixed_pairings + rest_pairings
                f.write(gen_survey_questions(judge_pairings, qc))
