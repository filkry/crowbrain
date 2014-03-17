import pandas as pd
import csv, os
import re
import networkx as nx
import numpy as np
from numpy import uint8, uint16, uint32, uint64, datetime64, int64, int32, float64
from collections import defaultdict, OrderedDict

def filter_repeats(df):
    new_df = df[df['is_repeat_worker'] == 0]
    return new_df

def metrics_folder(pd_folder, x):
        return '/%s/pilot18_metrics/%s' % (pd_folder, x)

def read_base_data(dirs):
    dfs = []

    for d in dirs:
      for f in os.listdir(d):
        full_name = os.path.join(d,f)
        if f == 'answers.csv':
            dfs.append(pd.read_csv(full_name, sep=',', quotechar='|',
                quoting=csv.QUOTE_NONNUMERIC, parse_dates=[12, 13],
                header=None, skiprows=1,
                names=['worker_id', 'question', 'question_code', 'post_date',
                    'screenshot', 'num_requested', 'answer_num', 'answer',
                    'word_count', 'start_time', 'end_time', 'answer_code',
                    'submit_datetime', 'accept_datetime']))

    con = pd.concat(dfs, ignore_index = True)
    return con

def read_cluster_data(files):
    dfs = []

    for f in files:
        dfs.append(pd.read_csv(f, sep=",", quotechar='"',
            header=None, skiprows=1,
            names =['question_code', 'answer_id', 'idea', 'answer', 'answer_num',
                'worker_id', 'post_date', 'num_requested']))

    df = pd.concat(dfs)
    return df

def read_file(f):
    with open(f) as fin:
        l = fin.readline()
        # If we see the follow, it's an early run we can ignore
        if 'Number of Answers' in l:
            return
        fin.seek(0)
        sep = '|'
        # Guess separator character
        if l.count(',') > l.count('|'):
            sep = ','
            
        # Just cache for now
        rows = []
        for row in csv.reader(fin, delimiter=sep):
            if 'hashed_worker_id' in row or 'worker_id' in row or 'cluster_parent' in row:
                continue
            rows.append(row + [f])
            
        return rows

def dumb_strip(s):
    return ''.join([c for c in s if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ\
    abcdefghijklmnopqrstuvwxyz1234567890 "])

def gen_depth(forest, node):
    parents = forest.predecessors(node)
    if parents:
        p = parents[0]
        if 'depth' not in forest.node[p]:
            gen_depth(forest, p)
        forest.node[node]['depth'] = forest.node[p]['depth'] + 1
    else:
        forest.node[node]['depth'] = 0
    
    smd = forest.graph['subtree_max_depth']
    root = forest.node[node]['subtree_root']
    smd[root] = max(smd[root], forest.node[node]['depth'])

def all_nodes_under(forest, node):
    ret = [node]
    for s in forest.successors(node):
        ret += all_nodes_under(forest, s)
    return ret


def annotated_cluster_forest(f):
    f = f.copy()

    for node in f.nodes():        
        # Subtree root
        cur = node
        passed = []
        while(len(f.predecessors(cur)) > 0):
            passed.append(cur)
            cur = f.predecessors(cur)[0]
            if(cur in passed):
                print("Cycle")
                break

        f.node[node]['subtree_root'] = cur
        f.graph['subtree_roots'].add(cur)
        
        # descendents
        f.node[node]['all_nodes_under'] = all_nodes_under(f, node)
        
    # depth
    for node in f.nodes():
        if 'depth' not in f.node[node]:
            gen_depth(f, node)
            
    # height
    for n in f.nodes():
        smd = f.graph['subtree_max_depth']
        root = f.node[n]['subtree_root']
        f.node[n]['height'] = smd[root] - f.node[n]['depth']
        
    # remix_targets
    for n in f.nodes():
        targets = set([n])
        
        # all parents
        cur = n
        while len(f.predecessors(cur)) > 0:
            cur = f.predecessors(cur)[0]
            targets.add(cur)
        
        # all siblings
        for p in f.predecessors(n):
            for s in f.successors(p):
                targets.add(s)
        
        # all children
        targets = targets.union(set(f.node[n]["all_nodes_under"]))
        
        f.node[n]['remix_of'] = targets

    return f
     

def mk_cluster_forest(structure_csv):
    f = nx.DiGraph()
    f.graph['subtree_roots'] = set()
    f.graph['subtree_max_depth'] = defaultdict(int)
    
    rows = read_file(structure_csv)
        
    if len(rows) == 0:
        return f
    
    for qc, child, parent, label, fn in rows:
        c = int(child)
        p = int(parent)
        if parent != '':
            f.add_edge(p, c)
        else:
            f.add_node(c)
        f.node[c]['label'] = dumb_strip(label) if len(label) > 0 else None
            
    # delete the single root node
    root = nx.topological_sort(f)[0]
    f.remove_nodes_from([root])
    
    return f

def num_instances_in(df, clusters):
    return sum(len(df[df['idea'] == c]) for c in clusters)

# TODO: function will need revision with new data model
def compute_mixing(clustered_df, df, cluster_forests):
    dist = pd.Series([None for i in df.index], index=df.index)
    dist_im = pd.Series([None for i in df.index], index=df.index)
    im = pd.Series([0 for i in df.index], index=df.index)
    mm = pd.Series([0 for i in df.index], index=df.index)
    om = pd.Series([0 for i in df.index], index=df.index)
    last_sim = pd.Series([None for i in df.index], index=df.index)
    related_inmix = pd.Series([None for i in df.index], index=df.index)
    
    for (nr, wid, qc), run in clustered_df.groupby(['num_requested', 'worker_id', 'question_code']):
        for ii, i in enumerate(run.index):
            if ii == 0:
                continue
            for jj, j in reversed(list(enumerate(run.index[0:ii]))):
                j_clus = run['idea'][j]
                hcm_nc = cluster_forests[qc].node[j_clus]['remix_of']
                if run['idea'][i] in hcm_nc:
                    last_sim[i] = j
                    dist[i] = ii - jj
                    mm[j] = 1
                    
                    if om[j] > 0:
                        dist_im[i] = dist[i] + dist_im[j]
                        related_inmix[i] = related_inmix[j]
                    else:
                        im[j] = 1
                        dist_im[i] = dist[i]
                        related_inmix[i] = j
                    
                    om[i] = 1
                    assert dist[i] > 0
                    break
    return dist, im, om, mm, dist_im, last_sim, related_inmix

def mk_run_count(instance_df):
    runs = instance_df.groupby(['num_requested', 'worker_id',
        'question_code', 'submit_datetime', 'accept_datetime'])
    return runs.size()


def mk_redundant_run_helper(full_df):
    runs = full_df.groupby(['num_requested', 'worker_id',
        'question_code', 'submit_datetime', 'accept_datetime'],
        as_index=False)

    agg_dict = {'answer': len }
    column_names = ['num_requested', 'worker_id', 'question_code',
            'submit_datetime', 'accept_datetime', 'num_received']

    rmdf = runs.agg(agg_dict)
    rmdf.columns = column_names

    return rmdf

    # num unique ideas
    # num unique subtrees
    # num inmixes
    # num outmixes

def mk_repeat_workers_series(adf):
    adf = adf.copy() # don't sort the passed dataframe
    
    is_repeat = pd.Series([0 for i in adf.index], index=adf.index)
    adf = adf.sort(['submit_datetime'])
    runs = adf.groupby(['worker_id', 'question_code', 'num_requested', 'submit_datetime'])

    seen_keys = set()
    for rid, ((wid, qc, nr, sdt), run) in enumerate(runs):
        assert (nr >= len(run) or nr == 0)

        if (wid, qc) in seen_keys:
            for i in run.index:
                is_repeat[i] = 1
        else:
            seen_keys.add((wid, qc))

    return is_repeat
                

def dump_csvs(processed_data_folder, idf):
    df_output_csv = '/%s/pilot18_notebook_output/instances.csv' % processed_data_folder
    #clustersdf_output_csv = '/%s/pilot18_notebook_output/cluster_trees.csv' % processed_data_folder

    idf.to_csv(df_output_csv)
    #clusters_df.to_csv(clustersdf_output_csv)

def mk_redundant(idf, cluster_forests):
    # annotate the cluster_forests
    ann_cfs = {key:annotated_cluster_forest(cluster_forests[key]) for key in cluster_forests} 

    # build cluster dataframe
    clusters_df = mk_redundant_cluster_df_helper(idf, ann_cfs)
    full_df = pd.merge(idf, clusters_df, 'left', ['idea', 'question_code'])

    for idx in clusters_df.index:
        if clusters_df['num_instances'][idx] > 0:
            assert(len(full_df[full_df['idea'] == clusters_df['idea'][idx]]) > 0)


    full_df['time_spent'] = full_df['end_time'] - full_df['start_time']
    #assert(min(full_df[!full_df['time_spent'].isnull()]['time_spent']) > 0)

    # Not using in current analysis, and can't be trusted anyway
    full_df = mk_redundant_riffing_helper(full_df, ann_cfs)

    rmdf = mk_redundant_run_helper(full_df) # TODO: needs a lot more metrics

    full_df = pd.merge(full_df,
            rmdf,
            'left',
            ['num_requested', 'worker_id', 'question_code', 'submit_datetime', 'accept_datetime'])

    return full_df, rmdf, clusters_df, ann_cfs


def mk_redundant_riffing_helper(full_df, ann_cluster_forests):
    cv_df = full_df[~full_df['idea'].isnull()]
    dist, im, om, mm, dist_im, last_sim, related_inmix = compute_mixing(cv_df,
            full_df, ann_cluster_forests)

    full_df['distance_from_similar'] = dist
    full_df['is_inmix'] = im
    full_df['is_midmix'] = mm
    full_df['is_outmix'] = om
    full_df['distance_from_inmix'] = dist_im
    full_df['previous_similar_inmix'] = last_sim
    full_df['inmix_index'] = related_inmix

    return full_df


def mk_redundant_cluster_df_helper(idf, ann_cluster_forests):
    #print(len(set(idf['idea'])),  sum(len(ann_cluster_forests[qc].nodes()) for qc in ann_cluster_forests))
    #assert(len(set(idf['idea'])) == sum(len(ann_cluster_forests[qc].nodes()) for qc in ann_cluster_forests))

    fields = defaultdict(list)
    for qc in ann_cluster_forests.keys():
        sub_df = idf[idf['question_code'] == qc]
        if len(sub_df) == 0:
            continue

        f = ann_cluster_forests[qc]
        total = len(set(sub_df['idea']));
        #for i, idea in enumerate(f.nodes()):
        for i, idea in enumerate(set(sub_df['idea'])):
            assert(idea in f.nodes())

            print("mk_redundant_cluster_df_helper: %i/%i for %i instances" % (i+1, total, len(idf)), end='\r')
            nd = f.node[idea]
            
            fields['idea'].append(idea)
            fields['idea_label'].append(nd['label'])
            preds = f.predecessors(idea)
            fields['parent'].append(preds[0] if len(preds) > 0 else -1)
            fields['is_root'].append(idea == nd['subtree_root'])
            fields['is_leaf'].append(len(f.successors(idea)) == 0)
            fields['num_children'].append(len(f.successors(idea)))
            fields['subtree_root'].append(nd['subtree_root'])
            fields['depth_in_subtree'].append(nd['depth'])
            fields['height_in_subtree'].append(nd['height'])
            fields['question_code'].append(qc)
            fields['num_nodes_under'].append(len(nd['all_nodes_under']))

            # Metrics requiring instance info
            fields['num_instances_under'].append(num_instances_in(sub_df, nd['all_nodes_under']))
            nus = num_instances_in(sub_df, f.node[nd['subtree_root']]['all_nodes_under'])
            fields['subtree_probability'].append(float(nus) / len(sub_df))

            nii = num_instances_in(sub_df,[idea])
            fields['idea_probability'].append(float(nii) / len(sub_df))
            if not (float(nii) >= 0 and float(nii) < len(sub_df)):
                print('\n', nii, len(sub_df))
                assert(False)
            fields['num_ideas_under'].append(nii)

            instance_df = sub_df[sub_df['idea'] == idea]
            fields['num_instances'].append(len(instance_df))
            fields['num_workers'].append(len(set(instance_df['worker_id'])))
        print("")
    clusters_df = pd.DataFrame(
        {key: pd.Series(fields[key]) for key in fields})

    clusters_df['subtree_oscore'] = 1 - clusters_df['subtree_probability']
    clusters_df['idea_oscore'] = 1 - clusters_df['idea_probability']
    for ios in clusters_df['idea_oscore']:
        assert(ios >= 0 and ios <= 1)

    return clusters_df
            

          

def add_worker_nums(adf):    
    next_wn = 0
    wns = dict()
    wn_row = pd.Series(index=adf.index)
    for ix in adf.index:
        wid = adf['worker_id'][ix]
        if wid not in wns:
            wns[wid] = next_wn
            next_wn += 1
        wn_row[ix] = wns[wid]

    adf['worker_num'] = wn_row
    return adf


def do_format_data(processed_data_folder, filter_instances = None):
    """
    This function is pretty gigantic and ugly. It's a copy-paste of about half
    an iPython notebook that was gradually built up to get my data into usable
    form. Now that this part is rarely, if ever, touched, I wanted to get it
    out of the way.

    Takes the folder that holds all the data, and returns instance dataframe,
    cluster dataframe, run dataframe, and cluster forests
    """

    base_data_dirs = map(lambda x : '%s/pilot%i' % (processed_data_folder, x),
            [11, 12, 13, 14, 16, 17, 18])

    idea_cluster_csvs = {qc: metrics_folder(processed_data_folder, "_%s.csv" % qc) for qc in \
                         ['charity', 'iPod', 'forgot_name', 'turk']}
                         #['iPod', 'turk']}
    cluster_tree_csvs = {qc: metrics_folder(processed_data_folder, "_%s_clusters.csv" % qc) for qc in \
                         ['charity', 'iPod', 'forgot_name', 'turk']}
                         #['iPod', 'turk']}
        
    
    merge_column_names = ['worker_id', 'question_code', 'answer_num', 'num_requested']
 
    # ========================================================
    # INSTANCE DATA
    # ========================================================

       
    # Read in the main processed data files
    idf_base = read_base_data(base_data_dirs)

    # Add worker numbers
    idf_base = add_worker_nums(idf_base)

    qc_conds = set(idf_base['question_code'])

    idf_base = idf_base[~(idf_base['num_requested'] == 0)]

    # Check for repeat workers
    idf_base['is_repeat_worker'] = mk_repeat_workers_series(idf_base)

    # Read in heirarchical clusters
    cdf =read_cluster_data([idea_cluster_csvs[key] for key in idea_cluster_csvs]) 
    idf = pd.merge(idf_base,
                  cdf,
                  'left', merge_column_names, suffixes=('', '_clus'))


    # ========================================================
    # FILTER DATA
    # ======================================================== 

    if filter_instances:
        idf = filter_instances(idf)

    # hack: filter known bad values
    # junk data, etc
    num_lost = 0
    bad_ideas = [1128295, 1128296, 1179173, 1178933, 1128294, 894122, 174901]
    for i in bad_ideas:
        old_len = len(idf)
        idf = idf[idf['idea'] != i]
        new_len = len(idf)
        num_lost += old_len - new_len

    assert(num_lost > 0)

    print("Dropped", num_lost, "bad ideas")
    
    # ========================================================
    # CLUSTER TOPOLOGY
    # ========================================================

    cluster_forests = {qc: mk_cluster_forest(cluster_tree_csvs[qc]) for qc in cluster_tree_csvs.keys()
            if len(idf[idf['question_code'] == qc]) > 0}


    # ========================================================
    # CLUSTER DATA
    # ========================================================

    dump_csvs(processed_data_folder, idf)
    return idf, cluster_forests

if __name__ == "__main__":
    idf, cfs = do_format_data("/home/fil/enc_projects/crowbrain/processed_data")
    df, rmdf, clusters_df, ann_cfs = mk_redundant(idf, cfs)
    
    print(df)
