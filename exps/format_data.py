import pandas as pd
import csv, os
import re
import networkx as nx
import numpy as np
from numpy import uint8, uint16, uint32, uint64, datetime64, int64, int32, float64
from collections import defaultdict, OrderedDict

def metrics_folder(pd_folder, x):
        return '/%s/pilot18_metrics/%s' % (pd_folder, x)

def read_files(fils):
        rows = []
        for f in fils:
            rows = rows + read_file(f)
        return rows

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
    return pd.concat(dfs)

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
        while(len(f.predecessors(cur)) > 0):
            cur = f.predecessors(cur)[0]
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


def mk_run_df(instance_df, agg_dict, column_names):
    runs = instance_df.groupby(['num_requested', 'worker_id',
        'question_code', 'submit_datetime', 'accept_datetime'],
        as_index=False)

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
        assert (nr >= len(run))
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
                         ['iPod', 'turk']}
                         #['charity', 'iPod', 'forgot_name', 'turk']}
    cluster_tree_csvs = {qc: metrics_folder(processed_data_folder, "_%s_clusters.csv" % qc) for qc in \
                         ['iPod', 'turk']}
                         #['charity', 'iPod', 'forgot_name', 'turk']}
        
    
    merge_column_names = ['worker_id', 'question_code', 'answer_num', 'num_requested']
 
    # ========================================================
    # INSTANCE DATA
    # ========================================================

       
    # Read in the main processed data files
    idf_base = read_base_data(base_data_dirs)

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
