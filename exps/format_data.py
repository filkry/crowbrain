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
            dfs.append(pd.read_csv(full_name, sep=',', quotechar='|', quoting=csv.QUOTE_NONNUMERIC, parse_dates=[12, 13]))
    return pd.concat(dfs)

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

def clean_missing(v):
    return 0 if v == '' or v == 'missing' else v

def series_from_row(rows, index, t):      
    if t == object:
        return pd.Series([row[index] for row in rows], dtype=t)
    elif t == datetime64:
        return pd.Series([np.datetime64(row[index]) for row in rows], dtype=t)
    else:
        s = pd.Series([clean_missing(row[index]) for row in rows], dtype=t)
        return s

def read_manual_csv(name, manual_csvs):
    index = 0 if name == 'fil' else 1
    rows = read_file(manual_csvs[index])
    
    df = pd.DataFrame({'worker_id': series_from_row(rows, 0, object),
            'question_code': series_from_row(rows, 2, object),
            'num_requested': series_from_row(rows, 5, uint8),
            'answer_num': series_from_row(rows, 6, uint8),
            ('utility_%s' % name): series_from_row(rows, 13, uint8),
            ('realistic_%s' % name): series_from_row(rows, 14, uint8),
            ('distance_%s' % name): series_from_row(rows, 16, uint8),
            ('valid_%s' % name): pd.Series([1 for row in rows], dtype=uint8),
    })
    return df

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

def cluster_forest(structure_csv):
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

def num_instances_in(df, clusters):
    return sum(len(df[df['idea'] == c]) for c in clusters)

def run_count(run, pass_func):
    count = 0
    for i in run.iterrows():
        row = i[1]
        if pass_func(row):
            count += 1
    return count

def run_mean(run, value_func):
    t = 0.0
    for i in run.iterrows():
        row = i[1]
        t += value_func(row)
    return t / len(run)

def mk_cluster_df(instance_df, cluster_forests):
    # Put cluster metrics in their own dataframe
    ids = []
    labels = []
    is_roots = []
    is_leafs = []
    roots = []
    num_nodes_under = []
    subtree_probabilitys = []
    depths = []
    heights = []
    qcs = []
    num_childrens = []
    num_instances_under = []
    subtree_probability = []
    idea_probability = []
    num_instances = []
    num_workers = []
    num_ideas = []
    parents = []

    #print qc_conds

    for qc in qc_conds:
        sub_df = instance_df[instance_df['question_code'] == qc]
        if len(sub_df) == 0:
            continue

        #print len(sub_df)
        #print sum(sub_df['is_repeat_worker'])
        
        if qc not in cluster_forests:
            #print "Skipping", qc
            continue
        
        f = cluster_forests[qc]
        for n in f.nodes():
            nd = f.node[n]
            
            idea = n
            
            ids.append(n)
            if len(f.predecessors(n)) > 0:
                parents.append(f.predecessors(n)[0])
            else:
                parents.append(-1)
            labels.append(nd['label'])
            is_roots.append(n == nd['subtree_root'])
            is_leafs.append(len(f.successors(n)) == 0)
            num_childrens.append(len(f.successors(n)))
            roots.append(nd['subtree_root'])

            depths.append(nd['depth'])
            heights.append(nd['height'])
            qcs.append(qc)
            num_nodes_under.append(len(nd['all_nodes_under']))
            
            # Metrics for entire dataset; see time-based below
            num_instances_under.append(num_instances_in(sub_df, all_nodes_under(f, idea)))
            
            root_idea = f.node[idea]['subtree_root']
            nus = num_instances_in(sub_df, all_nodes_under(f, root_idea))
            subtree_probability.append(float(nus) / len(sub_df))
            
            nii = num_instances_in(sub_df,[idea])
            idea_probability.append(float(nii) / len(sub_df))
            num_ideas.append(nii)
            
            instance_df = sub_df[sub_df['idea'] == idea]
            num_instances.append(len(instance_df))
            num_workers.append(len(set(instance_df['worker_id'])))

    #print len(ids), len(roots)
            
    clusters_df = pd.DataFrame({
            'idea': pd.Series(ids, dtype=uint64),
            'parent': pd.Series(parents, dtype=int64),
            'idea_label': pd.Series(labels, dtype=object),
            'is_root': pd.Series(is_roots, dtype=uint8),
            'is_leaf': pd.Series(is_leafs, dtype=uint8),
            'subtree_root': pd.Series(roots, dtype=uint64),
            'depth_in_subtree': pd.Series(depths, dtype=uint32),
            'height_in_subtree': pd.Series(heights, dtype=uint32),
            'question_code': pd.Series(qcs, dtype=object),
            'num_nodes_under': pd.Series(num_nodes_under, dtype=uint64),
            'num_children': pd.Series(num_childrens, dtype=uint64),
            'num_instances_under': pd.Series(num_instances_under, dtype=uint64),
            'subtree_probability': pd.Series(subtree_probability, dtype=float64),
            'idea_probability': pd.Series(idea_probability, dtype=float64),
            'num_instances': pd.Series(num_instances, dtype=uint64),
            'num_workers': pd.Series(num_workers, dtype=uint64),
            'num_ideas_under': pd.Series(num_ideas, dtype=uint64),
        })

    clusters_df['subtree_oscore'] = 1 - clusters_df['subtree_probability']
    clusters_df['idea_oscore'] = 1 - clusters_df['idea_probability']

    return clusters_df


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

def mk_run_df(instance_df):
    runs = instance_df.groupby(['num_requested', 'worker_id',
        'question_code', 'submit_datetime', 'accept_datetime',
        'run_id'])

    wids = pd.Series([wid for ((nr, wid, qc, sdt, adt), run) in runs], dtype=object)
    qcs = pd.Series([qc for ((nr, wid, qc, sdt, adt), run) in runs], dtype=object)
    nrs = pd.Series([nr for ((nr, wid, qc, sdt, adt), run) in runs], dtype=float64) # make float for normalization purposes
    nrc = pd.Series([len(run) for (name, run) in runs], dtype=float64) # make float for normalization purposes

    # TODO: generate run IDs in the instance DF
    rids = pd.Series([None for i in instance_df.index], index=instance_df.index)
    for i, (name, run) in enumerate(runs):
        for j in run.index:
            rids[j] = i
    df['run_id'] = rids

    # Test worked
    for i in nrc.index:
        run_df = df[df['run_id'] == i]
        assert(nrc[i] == len(run_df))
            
    for i in nrs.index:
        assert(nrs[i] >= nrc[i])

    adts = pd.Series(pd.to_datetime([adt for (nr, wid, qc, sdt, adt), run in runs]))
    sdts = pd.Series(pd.to_datetime([sdt for (nr, wid, qc, sdt, adt), run in runs]))

    mwc_val = lambda x: x['word_count']
    mwc = pd.Series([run_mean(run, mwc_val) for (name, run) in runs],
                    dtype=float64)

    nu = pd.Series([len(set(run['idea'])) for name, run in runs],
                    dtype=uint16)

    irw_l = []
    seen = set()
    for (nr, wid, qc, sdt, adt), run in runs:
        if run['is_repeat_worker'].iloc[0] == 1:
            irw_l.append(1)
        else:
            irw_l.append(0)
                
    irw = pd.Series(irw_l, dtype=uint8)

    #print sum(irw)

    nus = pd.Series([len(set(run['subtree_root'])) for (name, run) in runs], dtype=uint16)

    ni = pd.Series([sum(run['is_inmix']) for name, run in runs], dtype=float64)

    no = pd.Series([sum(run['is_outmix']) for name, run in runs], dtype=float64)

    cv_test = lambda run: len(run) == len(run[(run['valid_cluster'] > 0)])
    cv = pd.Series([cv_test(run) for name, run in runs], dtype=uint8)

    tv_test = lambda run: len(run) == len(run[(run['valid_time'] > 0)])
    tv = pd.Series([tv_test(run) for name, run in runs], dtype=uint8)

    rmdf = pd.DataFrame({'worker_id': wids,
                                'question_code': qcs,
                                'num_requested': nrs,
                                'num_received': nrc,
                                'accept_datetime': adts,
                                'submit_datetime': sdts,
                                #'is_repeat_worker': irw,
                                'r_mean_word_count': mwc,
                                #'r_num_unique_ideas': nu,
                                #'r_num_unique_subtrees': nus,
                                #'num_inmix': ni, # same as above
                                #'num_outmix': no, # same as above
                                'r_valid_cluster': cv,
                                'r_valid_time': tv,
                                })

    return rmdf





def do_format_data(processed_data_folder, filter_instances = None):
    """
    This function is pretty gigantic and ugly. It's a copy-paste of about half
    an iPython notebook that was gradually built up to get my data into usable
    form. Now that this part is rarely, if ever, touched, I wanted to get it
    out of the way.

    Takes the folder that holds all the data, and returns instance dataframe,
    cluster dataframe, run dataframe, and cluster forests
    """

    base_data_dirs = [
                  '/%s/pilot18' % processed_data_folder,
                  '/%s/pilot17' % processed_data_folder,
                  '/%s/pilot16' % processed_data_folder,
                  '/%s/pilot14' % processed_data_folder,
                  '/%s/pilot13' % processed_data_folder,
                  '/%s/pilot12' % processed_data_folder,
                  '/%s/pilot11' % processed_data_folder,]


    prefix = '%s/pilot18_metrics/' % processed_data_folder
    manual_csvs = list(map(lambda x: prefix + '%s-scores.csv' % x,
                             ['fil', 'mike']))
    

    idea_cluster_csvs = {qc: metrics_folder(processed_data_folder, "_%s.csv" % qc) for qc in \
                         ['iPod', 'turk']}
                         #['charity', 'iPod', 'forgot_name', 'turk']}
    cluster_tree_csvs = {qc: metrics_folder(processed_data_folder, "_%s_clusters.csv" % qc) for qc in \
                         ['iPod', 'turk']}
                         #['charity', 'iPod', 'forgot_name', 'turk']}
        
    df_output_csv = '/%s/pilot18_notebook_output/instances.csv' % processed_data_folder
    rmdf_output_csv = '/%s/pilot18_notebook_output/runs.csv' % processed_data_folder
    clustersdf_output_csv = '/%s/pilot18_notebook_output/cluster_trees.csv' % processed_data_folder

    # Read in a bunch of base data

    

    merge_column_names = ['worker_id', 'question_code', 'answer_num', 'num_requested']
        
    # Read in the main processed data files
    base_df = read_base_data(base_data_dirs)
    print(base_df)


    num_responses = len(all_rows)

    bad_times = [(0 if (r[9] == 'missing' or r[10] == 'missing' or \
                        int(r[9]) <=0 or int(r[10]) <= 0) else 1) for r in all_rows]

    df_base = pd.DataFrame({'worker_id': pd.Series([row[0] for row in all_rows], dtype=object),
                'question_code': pd.Series([row[2] for row in all_rows], dtype=object),
                'num_requested': pd.Series([row[5] for row in all_rows], dtype=uint8),
                'answer_num': pd.Series([row[6] for row in all_rows], dtype=uint8),
                'answer': pd.Series([row[7] for row in all_rows], dtype=object),
                'word_count': pd.Series([row[8] for row in all_rows], dtype=uint32),
                'submit_datetime': pd.Series(pd.to_datetime([row[12] for row in all_rows])),
                'accept_datetime': pd.Series(pd.to_datetime([row[13] for row in all_rows])),
                'start_time': series_from_row(all_rows, 9, uint64),
                'end_time': series_from_row(all_rows, 10, uint64),
                'valid_time': pd.Series(bad_times, dtype=uint8),
                'batch_file': series_from_row(all_rows, 14, object),
    })

    qc_conds = set(df_base['question_code'])

    #print len(df_base)

    ## Dropping unlimited condition
    #print "With unlimited", len(df_base)

    #print "iPod unlimited"
    #ipodf = df_base[df_base['question_code']=='iPod']
    #print "num instances", len(ipodf)

    #groups = ipodf.groupby(['worker_id', 'submit_datetime'])
    #print "num hits", len(groups)

    #print "num workers", len(set(ipodf['worker_id']))


    # ========================================================
    # INSTANCE DATA
    # ========================================================


    df_base = df_base[~(df_base['num_requested'] == 0)]
    #print "Without unlimited", len(df_base)

    # Check for repeat workers
    is_repeat = pd.Series([0 for i in df_base.index], index=df_base.index)
    df_base = df_base.sort(['submit_datetime'])
    runs = df_base.groupby(['worker_id', 'question_code', 'num_requested', 'submit_datetime'])

    last_sdt = None
    seen_keys = set()
    rids = []

    for rid, ((wid, qc, nr, sdt), run) in enumerate(runs):
        assert (last_sdt is None or sdt >= last_sdt)
        assert (nr >= len(run))
        if (wid, qc) in seen_keys:
            for i in run.index:
                is_repeat[i] = 1
        else:
            seen_keys.add((wid, qc))
        rids.append(rids)
                
    df_base['run_id'] = pd.Series(rids)
    df_base['is_repeat_worker'] = is_repeat


    # Read in heirarchical clusters
    rows = [r for key in idea_cluster_csvs
              for r in read_file(idea_cluster_csvs[key])]
    #print len(rows)
    df = pd.merge(df_base,
                  pd.DataFrame({'worker_id': series_from_row(rows, 5, object),
                      'question_code': series_from_row(rows, 0, object),
                      'num_requested': series_from_row(rows, 7, uint8),
                      'answer_num': series_from_row(rows, 4, uint8),
                      'idea': series_from_row(rows, 2, uint64),
                      'answer': series_from_row(rows, 3, object),
                      'valid_cluster': pd.Series([1 for row in rows], dtype=uint8),}),
                  'left', merge_column_names + ['answer'])


    #print "After clusters:", len(df)

    # Read in manual codes
    df = pd.merge(df, read_manual_csv('mike', manual_csvs), 'left', merge_column_names)
    df = pd.merge(df, read_manual_csv('fil', manual_csvs), 'left', merge_column_names)


    # ========================================================
    # FILTER DATA
    # ======================================================== 
    if filter_instances:
        df = filter_instances(df)
    
    # ========================================================
    # CLUSTER TOPOLOGY
    # ========================================================

    cluster_forests = {qc: cluster_forest(cluster_tree_csvs[qc]) for qc in cluster_tree_csvs.keys()
            if len(df[df['question_code'] == qc]) > 0}


    # ========================================================
    # CLUSTER DATA
    # ========================================================

    clusters_df = mk_cluster_df(df, cluster_forests)

    # Merge idea dataframe with cluster dataframe

    #print "Pre cluster merge data size:", len(df)
    df = pd.merge(df, clusters_df, 'left', ['idea', 'question_code'])
    #print "Post cluster merge data size:", len(df)

    #print df

    #df['time_spent'] = df['end_time'] - df['start_time']

    # Compute outmixing/inmixing

    #clustered_df = df[df['valid_cluster'] == 1]
    #dist, im, om, mm, dist_im, last_sim, related_inmix = compute_mixing(clustered_df, df, cluster_forests)

    #df['distance_from_similar'] = dist
    #df['is_inmix'] = im
    #df['is_midmix'] = mm
    #df['is_outmix'] = om
    #df['distance_from_inmix'] = dist_im
    #df['previous_similar_index'] = last_sim
    #df['inmix_index'] = related_inmix

    # ========================================================
    # RUN DATA
    # ========================================================

    # Generate run-level metrics
    rmdf = mk_run_df(df)

        # Drop redundant data from original dataframe
    df = df.drop('accept_datetime', 1)
    df = df.drop('submit_datetime', 1)
    df = df.drop('worker_id', 1)
    df = df.drop('num_requested', 1)
    df = df.drop('question_code', 1)

       

    

    #print df

    
    # assert(len(runs) == len(run_metrics_df)) # weird memory error

    #df = df.merge(rmdf, right_index=True, left_on=['run_id'])

    #print df
    df.to_csv(df_output_csv)
    rmdf.to_csv(rmdf_output_csv)
    clusters_df.to_csv(clustersdf_output_csv)

    return df, rmdf, clusters_df, cluster_forests

if __name__ == "__main__":
    do_format_data("/home/fil/enc_projects/crowbrain/processed_data")
