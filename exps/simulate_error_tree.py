import pandas as pd
import numpy as np
import csv, os
import networkx as nx
import random
import math
import matplotlib.pyplot as plt
import json
import format_data
import stats_fns as mystats
import modeling

from collections import defaultdict, OrderedDict

def all_pairings(s):
    return ((s1, s2) for i, s1 in enumerate(s[:-1])
                     for s2 in s[i+1:])

def defaultdict_to_dict(dd):
    out = dict()
    for k in dd:
        out[k] = dd[k]
    return out
import pystan

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

def validity_test_results(question_code):
    """
    Load the results of the validity judging task
    """
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    survey_results = "%s/validity_survey_%s/data.csv" % (processed_data_folder, question_code)
    with open(survey_results, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        all_results = [row for row in reader if row[0] != 'judge' and row[5] != '?']
        
        return all_results
    
def cull_repeat_pairings(results):
    """
    Go through judging results and yield only the first judgment of each
    node pairing
    """
    seen = set()
    for res in results:
        judge, n1id, n2id, bin1, bin2, rel, _, _, _, _ = res
        if (n1id, n2id) not in seen:
            seen.add((n1id, n2id))
            yield int(judge), int(n1id), int(n2id), int(bin1), int(bin2), int(rel)

def get_node_relationship(n1id, n2id, cluster_forest):
    f = cluster_forest

    n1 = f.node[n1id]
    n2 = f.node[n2id]
    
    if n1['subtree_root'] != n2['subtree_root']:
        return 3
    elif n1id in nx.ancestors(f, n2id):
        return 1
    elif n2id in nx.ancestors(f, n1id):
        return 2
    else:
        return 4

def test_violation(guess, judge_rel, violation_type = "all"):
    """
    Test if any or a specific violation has occured based on idea forest guess
    and oracle result (judge)
    """
    fail = guess != int(judge_rel)
    if violation_type == 'all':
        return fail
    elif violation_type == 'parent_child':
        return fail and (judge_rel == 1 or judge_rel == 2 or guess == 1 or guess == 2)
    elif violation_type == 'artificial_parent':
        return fail and (judge_rel == 4 or guess == 4)
    elif violation_type == 'single_node_per_idea':
        return fail and (judge_rel == 5)
    
def gen_grid_bernoullis(results, cluster_forest, violation_type = "all"):
    """
    Generate a grid of bernoulli variables for constraint violation probability
    based on number of violations

    Returns a dictionary keyed on a 2-tuple of bin numbers
    """
    grid = defaultdict(list)
    bern_grid = dict()
    
    for judge, n1id, n2id, bin1, bin2, rel in results:
        guess = get_node_relationship(n1id, n2id, cluster_forest)
        grid[(bin1, bin2)].append(int(test_violation(guess, rel, violation_type)))
    
    for k in grid.keys():
        trials = grid[k]
        p = mystats.beta_bernoulli_posterior(sum(trials), len(trials))
        bern_grid[k] = (p[0], p[1], p[2], sum(trials) / len(trials))
        
    return bern_grid

def bern_grid_sym(bg, key1, key2):
    if (key1, key2) in bg:
        return bg[(key1, key2)]
    else:
        return bg[(key2, key1)]

def plot_bern_grid(bern_grid, title):
    # mean is 0, empirical mean is 3
    which_mean = 0

    X, Y, Z = zip(*[(k[0], k[1], bern_grid[k][which_mean]) for k in bern_grid])
    
    # TODO: consider whether this arange assumes a number of bins
    X2d, Y2d = np.meshgrid(np.arange(0, 5, 1), np.arange(0, 5, 1))
    Z2d = np.array([bern_grid_sym(bern_grid, x, y)[which_mean] for x, y in zip(np.ravel(X2d), np.ravel(Y2d))])
    Z2d = Z2d.reshape(X2d.shape)
    
    plt.ion()
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.invert_xaxis()
    ax.plot_surface(X2d, Y2d, Z2d, shade=True, rstride=1, cstride=1)
    ax.set_zlim(0, 1)
    ax.set_xlabel("bin 1")
    ax.set_ylabel("bin 2")
    ax.set_zlabel("error proportion")
    ax.set_title(title)
    
    # headmap
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    A = np.zeros((5,5))
    for x, y, z in zip(X, Y, Z):
        A[x, y] = z
        A[y, x] = z
    
    ax.imshow(A, interpolation='bilinear')
    ax.invert_yaxis()
    ax.set_xlabel("bin 1")
    ax.set_ylabel("bin 2")
    ax.set_title(title)

def get_node_bin(bins, node):
    for i, binn in enumerate(bins):
        if node in binn:
            return i
    return "ERROR" + str(node)

def chain_between(forest, ancestor, descendent):
    """
    Return the chain of nodes between an ancestor and its descendent, excluding
    the ancestor itself (but including the descendent)
    """
    cur = descendent
    chain = [cur]
    while(cur != ancestor and forest.predecessors(cur)[0] != ancestor):
        cur = forest.predecessors(cur)[0]
        chain.append(cur)
    return chain

def get_root(forest, node):
    cur = node
    while(len(forest.predecessors(cur)) > 0):
        cur = forest.predecessors(cur)[0]
    return cur

"""
The following functions "repairs" errors into the forest they are passed. Each
performs the operation that would be necessary to repair the corresponding
error
"""

def introduce_parent_child_error(new_forest, n1, n2):
    # if there is already a parent-child relationship
    if n1 in nx.descendants(new_forest, n2) or n2 in nx.descendants(new_forest, n1):
        anc, desc = (n1, n2) if n2 in nx.descendants(new_forest, n1) else (n2, n1)
        chain = chain_between(new_forest, anc, desc)
        bad_link = random.sample(chain, 1)[0]
        new_forest.remove_edge(new_forest.predecessors(bad_link)[0], bad_link)
    else:
        pa, chi = (n1, n2) if random.random() > 0.5 else (n2, n1)
        remove_parent_edge(new_forest, chi)
        new_forest.add_edge(pa, chi)
    assert(nx.is_directed_acyclic_graph(new_forest))

def remove_parent_edge(forest, node):
    preds = forest.predecessors(node)
    if len(preds) > 0:
        forest.remove_edge(preds[0], node)

def introduce_artificial_error(new_forest, n1, n2):
    n1root = get_root(new_forest, n1)
    if n2 in nx.descendants(new_forest, n1root):
        err_node = n1 if n2 == n1root else (n2 if n1 == n1root or random.random() > 0.5 else n1)
        chain = chain_between(new_forest, n1root, err_node)
        bad_link = random.sample(chain, 1)[0]
        new_forest.remove_edge(new_forest.predecessors(bad_link)[0], bad_link)
    else: # if no artificial node, introduce one
        n2root = get_root(new_forest, n2)
        assert(n1root != n2root)
        new_node = max(new_forest.nodes()) + 1
        new_forest.add_node(new_node)
        new_forest.node[new_node]['label'] = 'artificial node (simulated error)'
        new_forest.add_edge(new_node, n1root)
        new_forest.add_edge(new_node, n2root)
    assert(nx.is_directed_acyclic_graph(new_forest))
        
def introduce_single_node_error(new_forest, instance_df, n1, n2):
    keep, lose = (n1, n2) if random.random() > 0.5 else (n2, n1)
    keep, lose = (lose, keep) if lose in nx.ancestors(new_forest, keep) else (keep, lose)
    
    for n in new_forest.successors(lose):
        new_forest.add_edge(keep, n)
    new_forest.remove_node(lose)
    
    new_idea = instance_df['idea']

    for idx in new_idea.index:
        if new_idea[idx] == lose:
            new_idea[idx] = keep

    instance_df['idea'] = new_idea
    
    assert(nx.is_directed_acyclic_graph(new_forest))
    return lose

def one_normalize(alist):
    total = sum(alist)
    return [float(v) / total for v in alist]


# This is really ugly, I should use more generalizable distribution/
# marginalization code, but fast > good sigh
def bern_grid_marginalize_bin2(bern_grid, bins):
    """
    Given a 2D grid of bernoullis parameters where each dimension is the same,
    marginalizes out one dimension.

    Returns a dict of bernoulli parameters
    """
    total_len_bins = sum(len(b) for b in bins)
    p_bins = [float(len(b)) / total_len_bins for b in bins]

    new_bern = dict()

    bin1s = sorted(list(set(b1 for b1, b2 in bern_grid.keys())))
    bin2s = sorted(list(set(b2 for b1, b2 in bern_grid.keys())))

    for bin1 in bin1s:
        sum_parts = []
        for bin2 in bin2s:
            bern_mean = bern_grid_sym(bern_grid, bin1, bin2)[0]
            sum_parts.append(bern_mean * p_bins[bin2])
        new_bern[bin1] = (sum(sum_parts), one_normalize(sum_parts))

    return new_bern

def get_error_node2(err_bern, n1bin, bins, exclude_nodes):
    """
    For a constraint violation in the relationship between two nodes, select
    a second node given the current node
    """
    x = random.random()
    for binno, rate in enumerate(err_bern[n1bin][1]):
        if x <= rate:
            sample_set = set(bins[binno]).difference(set(exclude_nodes))
            #assert(len(sample_set) > 0)
            if(len(sample_set) < 1):
                continue
            n2 = random.sample(sample_set, 1)[0]
            return n2

    return None

def gen_err_berns(question_code, cluster_forest):
    all_results = validity_test_results(question_code)
    culled = list(cull_repeat_pairings(all_results))

    return [gen_grid_bernoullis(culled, cluster_forest, 'parent_child'),
        gen_grid_bernoullis(culled, cluster_forest, 'artificial_parent'),
        gen_grid_bernoullis(culled, cluster_forest, 'single_node_per_idea')]

def redundant_subset(qc, instance_df, cluster_forest):
    qcdf = instance_df[instance_df['question_code'] == qc]
    nodes = [(idea, len(qcdf[qcdf['idea'] == idea]))
            for idea in set(qcdf['idea'])]

    return nodes

def simulate_error_node(qc, instance_df, ann_cluster_forest):
    # Generate a subset of necesary redundant data
    nodes = redundant_subset(qc, instance_df, ann_cluster_forest)

    bins = bin_sequence(nodes, lambda x: x[1], 5)
    bins = [[n[0] for n in binn] for binn in bins]
    err_berns = gen_err_berns(qc, ann_cluster_forest)
    err_berns = [bern_grid_marginalize_bin2(eb, bins) for eb in err_berns]

    new_forest = ann_cluster_forest.copy()
    new_idf = instance_df.copy()
    
    real_nodes = [n for n in ann_cluster_forest.nodes() if len(instance_df[instance_df['idea'] == n]) > 0]

    lost_nodes = []
    for j, n1 in enumerate(real_nodes):
        print("Simulating error on forest %i/%i" % (j, len(real_nodes)), end='\r')
        
        if n1 in lost_nodes:
            continue
        
        n1bin = get_node_bin(bins, n1)
        
        err_rates = [eb[n1bin][0] for eb in err_berns]
        x = random.random()

        actual_err = None
        n2 = None
        for i, (err, rate) in enumerate(zip(err_berns, err_rates)):
            if x <= rate:
                n2 = get_error_node2(err, n1bin, bins, lost_nodes + [n1])
                if n2 is not None: # Can't be error if no valid node
                    actual_err = i
                break
            x -= rate
            
        if actual_err == 0:
            introduce_parent_child_error(new_forest, n1, n2)
            last = "Parent"
        elif actual_err == 1:
            introduce_artificial_error(new_forest, n1, n2)
            last = "artificial"
        elif actual_err == 2:
            lost_nodes.append(introduce_single_node_error(new_forest, new_idf, n1, n2))
            last = 'single'

    print("Finished simulating error. %i nodes removed." % len(lost_nodes))
    assert(len(set(new_idf['idea'])) == len(set(instance_df['idea'])) - len(lost_nodes))
    assert(nx.is_directed_acyclic_graph(new_forest))
    return new_idf, new_forest

def gen_sym_tree_data(instance_df, idea_forests):
    new_forests = dict()
    new_idf = instance_df
    for key in idea_forests:
        f = idea_forests[key]
        edf, ef = simulate_error_node(key, new_idf, f)
        new_idf = edf
        new_forests[key] = ef

    return format_data.mk_redundant(new_idf, new_forests)


def table_error_rates(idea_forests):
    res = ['\\begin{table}', '\\centering', '\\begin{tabular}{| r l l | l l |}']
            
    qc_vios = dict()
    qc_judges = dict()
    qc_pairs = dict()

    for qc in idea_forests:
        ifo = idea_forests[qc]

        all_results = validity_test_results(qc)
        culled = list(cull_repeat_pairings(all_results))

        equivalence_violations = 0
        generalization_violations = 0
        common_parent_violations = 0
        non_equivalence_violations = 0
        total = 0
        judge_0_count = 0
        judges = set()

        for judge, n1id, n2id, bin1, bin2, rel in culled:
            judges.add(judge)
            judge_0_count += int(judge == 0)
            total += 1
            guess = get_node_relationship(n1id, n2id, ifo)
            #equivalence_violations += int(test_violation(guess, rel, 'TODO'))
            generalization_violations  += int(test_violation(guess, rel, 'parent_child'))
            common_parent_violations += int(test_violation(guess, rel, 'artificial_parent'))
            non_equivalence_violations += int(test_violation(guess, rel, 'single_node_per_idea'))

        eq_hpd = mystats.beta_bernoulli_posterior(equivalence_violations, total)
        gen_hpd = mystats.beta_bernoulli_posterior(generalization_violations, total)
        cp_hpd = mystats.beta_bernoulli_posterior(common_parent_violations, total)
        ne_hpd = mystats.beta_bernoulli_posterior(non_equivalence_violations, total)

        qc_vios[qc] = (eq_hpd, gen_hpd, cp_hpd, ne_hpd)
        qc_judges[qc] = len(judges)
        qc_pairs[qc] = judge_0_count

    res.extend(['\\hline \\textbf{constraint} & \\textbf{judges} & \\textbf{pairs} & equivalence & generalization \\\\ \\hline',])

    for qc in idea_forests.keys():
        eq_hpd, gen_hpd, cp_hpd, ne_hpd = qc_vios[qc]
        res.extend(['%s & %i & %i & %0.2f (%0.2f, %0.2f) & %0.2f (%0.2f, %0.2f) \\\\' % (qc,
            qc_judges[qc], qc_pairs[qc], eq_hpd[0], eq_hpd[1], eq_hpd[2], 
            gen_hpd[0], gen_hpd[1], gen_hpd[2])])

    res.extend(['\\hline \\textbf{constraint} & \\textbf{judges} & \\textbf{pairs} & common parent & non-equivalence \\\\ \\hline',])

    for qc in idea_forests.keys():
        eq_hpd, gen_hpd, cp_hpd, ne_hpd = qc_vios[qc]
        res.extend(['%s & %i & %i & %0.2f (%0.2f, %0.2f) & %0.2f (%0.2f, %0.2f) \\\\' % (qc,
            qc_judges[qc], qc_pairs[qc], cp_hpd[0], cp_hpd[1], cp_hpd[2], 
            ne_hpd[0], ne_hpd[1], ne_hpd[2])])

    # CONTINUE FROM HERE: need end of table in res
    res.extend(['\\hline', '\\end{tabular}', '\\caption{Idea forest error rates}',
        '\\label{tab:judge_results}', '\\end{table}'])

    with open('tex/error_rates_table.tex', 'w') as f:
        print('\n'.join(res), file=f)

def filter_today(df):
    df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    table_error_rates(cfs)

