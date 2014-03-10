import format_data, modeling
import simulate_error_tree as se
import networkx as nx
import json

def dict_from_node(ifo, n, df):
    succs = ifo.successors(n)
    if len(succs) == 0:
        size = len(df[df['idea'] == n])
        return size > 0, {'name': ifo.node[n]['label'], 'size': size}
    else:
        children = []
        for n in succs:
            good, child = dict_from_node(ifo, n, df)
            if good:
                children.append(child)

        return len(children) > 0, {'name': ifo.node[n]['label'], 'children': children}


def dict_from_forest(qc, ifo, df):
    roots = [n for n in ifo.nodes() if len(ifo.predecessors(n)) == 0]
    children = []
    for n in roots:
        good, c = dict_from_node(ifo, n, df)
        if good:
            children.append(c)

    return {'name': qc,
            'children': children}

def test_forest_dict(d):
    if 'children' in d:
        assert(len(d['children']) > 0)
        for c in d['children']:
            test_forest_dict(c)
    else:
        assert(d['size'] > 0)

def filter_today(df):
    df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    for qc in cfs.keys():
        with open('json/%s.json' % qc, 'w') as f:
            d = dict_from_forest(qc, cfs[qc], df)
            test_forest_dict(d)
            json.dump(d, f)

    iPod_forest = cfs['iPod']
    for i in range(3):
        se_idf, se_ifo = se.simulate_error_node('iPod', idf, iPod_forest)

        with open('json/iPod_se_%i.json' % i, 'w') as f:
            d = dict_from_forest('iPod', se_ifo, se_idf)
            test_forest_dict(d)
            json.dump(d, f)

