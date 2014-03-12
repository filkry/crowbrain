import format_data, modeling
import simulate_error_tree as se
import networkx as nx
import json
import jinja2
import subprocess as sp

def dict_from_node(ifo, n, df):

    children = []
    n_instances = len(df[df['idea'] == n])
    for i in range(n_instances):
        children.append({'name': 'instance', 'size': 1})

    succs = ifo.successors(n)
    if len(succs) > 0:
        for n in succs:
            good, child = dict_from_node(ifo, n, df)
            if good:
                children.append(child)

    return len(children) > 0, {'name': ifo.node[n]['label'], 'children': children}


def dict_from_forest(ifo, df):
    roots = [n for n in ifo.nodes() if len(ifo.predecessors(n)) == 0]
    children = []
    for n in roots:
        good, c = dict_from_node(ifo, n, df)
        if good:
            children.append(c)

    return {'name': 'root',
            'children': children}

def test_forest_dict(d):
    if 'children' in d:
        assert(len(d['children']) > 0)
        for c in d['children']:
            test_forest_dict(c)
    else:
        assert(d['size'] == 1)

def filter_today(df):
    #df = df[(df['question_code'] == 'iPod') | (df['question_code'] == 'turk')]
    df = format_data.filter_repeats(df)
    #df = filter_match_data_size(df)
    return df
 
def render_forest(idea_forest, idf, output_file, temp_directory):
    # dump json representation
    with open('%s/temp.json' % temp_directory, 'w') as f:
        d = dict_from_forest(idea_forest, idf)
        test_forest_dict(d)
        json.dump(d, f)

    # generate html file
    env = jinja2.Environment(loader=jinja2.FileSystemLoader('templates'))
    tpl = env.get_template('circle_packing_template.html')
    with open('%s/circle_packing_temp.html' % temp_directory, 'w') as f:
        print(tpl.render({'json_file': 'temp.json'}), file=f)

    # spawn mongoose
    mongoose = sp.Popen(["mongoose"], cwd=temp_directory)

    # render the page with PhantomJs
    tpl = env.get_template('phantom_render.js')
    url = 'http://127.0.0.1:8080/circle_packing_temp.html'
    render_script = '%s/render_temp.js' % temp_directory
    with open(render_script, 'w') as f:
        print(tpl.render({'url': url, 'output_file': output_file}), file=f)

    ret = sp.call(['phantomjs', '%s/render_temp.js' % temp_directory])

    # kill mongoose
    mongoose.kill()


if __name__ == '__main__':
    processed_data_folder = '/home/fil/enc_projects/crowbrain/processed_data'
    idf, cfs = format_data.do_format_data(processed_data_folder, filter_today)
    df, rmdf, cdf, cfs = modeling.get_redundant_data(cfs, idf)

    # Dump json representation for each tree
    for qc in cfs.keys():
        render_forest(cfs[qc], df, 'figures/%s_forest_viz.png' % qc, 'viz')

    iPod_forest = cfs['iPod']
    for i in range(3):
        se_idf, se_ifo = se.simulate_error_node('iPod', idf, iPod_forest)

        render_forest(se_ifo, se_idf, 'figures/iPod_forest_viz_se%i.png' % i, 'viz')


