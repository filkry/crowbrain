import matplotlib.pyplot as plt
import hashlib, pystan, os, pickle
import simulate_error_tree as se
import networkx as nx
import numpy as np
import format_data

def plot_convergence(la, name, param_index):
    print("Examine convergence for parameter %s" % name)
    dat = la[:,:,param_index]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    assert(not np.isnan(np.sum(dat)))
    ax.hist(dat)
    ax.set_title(name)
    plt.show()

def read_or_gen_cache(file_name, gen_fn):
    full_file_name = 'cache/%s' % file_name
    if os.path.isfile(full_file_name):
        return pickle.load(open(full_file_name, 'rb'))
    else:
        dat = gen_fn()
        pickle.dump(dat, open(full_file_name, 'wb'))
        return dat

def fit_and_extract(model, dat, iter, chains, init_params):
    if init_params is None:
        fit = model.sampling(data=dat, iter=iter, chains=chains)
    else:
        fit = model.sampling(data=dat, iter=iter, chains=chains,
                init=(lambda: init_params))

    las = (fit.extract(permuted=True), fit.extract(permuted=False))

    for name, index in zip(fit.model_pars, range(las[1].shape[2])):
        plot_convergence(las[1], name, index)

    return las

def hash_string(s):
    return hashlib.sha224(s.encode('utf-8')).hexdigest()

# These functions are hacks to avoid fixing an arcane error
def hash_dict(d):
    dict_str = ''
    for key in sorted(d):
        val = d[key]
        if isinstance(val, int):
            dict_str += str(val)
        elif isinstance(val, float):
            dict_str += str(val)
        else:
            dict_str += str(frozenset(val))

    return hash_string(dict_str)

def hash_idea_forests(fs):
    # TODO: assure this is deterministic
    fs_nodes = [str(n) for key in sorted(fs)
                       for n in nx.topological_sort(fs[key])]
    return hash_string(''.join(fs_nodes))

def hash_instance_df(df):
    # TODO: assure this is deterministic
    return hash_string(''.join(a for a in df['answer']))

def compile_and_fit(model_string, dat, n_iter, n_chains, init_params = None):
    model = read_or_gen_cache("%s.stanmodel" % hash_string(model_string),
        lambda: pystan.StanModel(model_code=model_string))

    if init_params is not None:
        pw_cache = "%s_%s_%s_%i_%i.stanfit" %\
            (hash_string(model_string), hash_dict(dat), hash_dict(init_params), n_iter, n_chains)
    else: 
        pw_cache = "%s_%s_%i_%i.stanfit" %\
            (hash_string(model_string), hash_dict(dat), n_iter, n_chains)

    param_walks = read_or_gen_cache(pw_cache,
        lambda: fit_and_extract(model, dat, n_iter, n_chains, init_params))

    return param_walks

def get_redundant_data(idea_forests, instance_df):
    fn = "%s_%s.redundant_data" %\
            (hash_idea_forests(idea_forests), hash_instance_df(instance_df))

    return read_or_gen_cache(fn,
        lambda: format_data.mk_redundant(instance_df, idea_forests))

def get_simulated_error_forests(idea_forests, instance_df, index):
    fn = "%s_%s_%i.simulated_error_redundant" %\
            (hash_idea_forests(idea_forests), hash_instance_df(instance_df), index)

    return read_or_gen_cache(fn,
        lambda: se.gen_sym_tree_data(instance_df, idea_forests))

def simulate_error_hypothesis(n_tests, model_string, n_iter, n_chains, dat_fn, hyp_fn,
        idea_forests, instance_df):

    successes = 0
    for i in range(n_tests):
        edf, ermdf, eclusters_df, eidea_forests = get_simulated_error_forests(idea_forests, instance_df, i)
        dat = dat_fn(edf, ermdf, eclusters_df, eidea_forests)
        param_walks = compile_and_fit(model_string, dat, n_iter, n_chains)
        successes += hyp_fn(param_walks[0])

    return successes

