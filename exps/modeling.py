import matplotlib.pyplot as plt
import hashlib, pystan, os, pickle

def plot_convergence(la, param_num):
    dat = la[:,:,param_num]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(dat)
    ax.set_title(param_num)
    plt.show()

def read_or_gen_cache(file_name, gen_fn):
    full_file_name = 'cache/%s' % file_name
    if os.path.isfile(full_file_name):
        return pickle.load(open(full_file_name, 'rb'))
    else:
        dat = gen_fn()
        pickle.dump(dat, open(full_file_name, 'wb'))
        return dat

def fit_and_extract(model, dat, iter, chains):
    fit = model.sampling(data=dat, iter=iter, chains=chains)
    las = (fit.extract(permuted=True), fit.extract(permuted=False))

    for i in range(las[1].shape[2]):
        plot_convergence(las[1], i)

    return las

def hash_string(s):
    return hashlib.sha224(s.encode('utf-8')).hexdigest()

def hash_dict(d):
    # This is a hack to avoid fixing an arcane error
    dict_str = ''
    for key in sorted(d):
        val = d[key]
        if isinstance(val, int):
            dict_str += str(val)
        else:
            dict_str += str(frozenset(val))

    return hash_string(dict_str)

def compile_and_fit(model_string, dat, n_iter, n_chains):
    model = read_or_gen_cache("%s.stanmodel" % hash_string(model_string),
        lambda: pystan.StanModel(model_code=model_string))

    pw_cache = "%s_%s_%i_%i.stanfit" %\
        (hash_string(model_string), hash_dict(dat), n_iter, n_chains)
    param_walks = read_or_gen_cache(pw_cache,
        lambda: fit_and_extract(model, dat, n_iter, n_chains))

    return param_walks
