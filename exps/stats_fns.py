import subprocess
import numpy as np

pystan_model_cache = dict()

_hdi_stats_cache = {}
# Use this one instead
def get_hdi(a, b, confidence_interval):
  if (a,b,confidence_interval) in _hdi_stats_cache:
    return _hdi_stats_cache[(a,b,confidence_interval)]
  #print "get_hdi:", a, b, confidence_interval
  if a == 1 or b == 1:
    if a == 1 and b == 1:
      lower_bound = (1.0-confidence_interval) / 2
      upper_bound = 1.0 - lower_bound
    else:
      #print "special qbeta:", a, b, confidence_interval
      results = subprocess.check_output(["R", "-q", "-e", 'qbeta(%f, 1, %d)' % (confidence_interval, max(a,b))])
      results = str(results)
      results = results.split('\\n')[1]
      upper_bound = float(results[results.index(' ')+1:])
      if a < b:
        lower_bound = 0
      else:
        lower_bound = 1.0 - upper_bound
        upper_bound = 1.0
  else:
    results = subprocess.check_output(["R", "-q", "-e", 
        'get_hdi = function(a, b, level=0.95) { density_diff = function(lower, a, b) { p_lower = pbeta(lower, a, b); p_upper = min(1.0, p_lower + level); upper = qbeta(p_upper, a, b); return(dbeta(lower, a, b) - dbeta(upper, a, b)); }; lower = uniroot(density_diff, c(0, qbeta(1.0-level, a, b)), a=a, b=b)$root; upper = qbeta(pbeta(lower, a, b) + level, a, b); return(c(lower, upper)); }; get_hdi(%d, %d, %f)' % (a, b, confidence_interval)])
    results = str(results)
    results = results.split('\\n')[1]
    results = results[results.index(' ')+1:]
    lower_bound, upper_bound = [float(x) for x in results.split(' ')]
  _hdi_stats_cache[(a,b,confidence_interval)] = (lower_bound, upper_bound)
  return lower_bound, upper_bound

def beta_bernoulli_posterior(num_successes, total_num):
  a_prior = 1
  b_prior = 1
  a_post = a_prior + num_successes
  b_post = b_prior + (total_num - num_successes)
  posterior_mean = a_post / float(total_num + a_prior + b_prior)
  posterior_variance = posterior_mean * (1.0-posterior_mean) / (1.0 + a_prior + b_prior + total_num)
#  old_lower_bound, old_upper_bound = run_qbeta(a_post, b_post, 0.95)
  lower_bound, upper_bound = get_hdi(a_post, b_post, 0.95)
  return posterior_mean, lower_bound, upper_bound#, old_lower_bound, old_upper_bound

def mean_and_hpd(data, level):
    return (np.mean(data),) + hpd(data, level)

# HPD calculation from biopy, copied under GPL. Using this purely
# because it is an existing implementation of a credible interval,
# and I don't know why I would choose HDI, etc over this
def hpd(data, level) :
  """ The Highest Posterior Density (credible) interval of data at level level.

  :param data: sequence of real values
  :param level: (0 < level < 1)
  """ 
  
  d = list(data)
  d.sort()

  nData = len(data)
  nIn = int(round(level * nData))
  if nIn < 2 :
    raise RuntimeError("not enough data")
  
  i = 0
  r = d[i+nIn-1] - d[i]
  for k in range(len(d) - (nIn - 1)) :
    rk = d[k+nIn-1] - d[k]
    if rk < r :
      r = rk
      i = k

  assert 0 <= i <= i+nIn-1 < len(d)
  
  return (d[i], d[i+nIn-1])

def stan_hyp_test(dats, model_string, testfunc, df = None):
    n_chain = 2 if pystan_test_mode else 3 
    n_saved_steps = 1000 if pystan_test_mode else 10000
    
    fits = []
    for i, dat in enumerate(dats):
        #print(dat)
        if True:
            try:
                if model_string not in pystan_model_cache:
                    pystan_model_cache[model_string] = pystan.StanModel(model_code=model_string)
                
                model = pystan_model_cache[model_string]
                fit = model.sampling(data=dat,
                              iter=math.ceil(n_saved_steps/n_chain), chains=n_chain)
                # stop thrashing hard drive
                #pystan_fit_cache[ck] = fit
                fits.append(fit)
            except:
                print("Exception")
                dat2 = {'N': dat['N'],
                        'y': [float(y) for y in dat['y']]}
                with open('ipython_output/exception_dat.json', 'w') as f:
                    f.write(json.dumps(dat2))
                with open('ipython_output/exception_model_string.json', 'w') as f:
                    f.write(model_string)
                if not df is None:
                    with open('ipython_output/exception_ideas.json', 'w') as f:
                        f.write(json.dumps([int(i) for i in df['idea']]))
                    with open('ipython_output/exception_cats.json', 'w') as f:
                        f.write(json.dumps([int(i) for i in df['subtree_root']]))
                raise
        else:
            fits.append(pystan_fit_cache[ck])
        
    success = testfunc(fits) if len(fits) == len(dats) else '%i,%i' % (len(fits), len(dats)) 
    return fits, success


