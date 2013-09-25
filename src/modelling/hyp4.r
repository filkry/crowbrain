require(rstan)

kruschke_script_dir = '/home/fil/code/kruschke/samples'
cur_dir = getwd()
setwd(kruschke_script_dir)
source('HDIofMCMC.R')
setwd(cur_dir)

model_string = "
data {
  int N;
  real y[N];
}

parameters {
  real<lower=0> alpha;
  real<lower=0> beta;
}

model {
  for (i in 1:N) {
    y[i] ~ beta(alpha, beta);
  }
}
";
  
fit_model = function(df, split, measure) {
  ldf = subset(df, num_requested <= split)
  rdf = subset(df, num_requested > split)
  
  l_data_list = list(
    N = nrow(ldf),
    y = ldf[,measure]
  );

  n_chains = 3;
  num_saved_steps = 20000;
  n_iter = ceiling(num_saved_steps / n_chains)
  
  print(paste("Fitting for", measure, split, "left"))
  lfit = stan(model_code = model_string, data = l_data_list, iter = n_iter, chains = n_chains)

  r_data_list = list(
    N = nrow(rdf),
    y = rdf[,measure]
  );

  print(paste("Fitting for", measure, split, "right"))
  rfit = stan(model_code = model_string, data = r_data_list, iter = n_iter, chains = n_chains)

  llist = extract(lfit, permuted=TRUE)
  rlist = extract(rfit, permuted=TRUE)

  la_list = llist$alpha
  lb_list = llist$beta

  ra_list = rlist$alpha
  rb_list = rlist$beta

  la_hdi = HDIofMCMC(la_list)
  lb_hdi = HDIofMCMC(lb_list)

  ra_hdi = HDIofMCMC(ra_list)
  rb_hdi = HDIofMCMC(rb_list)

  print(paste("left alpha:", median(la_list), la_hdi))
  print(paste("left beta:", median(lb_list), lb_hdi))
 
  print(paste("right alpha:", median(ra_list), ra_hdi))
  print(paste("right beta:", median(rb_list), rb_hdi))
    
}

df = read.csv('hyp6.csv', header=TRUE, sep='\t')

for (measure in c("tree_oscore", "idea_oscore")) {
  fit_model(df, 20, measure)
}
