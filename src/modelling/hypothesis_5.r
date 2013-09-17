require(rstan)

do_stan_time_analysis_lognormal = function(df) {
model_string = "
data {
  int N;
  int y[N];
  int num_cond;
  int cond_group[N];
}

parameters {
  real<lower=1> mu_cond[num_cond];
  real<lower=1> sigma_cond[num_cond];
}

model {
  for (i in 1:N) {
    y[i] ~ lognormal(mu_cond[cond_group[i]], sigma_cond[cond_group[i]]);
  }
}
";
  num_conditions = max(df$condition_num)
  data_list = list(
    N = nrow(df),
    y = df$time_spent,
    cond_group = df$condition_num,
    num_cond = num_conditions
  );

  n_chains = 3;
  num_saved_steps = 20000;
  n_iter = ceiling(num_saved_steps / n_chains)
  print(paste("Fitting lognormal model for time spent"))
  fit = stan(model_code = model_string, data = data_list, iter = n_iter, chains = n_chains)

  print(paste("Log model for time spent"))
  print(fit)
}

df = read.csv('time_spent_data.csv', header=TRUE, sep='\t')
df$condition_num = as.numeric(factor(df$follow_same))
results = c()
results = c(results, do_stan_time_analysis_lognormal(df))

