require(rstan)

kruschke_script_dir = '/home/mterry/Documents/probability_work/kruschke_scripts'
do_plot_post = function(data) {
  cur_dir = getwd()
  setwd(kruschke_script_dir)
  source('plotPost.R')
  plotPost(data)
  setwd(cur_dir)
}
  
do_stan_cumulative_analysis_log_model = function(df, measure) {
model_string = "
data {
  int N;
  real y[N];
  int x[N];
  int num_cond;
  int cond_group[N];
}

parameters {
  real<lower=0> b_cond[num_cond];
  real<lower=0> x_scale;
  real x_trans;
  real y_trans;
  real<lower=2> log_base;
  real<lower=0> sigma;
}

transformed parameters {
}

model {
  real mu[N];
  for (i in 1:N) {
    mu[i] <- y_trans + b_cond[cond_group[i]] * log(x_scale * x[i] + x_trans) / log(log_base);
     y[i] ~ normal(mu[i], sigma);
  }
}
";
  num_conditions = max(df$condition_num)
  data_list = list(
    N = nrow(df),
    y = df[,measure],
    x = df$response_num+1,
    cond_group = df$condition_num,
    num_cond = num_conditions
  );

  n_chains = 3;
  num_saved_steps = 20000;
  n_iter = ceiling(num_saved_steps / n_chains)
  print(paste("Fitting log model for", measure))
  fit = stan(model_code = model_string, data = data_list, iter = n_iter, chains = n_chains)

  print(paste("Log model for", measure))
  print(fit)
  par(mfrow=c(1,1))
  plot(fit)

  list_of_arrays = extract(fit, permuted=TRUE)
  b0_list = list_of_arrays$b0
  b_cond_list = list_of_arrays$b_cond
  x_scale_list = list_of_arrays$x_scale
  x_trans_list = list_of_arrays$x_trans
  sigma_list = list_of_arrays$sigma

  par(mfrow=c(2,2))
  hist(b0_list, main=paste("b0", measure))
  hist(x_scale_list, main=paste("x_scale", measure))
  hist(x_trans_list, main=paste("x_trans", measure))
  hist(sigma_list, main=paste("sigma", measure))
  b_cond = array(dim=c(num_conditions))
  for (i in 1:num_conditions) {
    print(paste("Stats for b_cond", i))
    print(summary(b_cond_list[,i]))
    print(sd(b_cond_list[,i]))
    hist(b_cond_list[,i], main=paste("b_cond", i, measure))
    do_plot_post(b_cond_list[,i])
    b_cond[i] = mean(b_cond_list[,i])
  }
  b0 = mean(b0_list)
  x_scale = mean(x_scale_list)
  x_trans = mean(x_trans_list)
  print(paste("b0", b0, measure))
  print(paste("x_scale", x_scale, measure))
  print(paste("x_trans", x_trans, measure))
  plot_fn_factory = function(cond_num) {
    plot_fn = function(x) {
      return(b0 + b_cond[cond_num] * log(x_scale * x + x_trans))
    }
    return(plot_fn)
  }
  plot_model(df, measure, plot_fn_factory, "Log Model")
  return(c(fit, plot_fn_factory))
}

do_stan_cumulative_analysis_exp_model = function(df, measure) {
model_string = "
data {
  int N;
  real y[N];
  int x[N];
  int num_cond;
  int cond_group[N];
}

parameters {
  real<lower=0, upper=1> b_cond[num_cond];
  real<lower=0> y_scale;
  real<lower=0> sigma;
}

transformed parameters {
}

model {
  real mu[N];
  for (i in 1:N) {
    mu[i] <- y_scale * pow(x[i], b_cond[cond_group[i]]);
     y[i] ~ normal(mu[i], sigma);
  }
}
";
  num_conditions = max(df$condition_num)
  data_list = list(
    N = nrow(df),
    y = df[,measure],
    x = df$response_num+1,
    cond_group = df$condition_num,
    num_cond = num_conditions
  );

  n_chains = 3;
  num_saved_steps = 20000;
  n_iter = ceiling(num_saved_steps / n_chains)
  print(paste("Fitting exp model for", measure))
  fit = stan(model_code = model_string, data = data_list, iter = n_iter, chains = n_chains)

  print(paste("Exponential model for", measure))
  print(fit)
  par(mfrow=c(1,1))
  plot(fit)

  list_of_arrays = extract(fit, permuted=TRUE)
  b_cond_list = list_of_arrays$b_cond
  y_scale_list = list_of_arrays$y_scale
  sigma_list = list_of_arrays$sigma

  par(mfrow=c(2,2))
  hist(y_scale_list, main=paste("y_scale", measure))
  hist(sigma_list, main=paste("sigma", measure))
  b_cond = array(dim=c(num_conditions))
  for (i in 1:num_conditions) {
    print(paste("Stats for b_cond", i))
    print(summary(b_cond_list[,i]))
    print(sd(b_cond_list[,i]))
    hist(b_cond_list[,i], main=paste("b_cond", i, measure))
    do_plot_post(b_cond_list[,i])
    b_cond[i] = mean(b_cond_list[,i])
  }
  y_scale = mean(y_scale_list)
  print(paste("y_scale", measure, y_scale))
  plot_fn_factory = function(cond_num) {
    plot_fn = function(x) {
      return(y_scale * (x ^ b_cond[cond_num]))
    }
    return(plot_fn)
  }
  plot_model(df, measure, plot_fn_factory, "Exp Model")
  return(c(fit, plot_fn_factory))
}

plot_model = function(df, measure, plot_fn_factory, model_name) {
  x = 1:900
  num_conditions = max(df$condition_num)
  for (i in 1:num_conditions) {
    plot_fn = plot_fn_factory(i)
    plot(x, plot_fn(x), type='l', ylim=c(1,500), main=paste(model_name, "Raw/Model", measure, i))
#    lines(x,x)
    cond_df = df[df$condition_num==i,]
    lines(cond_df$response_num, cond_df[,measure])
    rms = sqrt(sum((plot_fn(cond_df$response_num) - cond_df[,measure])^2) / length(cond_df$response_num))
    print(paste(model_name, "RMS for", measure, i, ":", rms))
  }
}

do_full_plot = function(df, measure, model_data, plot_name, ylab) {
  condition_labels = c("5", "10", "20", "50", "75", "100")
  stan_model = model_data[1]
  plot_fn_factory = model_data[2][[1]]
  xlim = c(0, max(max(df$response_num), df[,measure])+50)
  ylim = xlim
  num_conditions = max(df$condition_num)
  plot(1:xlim[2], 1:xlim[2], type='l', xlim=xlim, ylim=ylim, main=plot_name, xlab="Number of Responses", ylab=ylab)
  for (i in 1:num_conditions) {
    cond_df = df[df$condition_num==i,]
    lines(cond_df$response_num, cond_df[,measure], col="dark grey")
    max_x = max(cond_df$response_num)
    plot_fn = plot_fn_factory(i)
    lines(1:max_x, plot_fn(1:max_x), type='l', col="black")
    max_y = plot_fn(max_x)
    text_pos = 1
    if (i == 5) { # 75 case
      text_pos = 4
    }
    text(max_x, max_y, condition_labels[i], pos=text_pos)
  }
  text(xlim[2]/2, xlim[2]/2, "Perfect (1:1) ", pos=2)
}

df = read.csv('cumulative_data.csv', header=TRUE, sep='\t')
df$condition_num = as.numeric(factor(df$condition))
pdf('./model_comparisons.pdf')
results = c()
for (measure in c("num_cum_ideas", "num_cum_categories")) {
  results = c(results, do_stan_cumulative_analysis_log_model(df, measure))
  results = c(results, do_stan_cumulative_analysis_exp_model(df, measure))
}
dev.off()

num_ideas_log_model = results[1:2]
num_ideas_exp_model = results[3:4]
num_categories_log_model = results[5:6]
num_categories_exp_model = results[7:8]

pdf('./fitted_models.pdf')
par(mfrow=c(1,1))
do_full_plot(df, "num_cum_ideas", num_ideas_exp_model, "Number of Unique Ideas (Exp Model)", "Number of Unique Ideas")
do_full_plot(df, "num_cum_ideas", num_ideas_log_model, "Number of Unique Ideas (Log Model)", "Number of Unique Ideas")
do_full_plot(df, "num_cum_categories", num_categories_exp_model, "Number of Unique Categories (Exp Model)", "Number of Unique Categories")
do_full_plot(df, "num_cum_categories", num_categories_log_model, "Number of Unique Categories (Log Model)", "Number of Unique Categories")
dev.off()
