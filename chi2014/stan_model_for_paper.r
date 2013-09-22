require(rstan)

kruschke_script_dir = '/home/mterry/Documents/probability_work/kruschke_scripts'
all_data_fname = "./cumulative_data/cumulative_ideas_categories_leave_out_None.csv"

do_plot_post = function(data) {
  cur_dir = getwd()
  setwd(kruschke_script_dir)
  source('plotPost.R')
  plotPost(data)
  setwd(cur_dir)
}
  
do_stan_cumulative_analysis_exp_model_basic_fit = function(df, measure) {
model_string = "
data {
  int N;
  real y[N];
  int x[N];
}

parameters {
  real <lower=0, upper=1> rate;
  real<lower=0> y_scale;
  real<lower=0> sigma;
}

transformed parameters {
}

model {
  real mu[N];
  for (i in 1:N) {
    mu[i] <- y_scale * pow(x[i], rate);
     y[i] ~ normal(mu[i], sigma);
  }
}
";
  num_conditions = max(df$condition_num)
  data_list = list(
    N = nrow(df),
    y = df[,measure],
    x = df$response_num+1
  );

  n_chains = 3;
  n_iter = 10000;
  print(paste("Fitting basic exp model for", measure))
  fit = stan(model_code = model_string, data = data_list, iter = n_iter, chains = n_chains)

  print(paste("Exp model for", measure))
  print(fit)
  par(mfrow=c(1,1))
  plot(fit)

  list_of_arrays = extract(fit, permuted=TRUE)
  rate_list = list_of_arrays$rate
  y_scale_list = list_of_arrays$y_scale
  sigma_list = list_of_arrays$sigma

  print("y_scale")
  print(summary(y_scale_list))
  print(sd(y_scale_list))
  print("rate")
  print(summary(rate_list))
  print(sd(rate_list))

  par(mfrow=c(2,2))
  hist(rate_list, main=paste("rate", measure))
  do_plot_post(rate_list)
  hist(y_scale_list, main=paste("y_scale", measure))
  do_plot_post(y_scale_list)
  hist(sigma_list, main=paste("sigma", measure))
  rate = mean(rate_list)
  y_scale = mean(y_scale_list)
  print(paste("y_scale", y_scale, measure))
  print(paste("rate", rate, measure))
  plot_fn = function(x) {
    return(y_scale * x^rate)
  }
#  plot_model(df, measure, plot_fn_factory, "Log Model")
  return(c(fit, plot_fn))
}

do_full_basic_plot = function(df, measure, model_data, plot_name, ylab) {
  stan_model = model_data[1]
  plot_fn = model_data[2][[1]]
  xlim = c(0, max(max(df$response_num), df[,measure])+50)
  ylim = xlim
  num_conditions = max(df$condition_num)
  condition_labels = c("5", "10", "20", "50", "75", "100")
  par(mfrow=c(1,1))
  plot(1:xlim[2], 1:xlim[2], type='l', xlim=xlim, ylim=ylim, main=plot_name, xlab="Number of Responses", ylab=ylab)
  for (i in 1:num_conditions) {
    cond_df = df[df$condition_num==i,]
    lines(cond_df$response_num, cond_df[,measure], col="blue")
    max_x = max(cond_df$response_num)
    max_y = max(cond_df[,measure])
#    max_y = plot_fn(max_x)
    text_pos = 1
    if (i == 5) { # 75 case
      text_pos = 4
    }
    text(max_x, max_y, condition_labels[i], pos=text_pos)
  }
  last_offset = 30
  lines(1:(max_x+last_offset), plot_fn(1:(max_x+last_offset)), type='l', col="black")
  text(max_x+last_offset, plot_fn(max_x+last_offset), "Best Fit", pos=2)
  text(xlim[2]/2, xlim[2]/2, "Ideal (1:1) ", pos=2)
}


# New Paper stuff
all_results = c()
fname = all_data_fname
df = read.csv(all_data_fname, header=TRUE, sep='\t')
df$all_condition_nums = as.numeric(factor(df$condition))
df$condition_num = df$all_condition_nums
for (measure in c("num_cum_ideas", "num_cum_categories")) {
  fout = file(paste(fname, "_", measure, "_results.txt", sep=""))
  sink(fout)
  sink(fout, type=c("message"), append=TRUE)
  pdf(paste(fname, "_", measure, "_graphs.pdf", sep=""))
  print(paste(measure, " Results for", fname))
  results = do_stan_cumulative_analysis_exp_model_basic_fit(df, measure)
  all_results = c(all_results, results)
  if (measure == 'num_cum_ideas') {
    title = "Cumulative Number of Ideas"
  } else {
    title = "Cumulative Number of Idea Categories"
  }
  do_full_basic_plot(df, measure, results, title, title)
  dev.off()
  sink()
  sink(type="message")
}
pdf("cumulative_ideas.pdf")
do_full_basic_plot(df, "num_cum_ideas", all_results[1:2], "Cumulative Number of Unique Ideas", "Cumulative Number of Unique Ideas")
dev.off()
pdf("cumulative_categories.pdf")
do_full_basic_plot(df, "num_cum_categories", all_results[3:4], "Cumulative Number of Unique Idea Categories", "Cumulative Number of Unique Idea Categories")
dev.off()

