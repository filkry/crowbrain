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
  real<lower=0> mu;
  real<lower=0> sigma;
}

model {
  for (i in 1:N) {
    y[i] ~ normal(mu, sigma);
  }
}
";
  
add.error.bars <- function(X,ydown, yup,w,col=1){
    arrows(X, ydown, X, yup, code=3,angle=90,length=w,col=col);
}

fit_model = function(df, condition, measure) {
  sdf = subset(df, num_requested == condition)
  
  max_order = max(sdf$order)
  print(max_order)

  xs = seq(1, max_order)
  lys = c()
  rys = c()
  l_bot = c()
  r_bot = c()
  l_top = c()
  r_top = c()

  pass = 1
  last_pass = 0

  for (i in 1:max_order) {
    ldf = subset(sdf, order < i)
    rdf = subset(sdf, order >= i)

    l_data_list = list(
      N = nrow(ldf),
      y = ldf[,measure]
    );

    n_chains = 3;
    num_saved_steps = 20000;
    n_iter = ceiling(num_saved_steps / n_chains)
    print(paste("Fitting for", measure, condition,i, "left"))
    lfit = stan(model_code = model_string, data = l_data_list, iter = n_iter, chains = n_chains)

    r_data_list = list(
      N = nrow(rdf),
      y = rdf[,measure]
    );

    print(paste("Fitting for", measure, condition,i, "right"))
    rfit = stan(model_code = model_string, data = r_data_list, iter = n_iter, chains = n_chains)

    llist = extract(lfit, permuted=TRUE)
    rlist = extract(rfit, permuted=TRUE)

    lmu_list = llist$mu
    rmu_list = rlist$mu

    left_hdi = HDIofMCMC(lmu_list)
    right_hdi = HDIofMCMC(rmu_list)
    
    lys = c(lys, mean(lmu_list))
    rys = c(rys, mean(rmu_list))

    l_bot = c(l_bot, left_hdi[1])
    r_bot = c(r_bot, right_hdi[1])

    l_top = c(l_top, left_hdi[2])
    r_top = c(r_top, right_hdi[2])

    if (left_hdi[2] > right_hdi[1] & pass==1) {
        pass = 0
        print(paste(measure, condition, "left caught up to right at", last_pass))
    } else {
        last_pass = i
    }

  }
  
  print(l_bot)
  print(r_bot)

  minY = min(l_bot, r_bot)
  maxY = max(l_top, r_top)

  plot(xs, lys, type="l", col="red", ylim=c(minY, maxY), main=paste(condition, measure,"last right > left significant at", last_pass))
  add.error.bars(xs, l_bot, l_top, 0.1, 'red')
  lines(xs, rys, col="green")
  add.error.bars(xs, r_bot, r_top, 0.1, 'green')
}

df = read.csv('hyp6.csv', header=TRUE, sep='\t')

pdf('./originality_models.pdf')
par(mfrow=c(2,1))
for (measure in c("tree_oscore", "idea_oscore")) {
  for (condition in c(5, 10, 20, 50, 75, 100)) {
    fit_model(df, condition, measure)
  }
}
dev.off()

