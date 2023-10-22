library(ecp)
library(reticulate)
library(dplyr)

np <- import("numpy")

dataX <- read.csv('../data/simulation_data/simulation_id0.csv')

T <- dataX$time %>% max()
d <- dataX$type %>% max() + 1
prob <- matrix(nrow=T+1, ncol=d)

for(t in 1:(T+1)){
  prob[t,] <- dataX %>% filter(time == (t - 1)) %>%
    group_by(type) %>%
    summarise(n=n()) %>%
    arrange(type) %>%
    mutate(p=n/sum(n)) %>%
    pull(p)
}

res_temp <- e.divisive(prob, min.size=2)$estimate
res_temp <- res_temp[res_temp != 1]
res_temp <- res_temp[res_temp != (t + 1)]
res <- numeric(0)
if(length(res_temp) != 0){
  res <- c(res, res_temp - 2, 'done')
}else{
  res <- c(res, 'null', 'null')
}