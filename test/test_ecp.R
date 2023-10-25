library(ecp)
library(dplyr)

res <- list()

args <- commandArgs(trailingOnly = TRUE)
m <- as.integer(args[1])
# m <- 0

dataX <- read.csv(paste0('../data/simulation_data/simulation_id', as.character(m), '.csv'))
# dataX <- read.csv(paste0('../data/simulation_data/test_sample.csv'))

B <- dataX$batch %>% max() + 1
T <- dataX$time %>% max()
d <- dataX$type %>% max() + 1

for(b in 1:B){

  print(paste0('working on batch ', as.character(m), ' iteration ', as.character(b)))

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
  cp <- rep(0, T)
  if(length(res_temp) != 0){
    cp[res_temp-2] <- 1
  }
  res <- append(res, list(cp))
  print(paste0('working on batch ', as.character(m), 'iteration ', as.character(b)))

}

res <- do.call(rbind, res)
saveRDS(res, file=paste0('../results/simulation/test_ecp_id', as.character(m), '.RDS'))


