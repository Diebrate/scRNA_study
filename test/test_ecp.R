library(ecp)
library(reticulate)
library(dplyr)

res <- list()

B <- 50
for(m in 1:100){
  for(b in 0:(B-1)){
    dataX <- read.csv(paste0('../data/simulation_data/simulation_id', as.character(m), '_', as.character(b), '.csv'))

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
    cp <- rep(0, T)
    if(length(res_temp) != 0){
      cp[res_temp-2] <- 1
    }
    res <- append(res, list(cp))
  }
}

res <- do.call(rbind, res)
saveRDS(res, file='../results/simulation/test_ecp.RDS')
