library(ecp)
library(dplyr)


args <- commandArgs(trailingOnly = TRUE)
m <- as.integer(args[1])
# m <- 0

for(n0 in c('low', 'high')){

  for(nu0 in c('low', 'high')){

    for(eta0 in c('low', 'high')){

      dataX <- read.csv(paste0('../data/simulation_data/simulation_id', as.character(m), '_', n0, '_n_', nu0, '_nu_', eta0, '_eta.csv'))
      # dataX <- read.csv(paste0('../data/simulation_data/test_sample_', n0, '_n_', nu0, '_nu_', eta0, '_eta.csv'))
      B <- dataX$batch %>% max() + 1
      T <- dataX$time %>% max()
      d <- dataX$type %>% max() + 1

      res <- list()

      for(b in 1:B){

        data0 <- dataX %>% filter(batch == (b - 1))
        data0$type <- factor(data0$type, levels=0:(d-1))

        prob <- matrix(nrow=T+1, ncol=d)

        for(t in 1:(T+1)){
          prob[t,] <- data0 %>% filter(time == (t - 1)) %>%
            group_by(type, .drop=FALSE) %>%
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
        print(paste0('working on m = ', as.character(m), ', b = ', as.character(b - 1), ', n = ', n0, ', nu = ', nu0, ', eta = ', eta0))

      }

    res <- do.call(rbind, res)
    saveRDS(res, file=paste0('../results/simulation/', n0, '_n/', nu0, '_nu_', eta0, '_eta/test_ecp_id', as.character(m), '.RDS'))

    }

  }

}


