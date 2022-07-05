library(wbsts)
library(ecp)
library(reticulate)

set.seed(12345)
# setwd(dirname(rstudioapi::getSourceEditorContext()$path))

np <- import("numpy")

wbs_from_prob <- function(probs){
  dim_setting <- dim(probs)
  dim_mc <- dim(probs[1,1,1][[1]])
  n_ns <- dim_setting[1]
  n_nus <- dim_setting[2]
  n_etas <- dim_setting[3]
  res <- numeric(0)
  M <- dim_mc[1]
  t <- dim_mc[2]
  d <- dim_mc[3]
  for(i in 1:n_ns){
    for(j in 1:n_nus){
      for(k in 1:n_etas){
        for(m in 1:M){
          temp <- numeric(0)
          prob_temp <- probs[i, j, k][[1]][m, , ]
          for(d_temp in 1:d){
            res_temp <- 0
            tryCatch({
              res_temp <- changepoints(wbs(prob_temp[, d_temp]))$cpt.th[[1]]
            }, error=function(e){})
            if(length(res_temp) == 1){
              if(is.na(res_temp)){
                res_temp <- numeric(0)
              }
            }
            temp <- c(temp, res_temp)
          }
          temp <- unique(temp)
          temp <- temp[temp > 0]
          if(length(temp) != 0){
            res <- c(res, temp - 1, 'done')
          }else{
            res <- c(res, 'null', 'null')
          }
        }
      }
    }
  }
  return(res)
}

wbs_from_prob_ng <- function(probs){
  dim_setting <- dim(probs)
  dim_mc <- dim(probs[1, 1][[1]])
  n_ns <- dim_setting[1]
  n_etas <- dim_setting[2]
  res <- numeric(0)
  M <- dim_mc[1]
  t <- dim_mc[2]
  d <- dim_mc[3]
  for(i in 1:n_ns){
    for(j in 1:n_etas){
      temp <- numeric(0)
      for(m in 1:M){
        temp <- numeric(0)
        prob_temp <- probs[i, j][[1]][m, , ]
        for(d_temp in 1:d){
          res_temp <- 0
          tryCatch({
            res_temp <- changepoints(wbs(prob_temp[, d_temp]))$cpt.th[[1]]
          }, error=function(e){})
          if(length(res_temp) == 1){
            if(is.na(res_temp)){
              res_temp <- numeric(0)
            }
          }
          temp <- c(temp, res_temp)
        }
        temp <- unique(temp)
        temp <- temp[temp > 0]
        if(length(temp) != 0){
          res <- c(res, temp - 1, 'done')
        }else{
          res <- c(res, 'null', 'done')
        }
      }
    }
  }
  return(res)
}

ecp_from_prob <- function(probs){
  dim_setting <- dim(probs)
  dim_mc <- dim(probs[1,1,1][[1]])
  n_ns <- dim_setting[1]
  n_nus <- dim_setting[2]
  n_etas <- dim_setting[3]
  res <- numeric(0)
  M <- dim_mc[1]
  t <- dim_mc[2]
  d <- dim_mc[3]
  for(i in 1:n_ns){
    for(j in 1:n_nus){
      for(k in 1:n_etas){
        for(m in 1:M){
          res_temp <- e.divisive(probs[i, j, k][[1]][m, , ], min.size=2)$estimates
          res_temp <- res_temp[res_temp != 1]
          res_temp <- res_temp[res_temp != (t + 1)]
          if(length(res_temp) != 0){
            res <- c(res, res_temp - 2, 'done')
          }else{
            res <- c(res, 'null', 'null')
          }
        }
      }
    }
  }
  return(res)
}

ecp_from_prob_ng <- function(probs){
  dim_setting <- dim(probs)
  dim_mc <- dim(probs[1,1][[1]])
  n_ns <- dim_setting[1]
  n_etas <- dim_setting[2]
  res <- numeric(0)
  M <- dim_mc[1]
  t <- dim_mc[2]
  d <- dim_mc[3]
  for(i in 1:n_ns){
    for(j in 1:n_etas){
      for(m in 1:M){
        res_temp <- e.divisive(probs[i, j][[1]][m, , ], min.size=2)$estimates
        res_temp <- res_temp[res_temp != 1]
        res_temp <- res_temp[res_temp != (t + 1)]
        if(length(res_temp) != 0){
          res <- c(res, res_temp - 2, 'done')
        }else{
          res <- c(res, 'null', 'null')
        }
      }
    }
  }
  return(res)
}

offsets = c(TRUE, FALSE)

for(offset in offsets){
  
  if(offset){
    prob_all <- np$load('../results/simulation/multisetting_data_prob_cor.npy', allow_pickle=TRUE)
  }else{
    prob_all <- np$load('../results/simulation/multisetting_data_prob.npy', allow_pickle=TRUE)
  }
  # prob_all_ng <- np$load('../results/simulation/multisetting_data_prob_ng.npy', allow_pickle=TRUE)
  
  # res_all <- np$load('../results/simulation/multisetting_rtemplate.npy', allow_pickle=TRUE)
  # res_all_ng <- np$load('../results/simulation/multisetting_rtemplate_ng.npy', allow_pickle=TRUE)
  
  # n_ns <- dim(prob_all)[1]
  # n_nus <- dim(prob_all)[2]
  # n_etas <- dim(prob_all)[3]
  # M <- dim(prob_all[1,1,1][[1]])[1]
  # t <- dim(prob_all[1,1,1][[1]])[2]
  # d <- dim(prob_all[1,1,1][[1]])[3]
  # for(i in 1:n_ns){
  #   for(j in 1:n_nus){
  #     for(k in 1:n_etas){
  #       for(m in 1:M){
  #         for(d_temp in 1:d){
  #           temp <- 0
  #           print(paste('i =', i, 'j =', j, 'k =', k, 'm =', m, 'd =', d_temp))
  #           temp <- wbs.lsw(prob_all[i, j, k][[1]][m,,d_temp])$cp.bef
  #           print(temp)
  #         }
  #       }
  #     }
  #   }
  # }
  
  # res_ot_wbs <- wbs_from_prob(prob_all)
  # res_ot_wbs_ng <- wbs_from_prob_ng(prob_all_ng)
  
  res_ot_ecp <- ecp_from_prob(prob_all)
  # res_ot_ecp_ng <- ecp_from_prob_ng(prob_all_ng)
  
  # if(offset){
  #   np$save('../results/simulation/multisetting_res_wbs_cor.npy', res_ot_wbs)
  # }else{
  #   np$save('../results/simulation/multisetting_res_wbs.npy', res_ot_wbs)
  # }
  # np$save('../results/simulation/multisetting_res_wbs_ng.npy', res_ot_wbs_ng)
  
  if(offset){
    np$save('../results/simulation/multisetting_res_ecp_cor.npy', res_ot_ecp)
  }else{
    np$save('../results/simulation/multisetting_res_ecp.npy', res_ot_ecp)
  }
  # np$save('../results/simulation/multisetting_res_ecp_ng.npy', res_ot_ecp_ng)
  
  # saveRDS(res_ot_wbs, '../results/simulation/multisetting_res_wbs.rds')
  # saveRDS(res_ot_wbs_ng, '../results/simulation/multisetting_res_wbs_ng.rds')

}

prob_all_ng <- np$load('../results/simulation/multisetting_data_prob_ng.npy', allow_pickle=TRUE)
res_ot_ecp_ng <- ecp_from_prob_ng(prob_all_ng)
np$save('../results/simulation/multisetting_res_ecp_ng.npy', res_ot_ecp_ng)








































