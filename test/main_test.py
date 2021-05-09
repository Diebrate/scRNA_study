import numpy as np
import pandas as pd
import scipy.io
import test_util

import time
start_time = time.time()

np.random.seed(1030)

index = ['OT', 'CS']

nus = [0.25, 0.75]
etas = [0.25, 0.75]
ns = [250, 1000]
pwrs = [1.25, 1.5, 1.75, 2, 2.5]
d = 10
T = 50
sep = int(d / 2)
cp = [int(i * T) for i in [0.2, 0.4, 0.6, 0.8]]
# g = np.tile(np.concatenate((np.exp(np.arange(sep)), np.exp(np.arange(d - sep)))), (T, 1))
# g = np.concatenate((np.sin(2 * np.arange(1, sep + 1) * np.pi / sep), 
#                     np.cos(2 * np.arange(1, d - sep + 1) * np.pi / (d - sep))))
# g = np.tile(g, (T, 1))
g = np.ones((T, d))
for t in range(T):
    ind = t
    g[t, ] = np.sin(np.arange(ind, ind + d) * np.pi / d)
    # g[t, ] = np.concatenate((np.sin(2 * np.arange(ind, ind + sep) * np.pi / sep), 
    #                          np.cos(2 * np.arange(ind, ind + d - sep) * np.pi / (d - sep))))
    g[t, ] = np.exp(g[t, ])
M = 30

costm = test_util.constant_costm(d, 1, 2)
reg = 1
reg1 = 1
reg2 = 1
sink = True
balanced = False
n_conv = 5

seed = 99999

do_sim = False
do_load = False
do_test = False
do_comp = True

data_type = 'g'
switch = False

index = ['OT', 'MN']

if do_sim:
    if data_type == 'g':
        data_all = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        data_mat = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        prob_all = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        cost_all = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        for i in range(len(ns)):
            for j in range(len(nus)):
                for k in range(len(etas)):
                    data_all[i, j, k] = test_util.dgp(nus[j], etas[k], cp, g, d, T, ns[i], M)
                    data_mat[i, j, k] = test_util.get_weight_no_ignore_mc(data_all[i, j, k], T, d)
                    prob_all[i, j, k] = test_util.get_weight_no_ignore_mc(data_all[i, j, k], T, d, matformat=False, count=False)
                    cost_all[i, j, k] = test_util.get_ot_unbalanced_cost_mc(prob_all[i, j, k], costm, reg, reg1, reg2, sink=sink)
        np.save(r'..\results\simulation\multisetting_data.npy', data_all)
        np.save(r'..\results\simulation\multisetting_data_prob.npy', prob_all)
        np.save(r'..\results\simulation\multisetting_data_cost.npy', cost_all)
        scipy.io.savemat(r'..\results\simulation\multisetting_data.mat', dict(data=data_mat, ns=ns, zetas=etas, nus=nus, M=M, T=T, p=d))
    elif data_type == 'ng':
        if switch:
            data_all = np.empty((len(ns), len(pwrs)), dtype=object)
            data_mat = np.empty((len(ns), len(pwrs)), dtype=object)
            prob_all = np.empty((len(ns), len(pwrs)), dtype=object)
            cost_all = np.empty((len(ns), len(pwrs)), dtype=object)
            for i in range(len(ns)):
                for j in range(len(pwrs)):
                    data_all[i, j] = test_util.dgp_ng_switch(pwrs[j], cp, d, T, ns[i], M)
                    data_mat[i, j] = test_util.get_weight_no_ignore_mc(data_all[i, j], T, d)
                    prob_all[i, j] = test_util.get_weight_no_ignore_mc(data_all[i, j], T, d, matformat=False, count=False)
                    cost_all[i, j] = test_util.get_ot_unbalanced_cost_mc(prob_all[i, j], costm, reg, reg1, reg2, sink=sink)
            scipy.io.savemat(r'..\results\simulation\multisetting_data_ng.mat', dict(data=data_mat, ns=ns, pwrs=pwrs, M=M, T=T, p=d))
        else:
            data_all = np.empty((len(ns), len(etas)), dtype=object)
            data_mat = np.empty((len(ns), len(etas)), dtype=object)
            prob_all = np.empty((len(ns), len(etas)), dtype=object)
            cost_all = np.empty((len(ns), len(etas)), dtype=object)
            for i in range(len(ns)):
                for j in range(len(etas)):
                    data_all[i, j] = test_util.dgp_ng(etas[j], cp, d, T, ns[i], M)
                    data_mat[i, j] = test_util.get_weight_no_ignore_mc(data_all[i, j], T, d)
                    prob_all[i, j] = test_util.get_weight_no_ignore_mc(data_all[i, j], T, d, matformat=False, count=False)
                    cost_all[i, j] = test_util.get_ot_unbalanced_cost_mc(prob_all[i, j], costm, reg, reg1, reg2, sink=sink)
            scipy.io.savemat(r'..\results\simulation\multisetting_data_ng.mat', dict(data=data_mat, ns=ns, etas=etas, M=M, T=T, p=d))
        np.save(r'..\results\simulation\multisetting_data_ng.npy', data_all)
        np.save(r'..\results\simulation\multisetting_data_prob_ng.npy', prob_all)
        np.save(r'..\results\simulation\multisetting_data_cost_ng.npy', cost_all)
    n_temp = len(pwrs) if switch else len(etas)
    res_rtemp_ng = np.empty((len(ns), n_temp), dtype=object)
    res_rtemp = np.empty((len(ns), len(nus), len(etas)), dtype=object)
    for i in range(len(ns)):
        for j in range(n_temp):
            res_rtemp_ng[i, j] = np.zeros((M, T + 1))
        for j in range(len(nus)):
            for k in range(len(etas)):
                res_rtemp[i, j, k] = np.zeros((M, T + 1))
    np.save(r'..\results\simulation\multisetting_rtemplate_ng.npy', res_rtemp_ng)
    np.save(r'..\results\simulation\multisetting_rtemplate.npy', res_rtemp)

    
if do_load:
    if data_type == 'g':
        data_all = np.load(r'..\results\simulation\multisetting_data.npy', allow_pickle=True)
        prob_all = np.load(r'..\results\simulation\multisetting_data_prob.npy', allow_pickle=True)
        cost_all = np.load(r'..\results\simulation\multisetting_data_cost.npy', allow_pickle=True)
    elif data_type == 'ng':
        data_all = np.load(r'..\results\simulation\multisetting_data_ng.npy', allow_pickle=True)
        prob_all = np.load(r'..\results\simulation\multisetting_data_prob_ng.npy', allow_pickle=True)
        cost_all = np.load(r'..\results\simulation\multisetting_data_cost_ng.npy', allow_pickle=True)


if do_test:
    if data_type == 'g':
        res_ot = test_util.multisetting_cp_ot_cost(cost_all, T)
        np.save(r'..\results\simulation\multisetting_res_ot.npy', res_ot)
    elif data_type == 'ng':
        res_ot = test_util.multisetting_cp_ot_cost_ng(cost_all, T)
        np.save(r'..\results\simulation\multisetting_res_ot_ng.npy', res_ot)
  
        
if do_comp:
    if data_type == 'g':
        
        res_ot = np.load(r'..\results\simulation\multisetting_res_ot.npy', allow_pickle=True)
        
        res_mn_raw = scipy.io.loadmat(r'..\results\simulation\multisetting_res_mn.mat')['cpt']
        res_mn = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        for i in range(len(ns)):
            for j in range(len(nus)):
                for k in range(len(etas)):
                    res_mn[i, j, k] = []
                    for m in range(M):
                        res_mn[i, j, k].append(res_mn_raw[i, j, k, m].flatten().astype(dtype=int))
                    res_mn[i, j, k] = np.array(res_mn[i, j, k], dtype=object)
                    
        res_all = test_util.prf_table_all(res_ot, res_mn, ns=ns, nus=nus, etas=etas, real_cp=cp, index=index) 
    
    elif data_type == 'ng':
        
        n_temp = len(pwrs) if switch else len(etas)
        temps = pwrs.copy() if switch else etas.copy()
        res_ot = np.load(r'..\results\simulation\multisetting_res_ot_ng.npy', allow_pickle=True)
        
        res_mn_raw = scipy.io.loadmat(r'..\results\simulation\multisetting_res_mn_ng.mat')['cpt']
        res_mn = np.empty((len(ns), n_temp), dtype=object)
        for i in range(len(ns)):
            for j in range(n_temp):
                res_mn[i, j] = []
                for m in range(M):
                    res_mn[i, j].append(res_mn_raw[i, j, m].flatten().astype(dtype=int))
                res_mn[i, j] = np.array(res_mn[i, j], dtype=object)
        
        res_all = test_util.prf_table_all_ng(res_ot, res_mn, ns=ns, etas=temps, real_cp=cp, switch=switch, index=index)

    pd.set_option('display.max_columns', None)
    print(res_all)

print("--- %s seconds ---" % (time.time() - start_time))