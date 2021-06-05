import numpy as np
import pandas as pd
import scipy.io
import test_util

import time
start_time = time.time()

np.random.seed(1030)

nus = [0.25, 0.75]
etas = [0.25, 0.75]
ns = [250, 1000]
pwrs = [1.25, 1.5, 1.75, 2, 2.5]
d = 10
T = 50
sep = int(d / 2)
real_cp = [int(i * T) for i in [0.2, 0.4, 0.6, 0.8]]
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

do_sim = True
do_load = True
do_test = True
do_comp = True

data_type = 'g'
offset = True
win_size = 2
weight = 'frac'
switch = False

index = ['OT', 'MN', 'ECP']
name_setting = ('' if win_size == 1 else '_m' + str(win_size)) + ('' if weight is None else '_' + weight)

if do_sim:
    if data_type == 'g':
        data_all = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        data_mat = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        prob_all = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        prob_cor = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        data_cor = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        cost_all = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        for i in range(len(ns)):
            for j in range(len(nus)):
                for k in range(len(etas)):
                    data_all[i, j, k] = test_util.dgp(nus[j], etas[k], real_cp, g, d, T, ns[i], M)
                    data_mat[i, j, k] = test_util.get_weight_no_ignore_mc(data_all[i, j, k], T, d)
                    prob_all[i, j, k] = test_util.get_weight_no_ignore_mc(data_all[i, j, k], T, d, matformat=False, count=False)
                    prob_cor[i, j, k] = test_util.offset_growth_mc(prob_all[i, j, k], costm, reg, reg1)
                    data_cor[i, j, k] = test_util.dgp_with_prob(prob_cor[i, j, k], ns[i])
                    # cost_all[i, j, k] = test_util.get_ot_unbalanced_cost_mc(prob_all[i, j, k], costm, reg, reg1, reg2, sink=sink)
                    cost_all[i, j, k] = test_util.get_ot_unbalanced_cost_local_mc(prob_all[i, j, k], costm, reg, reg1, reg2, sink=sink, win_size=win_size, weight=weight)
        np.save(r'..\results\simulation\multisetting_data.npy', data_all)
        np.save(r'..\results\simulation\multisetting_data_prob.npy', prob_all)
        np.save(r'..\results\simulation\multisetting_data_prob_cor.npy', prob_cor)
        np.save(r'..\results\simulation\multisetting_data_cost' + name_setting + '.npy', cost_all)
        scipy.io.savemat(r'..\results\simulation\multisetting_data.mat', dict(data=data_mat, ns=ns, zetas=etas, nus=nus, M=M, T=T, p=d))
        scipy.io.savemat(r'..\results\simulation\multisetting_data_cor.mat', dict(data=data_cor, ns=ns, zetas=etas, nus=nus, M=M, T=T, p=d))
    elif data_type == 'ng':
        if switch:
            data_all = np.empty((len(ns), len(pwrs)), dtype=object)
            data_mat = np.empty((len(ns), len(pwrs)), dtype=object)
            prob_all = np.empty((len(ns), len(pwrs)), dtype=object)
            cost_all = np.empty((len(ns), len(pwrs)), dtype=object)
            for i in range(len(ns)):
                for j in range(len(pwrs)):
                    data_all[i, j] = test_util.dgp_ng_switch(pwrs[j], real_cp, d, T, ns[i], M)
                    data_mat[i, j] = test_util.get_weight_no_ignore_mc(data_all[i, j], T, d)
                    prob_all[i, j] = test_util.get_weight_no_ignore_mc(data_all[i, j], T, d, matformat=False, count=False)
                    # cost_all[i, j] = test_util.get_ot_unbalanced_cost_mc(prob_all[i, j], costm, reg, reg1, reg2, sink=sink)
                    cost_all[i, j] = test_util.get_ot_unbalanced_cost_local_mc(prob_all[i, j], costm, reg, reg1, reg2, sink=sink, win_size=win_size, weight=weight)
            scipy.io.savemat(r'..\results\simulation\multisetting_data_ng.mat', dict(data=data_mat, ns=ns, pwrs=pwrs, M=M, T=T, p=d))
        else:
            data_all = np.empty((len(ns), len(etas)), dtype=object)
            data_mat = np.empty((len(ns), len(etas)), dtype=object)
            prob_all = np.empty((len(ns), len(etas)), dtype=object)
            cost_all = np.empty((len(ns), len(etas)), dtype=object)
            for i in range(len(ns)):
                for j in range(len(etas)):
                    data_all[i, j] = test_util.dgp_ng(etas[j], real_cp, d, T, ns[i], M)
                    data_mat[i, j] = test_util.get_weight_no_ignore_mc(data_all[i, j], T, d)
                    prob_all[i, j] = test_util.get_weight_no_ignore_mc(data_all[i, j], T, d, matformat=False, count=False)
                    # cost_all[i, j] = test_util.get_ot_unbalanced_cost_mc(prob_all[i, j], costm, reg, reg1, reg2, sink=sink)
                    cost_all[i, j] = test_util.get_ot_unbalanced_cost_local_mc(prob_all[i, j], costm, reg, reg1, reg2, sink=sink, win_size=win_size, weight=weight)
            scipy.io.savemat(r'..\results\simulation\multisetting_data_ng.mat', dict(data=data_mat, ns=ns, etas=etas, M=M, T=T, p=d))
        np.save(r'..\results\simulation\multisetting_data_ng.npy', data_all)
        np.save(r'..\results\simulation\multisetting_data_prob_ng.npy', prob_all)
        np.save(r'..\results\simulation\multisetting_data_cost_ng' + name_setting + '.npy', cost_all)
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
        cost_all = np.load(r'..\results\simulation\multisetting_data_cost' + name_setting + '.npy', allow_pickle=True)
    elif data_type == 'ng':
        data_all = np.load(r'..\results\simulation\multisetting_data_ng.npy', allow_pickle=True)
        prob_all = np.load(r'..\results\simulation\multisetting_data_prob_ng.npy', allow_pickle=True)
        cost_all = np.load(r'..\results\simulation\multisetting_data_cost_ng' + name_setting + '.npy', allow_pickle=True)


if do_test:
    if data_type == 'g':
        res_ot = test_util.multisetting_cp_ot_cost(cost_all, T, win_size=win_size)
        np.save(r'..\results\simulation\multisetting_res_ot' + name_setting + '.npy', res_ot)
    elif data_type == 'ng':
        res_ot = test_util.multisetting_cp_ot_cost_ng(cost_all, T, win_size=win_size)
        np.save(r'..\results\simulation\multisetting_res_ot_ng' + name_setting + '.npy', res_ot)
  
        
if do_comp:
    if data_type == 'g':
        
        res_ot = np.load(r'..\results\simulation\multisetting_res_ot' + name_setting + '.npy', allow_pickle=True)
        
        # res_mn_raw = scipy.io.loadmat(r'..\results\simulation\multisetting_res_mn.mat')['cpt']
        res_mn_raw = scipy.io.loadmat(r'..\results\simulation\multisetting_res_mn' + ('_cor' if offset else '') + '.mat')['cpt']
        res_mn = test_util.get_res_from_others(res_mn_raw, len(ns), len(nus), len(etas), M, ftype='mat')
                    
        res_ecp_raw = np.load(r'..\results\simulation\multisetting_res_ecp' + ('_cor' if offset else '') + '.npy')
        res_ecp = test_util.get_res_from_others(res_ecp_raw, len(ns), len(nus), len(etas), M, ftype='r')
                    
        # res_wbs_raw = np.load(r'..\results\simulation\multisetting_res_wbs' + ('_cor' if offset else '') + '.npy')
        # res_wbs = test_util.get_res_from_others(res_wbs_raw, len(ns), len(nus), len(etas), M, ftype='r')
                    
        res_all = test_util.prf_table_all(res_ot, res_mn, res_ecp, ns=ns, nus=nus, etas=etas, real_cp=real_cp, index=index, win_size=win_size) 
        res_all.to_csv(r'..\results\simulation\multisetting_res_table' + name_setting + '.csv')
    
    elif data_type == 'ng':
        
        n_temp = len(pwrs) if switch else len(etas)
        temps = pwrs.copy() if switch else etas.copy()
        res_ot = np.load(r'..\results\simulation\multisetting_res_ot_ng' + name_setting + '.npy', allow_pickle=True)
        
        res_mn_raw = scipy.io.loadmat(r'..\results\simulation\multisetting_res_mn_ng.mat')['cpt']
        res_mn = test_util.get_res_from_others_ng(res_mn_raw, len(ns), n_temp, M, ftype='mat')
        
        res_ecp_raw = np.load(r'..\results\simulation\multisetting_res_ecp_ng.npy')
        res_ecp = test_util.get_res_from_others_ng(res_ecp_raw, len(ns), n_temp, M, ftype='r')
        
        # res_wbs_raw = np.load(r'..\results\simulation\multisetting_res_wbs_ng.npy')
        # res_wbs = test_util.get_res_from_others_ng(res_wbs_raw, len(ns), n_temp, M, ftype='r')
        
        res_all = test_util.prf_table_all_ng(res_ot, res_mn, res_ecp, ns=ns, etas=temps, real_cp=real_cp, switch=switch, index=index, win_size=win_size)
        res_all.to_csv(r'..\results\simulation\multisetting_res_table_ng' + name_setting + '.csv')

    pd.set_option('display.max_columns', None)
    print(res_all)

print("--- %s seconds ---" % (time.time() - start_time))