import numpy as np
import pandas as pd
import scipy.io
import test_util
import solver

import time
start_time = time.time()

np.random.seed(1030)

nus = [0.05, 0.1]
etas = [0.5, 1]
ns = [500, 1000]
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
M = 500

costm = test_util.constant_costm(d, 1, 2)
reg = 0.05
reg1 = 1
reg2 = 1
sink = True
balanced = False
n_conv = 5

seed = 99999

do_sim = True
do_load = False
do_tune = False
do_test = False
do_comp = False

### shortcut setup
# do_sim = do_load = do_tune = do_test = do_comp = False
###

data_type = 'g'
offset = True
win_size = 2
weight = None
switch = False
multimarg = False

use_sys = True
if use_sys:
    import sys
    data_type = sys.argv[1]

index = ['OT', 'MN', 'ECP']
name_setting = ('' if win_size is None else '_m' + str(win_size)) + ('' if weight is None else '_' + weight)

###############################################################################
###############################################################################
###############################################################################

'''
The following codes are for model exploration only. Sections include
unbalanced transport map convergence, true transport cost analysis and
potential model performance evaluation.
'''
# parameters for model exploration

do_check = False
do_cost = False
do_param_select = False
n_sim = 100
ns_size = [100, 1000, 10000]
n_grid = 100
grid_min = 0
grid_max = 5
time_check = 20
nu_check = 0
eta_check = 1

real_prob = np.empty((len(nus), len(etas)), dtype=object)
real_tmap = np.empty((len(nus), len(etas)), dtype=object)
for i in range(len(ns)):
    for j in range(len(nus)):
        real_prob[i, j] = test_util.get_prob_all(T, d, g, real_cp, etas[j], nus[i])
        real_tmap[i, j] = solver.ot_unbalanced_all(real_prob[i, j][:T, ].T, real_prob[i, j][1:, ].T, costm, reg, reg1, reg2)
        
real_prob_ng = np.empty(len(etas), dtype=object)
real_tmap_ng = np.empty(len(etas), dtype=object)
for i in range(len(etas)):
    real_prob_ng[i] = test_util.get_prob_ng_all(T, d, real_cp, etas[i])
    real_tmap_ng[i] = solver.ot_unbalanced_all(real_prob_ng[i][:T, ].T, real_prob_ng[i][1:, ].T, costm, reg, reg1, reg2)

# the following part is for checking convergence of unbalanced transport map

if do_check:
    res_sim = [test_util.check_unbalanced_tmap_conv(real_prob[nu_check, eta_check][time_check],
                                                    real_prob[nu_check, eta_check][time_check + 1], 
                                                    d, costm, reg, reg1, reg2, n_size, n_sim)
               for n_size in ns_size]
    test_util.plot_diff(*res_sim, ns=ns_size, title='growth')
    
    res_ng_sim = [test_util.check_unbalanced_tmap_conv(real_prob_ng[eta_check][time_check],
                                                       real_prob_ng[eta_check][time_check + 1], 
                                                       d, costm, reg, reg1, reg2, n_size, n_sim)
               for n_size in ns_size]
    test_util.plot_diff(*res_ng_sim, ns=ns_size, title='no growth')
    
# the following part is for checking true transport cost

if do_cost:
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(nrows=len(nus), ncols=len(etas), figsize = (10, 10))
    for i in range(len(nus)):
        for j in range(len(etas)):
            real_cost = solver.loss_unbalanced_all_local(real_prob[i, j], costm, reg, reg1, reg2, win_size=win_size, sink=sink)
            p_low_n = np.zeros((T + 1, d))
            p_high_n = np.zeros((T + 1, d))
            for t in range(T + 1):
                p_low_n[t, ] = np.random.multinomial(ns[0], pvals=real_prob[i, j][t, ]) / ns[0]
                p_high_n[t, ] = np.random.multinomial(ns[1], pvals=real_prob[i, j][t, ]) / ns[1]
            cost_low_n = solver.loss_unbalanced_all_local(p_low_n, costm, reg, reg1, reg2, win_size=win_size, sink=sink, weight=weight)
            cost_high_n = solver.loss_unbalanced_all_local(p_high_n, costm, reg, reg1, reg2, win_size=win_size, sink=sink, weight=weight)
            ax[i, j].plot(real_cost, label='real')
            ax[i, j].plot(cost_low_n, label='low n')
            ax[i, j].plot(cost_high_n, label='high n')
            ax[i, j].set_title('nu = ' + str(nus[i]) + ', eta = ' + str(etas[j]))
    fig.suptitle('growth')
    fig.text(0.5, 0.04, 'time', ha='center', fontsize=14)
    fig.text(0.04, 0.5, 'cost', va='center', rotation='vertical', fontsize=14)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc='upper right')
    
    fig_ng, ax_ng = plt.subplots(nrows=len(etas), figsize = (8, 8))
    for i in range(len(etas)):
        real_cost_ng = solver.loss_unbalanced_all_local(real_prob_ng[i], costm, reg, reg1, reg2, win_size=win_size, sink=sink)
        p_low_n_ng = np.zeros((T + 1, d))
        p_high_n_ng = np.zeros((T + 1, d))
        for t in range(T + 1):
            p_low_n_ng[t, ] = np.random.multinomial(ns[0], pvals=real_prob_ng[i][t, ]) / ns[0]
            p_high_n_ng[t, ] = np.random.multinomial(ns[1], pvals=real_prob_ng[i][t, ]) / ns[1]
        cost_low_n_ng = solver.loss_unbalanced_all_local(p_low_n_ng, costm, reg, reg1, reg2, win_size=win_size, sink=sink, weight=weight)
        cost_high_n_ng = solver.loss_unbalanced_all_local(p_high_n_ng, costm, reg, reg1, reg2, win_size=win_size, sink=sink, weight=weight)
        ax_ng[i].plot(real_cost_ng, label='real')
        ax_ng[i].plot(cost_low_n_ng, label='low n')
        ax_ng[i].plot(cost_high_n_ng, label='high n')
        ax_ng[i].set_title('eta = ' + str(etas[i]))
    fig_ng.suptitle('no growth')
    fig_ng.text(0.5, 0.04, 'time', ha='center', fontsize=14)
    fig_ng.text(0.04, 0.5, 'cost', va='center', rotation='vertical', fontsize=14)
    fig_ng.legend(*ax_ng[0].get_legend_handles_labels(), loc='upper right')
    
# the following part is for parameter tuning

if do_param_select:
    import matplotlib.pyplot as plt
    
    res_tune = solver.optimal_lambda_ts(real_prob[nu_check, eta_check], costm, reg, grid_min, grid_max, grid_size=n_grid)
    plt.plot(np.linspace(grid_min, grid_max, n_grid + 1)[1:], res_tune['obj_func'])
    plt.xlabel('lambda')
    plt.ylabel('objection function')

###############################################################################
###############################################################################
###############################################################################

if do_sim:
    if data_type == 'g':
        data_all = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        data_mat = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        prob_all = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        prob_cor = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        data_cor = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        cost_all = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        cost_cor = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        tmap_all = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        tmap_cor = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        # cost_marg_all = np.empty((len(ns), len(nus), len(etas)), dtype=object)
        for i in range(len(ns)):
            for j in range(len(nus)):
                for k in range(len(etas)):
                    # data_all[i, j, k] = test_util.dgp(nus[j], etas[k], real_cp, g, d, T, ns[i], M)
                    data_all[i, j, k] = test_util.dgp_from_prob(real_prob[j, k], ns[i], T, d, M)
                    data_mat[i, j, k] = test_util.get_weight_no_ignore_mc(data_all[i, j, k], T, d)
                    prob_all[i, j, k] = test_util.get_weight_no_ignore_mc(data_all[i, j, k], T, d, matformat=False, count=False)
                    prob_cor[i, j, k] = test_util.offset_growth_mc(prob_all[i, j, k], costm, reg, reg1)
                    data_cor[i, j, k] = test_util.dgp_with_prob(prob_cor[i, j, k], ns[i])
                    # cost_all[i, j, k] = test_util.get_ot_unbalanced_cost_mc(prob_all[i, j, k], costm, reg, reg1, reg2, sink=sink)
                    cost_all[i, j, k] = test_util.get_ot_unbalanced_cost_local_mc(prob_all[i, j, k], costm, reg, reg1, reg2, sink=sink, win_size=win_size, weight=weight)
                    # cost_marg_all[i, j, k] = test_util.get_ot_unbalanced_cost_mm_mc(prob_all[i, j, k], costm, reg, reg1, coeff=weight, win_size=win_size)
                    tmap_all[i, j, k] = solver.ot_unbalanced_all_mc(prob_all[i, j, k], costm, reg, reg1, reg2)
                    cost_cor[i, j, k] = test_util.get_ot_unbalanced_cost_local_mc(prob_cor[i, j, k], costm, reg, reg1, reg2, sink=sink, win_size=win_size, weight=weight)
                    tmap_cor[i, j, k] = solver.ot_unbalanced_all_mc(prob_cor[i, j, k], costm, reg, reg1, 50)
        np.save(r'..\results\simulation\multisetting_data.npy', data_all)
        np.save(r'..\results\simulation\multisetting_data_prob.npy', prob_all)
        np.save(r'..\results\simulation\multisetting_data_prob_cor.npy', prob_cor)
        np.save(r'..\results\simulation\multisetting_data_cost' + name_setting + '.npy', cost_all)
        np.save(r'..\results\simulation\multisetting_data_cost_cor' + name_setting + '.npy', cost_cor)
        # np.save(r'..\results\simulation\multisetting_data_cost_mm' + name_setting + '.npy', cost_marg_all)
        np.save(r'..\results\simulation\multisetting_data_tmap' + name_setting + '.npy', tmap_all)
        np.save(r'..\results\simulation\multisetting_data_tmap_cor' + name_setting + '.npy', tmap_cor)
        scipy.io.savemat(r'..\results\simulation\multisetting_data.mat', dict(data=data_mat, ns=ns, zetas=etas, nus=nus, M=M, T=T, p=d))
        scipy.io.savemat(r'..\results\simulation\multisetting_data_cor.mat', dict(data=data_cor, ns=ns, zetas=etas, nus=nus, M=M, T=T, p=d))
    elif data_type == 'ng':
        if switch:
            data_all = np.empty((len(ns), len(pwrs)), dtype=object)
            data_mat = np.empty((len(ns), len(pwrs)), dtype=object)
            prob_all = np.empty((len(ns), len(pwrs)), dtype=object)
            cost_all = np.empty((len(ns), len(pwrs)), dtype=object)
            tmap_all = np.empty((len(ns), len(pwrs)), dtype=object)
            for i in range(len(ns)):
                for j in range(len(pwrs)):
                    data_all[i, j] = test_util.dgp_ng_switch(pwrs[j], real_cp, d, T, ns[i], M)
                    data_mat[i, j] = test_util.get_weight_no_ignore_mc(data_all[i, j], T, d)
                    prob_all[i, j] = test_util.get_weight_no_ignore_mc(data_all[i, j], T, d, matformat=False, count=False)
                    # cost_all[i, j] = test_util.get_ot_unbalanced_cost_mc(prob_all[i, j], costm, reg, reg1, reg2, sink=sink)
                    cost_all[i, j] = test_util.get_ot_unbalanced_cost_local_mc(prob_all[i, j], costm, reg, reg1, reg2, sink=sink, win_size=win_size, weight=weight)
                    # cost_marg_all[i, j] = test_util.get_ot_unbalanced_cost_mm_mc(prob_all[i, j], costm, reg, reg1, coeff=weight, win_size=win_size)
                    tmap_all[i, j] = solver.ot_unbalanced_all_mc(prob_all[i, j], costm, reg, reg1, reg2)
            scipy.io.savemat(r'..\results\simulation\multisetting_data_ng.mat', dict(data=data_mat, ns=ns, pwrs=pwrs, M=M, T=T, p=d))
        else:
            data_all = np.empty((len(ns), len(etas)), dtype=object)
            data_mat = np.empty((len(ns), len(etas)), dtype=object)
            prob_all = np.empty((len(ns), len(etas)), dtype=object)
            cost_all = np.empty((len(ns), len(etas)), dtype=object)
            tmap_all = np.empty((len(ns), len(pwrs)), dtype=object)
            for i in range(len(ns)):
                for j in range(len(etas)):
                    # data_all[i, j] = test_util.dgp_ng(etas[j], real_cp, d, T, ns[i], M)
                    data_all[i, j] = test_util.dgp_from_prob(real_prob_ng[j], ns[i], T, d, M)
                    data_mat[i, j] = test_util.get_weight_no_ignore_mc(data_all[i, j], T, d)
                    prob_all[i, j] = test_util.get_weight_no_ignore_mc(data_all[i, j], T, d, matformat=False, count=False)
                    # cost_all[i, j] = test_util.get_ot_unbalanced_cost_mc(prob_all[i, j], costm, reg, reg1, reg2, sink=sink)
                    cost_all[i, j] = test_util.get_ot_unbalanced_cost_local_mc(prob_all[i, j], costm, reg, reg1, reg2, sink=sink, win_size=win_size, weight=weight)
                    # cost_marg_all[i, j] = test_util.get_ot_unbalanced_cost_mm_mc(prob_all[i, j], costm, reg, reg1, coeff=weight, win_size=win_size)
                    tmap_all[i, j] = solver.ot_unbalanced_all_mc(prob_all[i, j], costm, reg, reg1, reg2)
            scipy.io.savemat(r'..\results\simulation\multisetting_data_ng.mat', dict(data=data_mat, ns=ns, etas=etas, M=M, T=T, p=d))
        np.save(r'..\results\simulation\multisetting_data_ng.npy', data_all)
        np.save(r'..\results\simulation\multisetting_data_prob_ng.npy', prob_all)
        np.save(r'..\results\simulation\multisetting_data_cost_ng' + name_setting + '.npy', cost_all)
        # np.save(r'..\results\simulation\multisetting_data_cost_ng_mm' + name_setting + '.npy', cost_marg_all)
        np.save(r'..\results\simulation\multisetting_data_tmap_ng' + name_setting + '.npy', tmap_all)
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
        tmap_all = np.load(r'..\results\simulation\multisetting_data_tmap' + name_setting + '.npy', allow_pickle=True)
        prob_cor = np.load(r'..\results\simulation\multisetting_data_prob_cor.npy', allow_pickle=True)
        tmap_cor = np.load(r'..\results\simulation\multisetting_data_tmap_cor' + name_setting + '.npy', allow_pickle=True)
        if multimarg:
            cost_all = np.load(r'..\results\simulation\multisetting_data_cost_mm' + name_setting + '.npy', allow_pickle=True)
        else:
            cost_all = np.load(r'..\results\simulation\multisetting_data_cost' + name_setting + '.npy', allow_pickle=True)
        cost_cor = np.load(r'..\results\simulation\multisetting_data_cost_cor' + name_setting + '.npy', allow_pickle=True)
    elif data_type == 'ng':
        data_all = np.load(r'..\results\simulation\multisetting_data_ng.npy', allow_pickle=True)
        prob_all = np.load(r'..\results\simulation\multisetting_data_prob_ng.npy', allow_pickle=True)
        tmap_all = np.load(r'..\results\simulation\multisetting_data_tmap_ng' + name_setting + '.npy', allow_pickle=True)
        if multimarg:
            cost_all = np.load(r'..\results\simulation\multisetting_data_cost_ng_mm' + name_setting + '.npy', allow_pickle=True)
        else:
            cost_all = np.load(r'..\results\simulation\multisetting_data_cost_ng' + name_setting + '.npy', allow_pickle=True)
    if do_tune:
        ot_tune = test_util.ot_analysis_tuning(prob_all, costm, reg, 
                                               file_path=r'..\results\simulation\multisetting_data_' + ('g' if data_type == 'g' else 'ng'),
                                               grid_min=grid_min,
                                               grid_max=grid_max,
                                               grid_size=n_grid,
                                               sink=sink,
                                               win_size=win_size,
                                               weight=weight)
        cost_tune = ot_tune['cost']
        tmap_tune = ot_tune['tmap']
        reg1_tune = ot_tune['reg1']
    else:
        cost_tune = np.load(r'..\results\simulation\multisetting_data_' + ('g' if data_type == 'g' else 'ng') + '_cost_tuning' + name_setting + '.npy', allow_pickle=True)
        tmap_tune = np.load(r'..\results\simulation\multisetting_data_' + ('g' if data_type == 'g' else 'ng') + '_tmap_tuning' + name_setting + '.npy', allow_pickle=True)
        reg1_tune = np.load(r'..\results\simulation\multisetting_data_' + ('g' if data_type == 'g' else 'ng') + '_reg1' + name_setting + '.npy', allow_pickle=True)
        

if do_test:
    if data_type == 'g':
        res_ot = test_util.multisetting_cp_ot_cost(cost_all, T, win_size=win_size)
        np.save(r'..\results\simulation\multisetting_res_ot' + name_setting + '.npy', res_ot)
        res_cor = test_util.multisetting_cp_ot_cost(cost_cor, T, win_size=win_size)
        np.save(r'..\results\simulation\multisetting_res_ot_cor' + name_setting + '.npy', res_cor)
        res_tune = test_util.multisetting_cp_ot_cost(cost_tune, T, win_size=win_size)
        np.save(r'..\results\simulation\multisetting_res_ot_tune' + name_setting + '.npy', res_tune)
    elif data_type == 'ng':
        res_ot = test_util.multisetting_cp_ot_cost_ng(cost_all, T, win_size=win_size)
        np.save(r'..\results\simulation\multisetting_res_ot_ng' + name_setting + '.npy', res_ot)
        res_tune = test_util.multisetting_cp_ot_cost(cost_tune, T, win_size=win_size)
        np.save(r'..\results\simulation\multisetting_res_ot_tune_ng' + name_setting + '.npy', res_tune)
        
  
if do_comp:
    if data_type == 'g':
        
        res_ot = np.load(r'..\results\simulation\multisetting_res_ot' + name_setting + '.npy', allow_pickle=True)
        
        res_tune = np.load(r'..\results\simulation\multisetting_res_ot_tune' + name_setting + '.npy', allow_pickle=True)
        
        # res_cor = np.load(r'..\results\simulation\multisetting_res_ot_cor' + name_setting + '.npy', allow_pickle=True)
        
        # res_mn_raw = scipy.io.loadmat(r'..\results\simulation\multisetting_res_mn.mat')['cpt']
        res_mn_raw = scipy.io.loadmat(r'..\results\simulation\multisetting_res_mn' + ('_cor' if offset else '') + '.mat')['cpt']
        res_mn = test_util.get_res_from_others(res_mn_raw, len(ns), len(nus), len(etas), M, ftype='mat')
                    
        res_ecp_raw = np.load(r'..\results\simulation\multisetting_res_ecp' + ('_cor' if offset else '') + '.npy')
        res_ecp = test_util.get_res_from_others(res_ecp_raw, len(ns), len(nus), len(etas), M, ftype='r')
                    
        # res_wbs_raw = np.load(r'..\results\simulation\multisetting_res_wbs' + ('_cor' if offset else '') + '.npy')
        # res_wbs = test_util.get_res_from_others(res_wbs_raw, len(ns), len(nus), len(etas), M, ftype='r')
                    
        res_all = test_util.prf_table_all(res_ot, res_mn, res_ecp, ns=ns, nus=nus, etas=etas, real_cp=real_cp, index=index, win_size=win_size) 
        res_all.to_csv(r'..\results\simulation\multisetting_res_table' + name_setting + ('_cor' if offset else '') + '.csv')
    
        res_all_cor = test_util.prf_table_all(res_ot, res_cor, res_mn, res_ecp, ns=ns, nus=nus, etas=etas, real_cp=real_cp, index=['OT', 'OT COR', 'MN', 'ECP'], win_size=win_size) 
    
        res_all_tune = test_util.prf_table_all(res_ot, res_tune, res_mn, res_ecp, ns=ns, nus=nus, etas=etas, real_cp=real_cp, index=['OT', 'OTT', 'MN', 'ECP'], win_size=win_size)
    
    elif data_type == 'ng':
        
        n_temp = len(pwrs) if switch else len(etas)
        temps = pwrs.copy() if switch else etas.copy()
        res_ot = np.load(r'..\results\simulation\multisetting_res_ot_ng' + name_setting + '.npy', allow_pickle=True)
        
        res_tune = np.load(r'..\results\simulation\multisetting_res_ot_tune_ng' + name_setting + '.npy', allow_pickle=True)
        
        res_mn_raw = scipy.io.loadmat(r'..\results\simulation\multisetting_res_mn_ng.mat')['cpt']
        res_mn = test_util.get_res_from_others_ng(res_mn_raw, len(ns), n_temp, M, ftype='mat')
        
        res_ecp_raw = np.load(r'..\results\simulation\multisetting_res_ecp_ng.npy')
        res_ecp = test_util.get_res_from_others_ng(res_ecp_raw, len(ns), n_temp, M, ftype='r')
        
        # res_wbs_raw = np.load(r'..\results\simulation\multisetting_res_wbs_ng.npy')
        # res_wbs = test_util.get_res_from_others_ng(res_wbs_raw, len(ns), n_temp, M, ftype='r')
        
        res_all = test_util.prf_table_all_ng(res_ot, res_mn, res_ecp, ns=ns, etas=temps, real_cp=real_cp, switch=switch, index=index, win_size=win_size)
        res_all.to_csv(r'..\results\simulation\multisetting_res_table_ng' + name_setting + '.csv')

        res_all_tune = test_util.prf_table_all_ng(res_ot, res_tune, res_mn, res_ecp, ns=ns, etas=temps, real_cp=real_cp, switch=switch, index=['OT', 'OTT', 'MN', 'ECP'], win_size=win_size)

    pd.set_option('display.max_columns', None)
    print(res_all)

print("--- %s seconds ---" % (time.time() - start_time))