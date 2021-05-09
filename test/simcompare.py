import numpy as np
import test_util
import scipy.io
import pandas as pd

import time
start_time = time.time()

do_sim = True
growth = False
do_comp = False

save_res = False  # for cluster script
file_name = None

np.random.seed(12345)

d = 16
T = 20
n = 1000
M = 10

pwr = 1.25
nu = 0.1
eta = 0.75
delta = 0.5

balanced = False
sink = True 
perm = True
n_conv = 10

reg = 1
reg1 = 1
reg2 = 1

order = None

costm = np.zeros((d, d))
B = np.ones((d, d))
sep = int(d / 2)
for i in range(d):
    for j in range(d):
        if i < d / 2:
            if j >= d / 2:
                B[i, j] = eta / (d - sep)
            else:
                B[i, j] = 0
        else:
            if j < d / 2:
                B[i, j] = eta / sep
            else:
                B[i, j] = 0
    B[i, i] = 1 - eta
for i in range(d):
    for j in range(i, d):
        if i < d / 2:
            if j >= d / 2:
                costm[i, j] = 5
                costm[j, i] = costm[i, j]
            else:
                costm[i, j] = 1
                costm[j, i] = costm[i, j]
        else:
            if j < d / 2:
                costm[i, j] = 5
                costm[j, i] = costm[i, j]
            else:
                costm[i, j] = 1
                costm[j, i] = costm[i, j]
np.fill_diagonal(costm, 0)

# B = B1 = B2 = np.random.rand(d, d)
# B = np.ones((d, d))

B = np.diag(1 / B.sum(axis=1)) @ B
B1 = B.copy()
B2 = B.copy()

def theta(t):
    if 0 <= t and t <= 0.25 * T:
        r = 0
    elif t > 0.25 * T and t <= 0.5 * T:
        r = 1
    elif t > 0.5 * T and t <= 0.75 * T:
        r = 0
    else:
        r = 1
    return r

if do_sim:
    data_mc = []
    if not growth:
        for m in range(M):
            data_temp = np.zeros((T + 1, n), dtype=int)
            for t in range(T + 1):
                p_temp = np.ones(d)
                if theta(t) == 0:
                    p_temp[:d // 2] = pwr
                else:
                    p_temp[d // 2:] = pwr
                p_temp = p_temp / np.sum(p_temp)
                data_temp[t, ] = np.random.choice(np.arange(d), size=n, p=p_temp)
            data_mc.append(data_temp)
    else:
        for m in range(M):
            data_temp = np.zeros((T + 1, n), dtype=int)
            p_temp = np.repeat(1 / d, d)
            data_temp[0, ] = np.random.choice(np.arange(d), size=n, p=p_temp)
            for t in range(1, T + 1):
                g_rate = np.ones(d)
                if theta(t) == 0:
                    g_rate[:sep] = np.exp(np.random.normal(loc=0, scale=1, size=sep) * nu + delta)
                    g_rate[sep:] = np.exp(np.random.normal(loc=0, scale=1, size=sep) * nu)
                else:
                    g_rate[:sep] = np.exp(np.random.normal(loc=0, scale=1, size=sep) * nu)
                    g_rate[sep:] = np.exp(np.random.normal(loc=0, scale=1, size=sep) * nu + delta)
                p_temp = p_temp * g_rate
                p_temp = p_temp / np.sum(p_temp)
                if theta(t) != theta(t - 1):
                    if theta(t) == 1:
                        p_temp = B1.transpose() @ p_temp
                    else:
                        p_temp = B2.transpose() @ p_temp
                data_temp[t, ] = np.random.choice(np.arange(d), size=n, p=p_temp)
            data_mc.append(data_temp)
                    
    data_mc = np.array(data_mc)
        
    # # single test
    # res = test_util.cp_detection(data_mc[0], d, costm, reg=reg, reg1=reg1, reg2=reg2, balanced=False, sink=True)
    
    # monte carlo simulation
    res_mc = []
    for m in range(M):
        res_mc.append(test_util.cp_detection(data_mc[m], d, costm, reg=reg, reg1=reg1, reg2=reg2, balanced=balanced, sink=sink, order=order, perm=perm, n_conv=n_conv))
        print('finished iteration ' + str(m))
    # ps = test_util.cp_detection_mc(data_mc, d, costm, reg=reg, reg1=reg1, reg2=reg2, balanced=False, sink=True, order=order)

    # cusum test
    res_cs = []
    for m in range(M):
        res_cs.append(test_util.run_cusum(data_mc[m], d))
    res_cs = np.array(res_cs, dtype=object)

if do_comp:
    real = np.array([5, 10, 15])
    
    ns = [250, 500, 750, 1000]
    pwrs = [1.25, 1.5, 1.75, 2, 2.5]
    
    res_total = {}
    
    for n0 in ns:
        res_n0 = []
        for pwr0 in pwrs:  
            ot_raw = np.load(r'..\results\simulation\mc500_unbal_sink_' + str(n0) + '_1_1_50_o' + str(pwr0).replace('.', '_') + '.npy')
            ot_res = {'oe': [], 'ue': [], 'e': []}
            for a in ot_raw:
                t_temp = np.arange(T)[a <= 0.05]
                ot_res['oe'].append(test_util.get_oe(real, t_temp, T=T))
                ot_res['ue'].append(test_util.get_ue(real, t_temp, T=T))
                ot_res['e'].append(test_util.get_e(real, t_temp))
            
            mn_raw = scipy.io.loadmat(r'..\results\simulation\mc1000_o' + str(pwr0).replace('.', '_') + '.mat')['th'][np.argmax([np.array(ns) == n0])]
            mn_res = {'oe': [], 'ue': [], 'e': []}
            for a in mn_raw:
                t_temp = a.flatten().astype(int)
                mn_res['oe'].append(test_util.get_oe(real, t_temp, T=T))
                mn_res['ue'].append(test_util.get_ue(real, t_temp, T=T))
                mn_res['e'].append(test_util.get_e(real, t_temp))
                
            cs_raw = np.load(r'..\results\simulation\mc1000_cusum_' + str(n0) + '_o' + str(pwr0).replace('.', '_') + '.npy', allow_pickle=True)
            cs_res = {'oe': [], 'ue': [], 'e': []}
            for a in cs_raw:
                t_temp = a.flatten().astype(int)
                cs_res['oe'].append(test_util.get_oe(real, t_temp, T=T))
                cs_res['ue'].append(test_util.get_ue(real, t_temp, T=T))
                cs_res['e'].append(test_util.get_e(real, t_temp))
                
            res_sum = pd.DataFrame(data={'OE': np.array([np.mean(ot_res['oe']),
                                                         np.mean(mn_res['oe']),
                                                         np.mean(cs_res['oe'])]),
                                         'UE': np.array([np.mean(ot_res['ue']),
                                                         np.mean(mn_res['ue']),
                                                         np.mean(cs_res['ue'])]),
                                         '#E': np.array([np.mean(ot_res['e']),
                                                         np.mean(mn_res['e']),
                                                         np.mean(cs_res['e'])])},
                                   index=['OT', 'MN', 'CUSUM'])
            res_n0.append(res_sum)
        res_total['n='+str(n0)] = pd.concat(res_n0, axis='columns', keys=['w=' + str(p) for p in pwrs])
    
    res_total = pd.concat(res_total)    
    
if save_res:
    np.save(file_name, np.array([a['ps'] for a in res_mc], dtype=object))
    
print(np.array([a['res'] for a in res_mc], dtype=object))
print(res_cs)
    
print("--- %s seconds ---" % (time.time() - start_time))
























