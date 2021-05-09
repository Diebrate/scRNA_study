import numpy as np
import pandas as pd
import test_util

import time
start_time = time.time()

np.random.seed(12345)

d = 10
T = 20
n = 1000
M = 100

nu = 0.25
eta = 0.75
delta = 0

reg = 1
reg1 = 1
reg2 = 1

balanced = False
sink = True
perm = False
n_conv = 5

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

def param_str(*param):
    s = ''
    for p in param:
        s = s + str(p).replace('.', '_') + '_'
    return s[:-1]

file_name_mc = 'mc' + str(M) + '_growth_' + ('bal_' if balanced else 'unbal_') + ('sink_' if sink else 'unsink_') + param_str(n, reg, reg1, reg2, nu, eta, delta) + '.npy'
file_name_cs = 'mc' + str(M) + '_growth_' +  '_cusum_' + param_str(n, nu, eta, delta) + '.npy'

data_mc = []
sep = int(d / 2)
B = np.ones((d, d))

# B1[:, sep:] = eta
# B2[:, :sep] = eta

costm = np.zeros((d, d))
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
                costm[i, j] = 4.5
                costm[j, i] = costm[i, j]
            else:
                costm[i, j] = 0.5
                costm[j, i] = costm[i, j]
        else:
            if j < d / 2:
                costm[i, j] = 4.5
                costm[j, i] = costm[i, j]
            else:
                costm[i, j] = 0.5
                costm[j, i] = costm[i, j]
np.fill_diagonal(costm, 0)

# B = B1 = B2 = np.random.rand(d, d)
# B = np.ones((d, d))

B = np.diag(1 / B.sum(axis=1)) @ B

p = np.zeros((T + 1, d))
p[0, ] = np.repeat(1 / d, d)
g_rate = np.zeros((T, d))
for t in range(1, T + 1):
    if theta(t) == 0:
        g_rate[t - 1, :sep] = np.exp(np.random.normal(loc=0, scale=1, size=sep) * nu + delta)
        g_rate[t - 1, sep:] = np.exp(np.random.normal(loc=0, scale=1, size=sep) * nu)
    else:
        g_rate[t - 1, :sep] = np.exp(np.random.normal(loc=0, scale=1, size=sep) * nu)
        g_rate[t - 1, sep:] = np.exp(np.random.normal(loc=0, scale=1, size=sep) * nu + delta)
    p[t, ] = p[t - 1, ] * g_rate[t - 1, ]
    p[t, ] = p[t, ] / np.sum(p[t, ])
    if theta(t) != theta(t - 1):
        # if theta(t) == 1:
        #     p[t, ] = B1.transpose() @ p[t, ]
        # else:
        #     p[t, ] = B2.transpose() @ p[t, ]
        p[t, ] = B.transpose() @ p[t, ]

data_mc = []        
for m in range(M):
    data_temp = np.zeros((T + 1, n), dtype=int)
    for t in range(T + 1):
        data_temp[t, ] = np.random.choice(np.arange(d), size=n, p=p[t, ])
    data_mc.append(data_temp)
                    
data_mc = np.array(data_mc)

# monte carlo simulation
# res_mc = []
# for m in range(M):
#     res_mc.append(test_util.cp_detection(data_mc[m], d, costm, reg=reg, reg1=reg1, reg2=reg2, balanced=balanced, sink=sink, perm=perm, n_conv=n_conv))
#     print('finished iteration ' + str(m))

res_otcs = []
for m in range(M):
    res_otcs.append(test_util.cp_detection_ot(data_mc[m], d, costm, reg, reg1, reg2, balanced=balanced, sink=sink, n_conv=n_conv)[0])
res_otcs = np.array(res_otcs)

# cusum test
res_cs = []
for m in range(M):
    res_cs.append(test_util.run_cusum(data_mc[m], d))
res_cs = np.array(res_cs, dtype=object)
    
# np.save(file_name_mc, np.array([a['ps'] for a in res_mc], dtype=object))
# np.save(file_name_cs, res_cs)

# print(np.array([a['res'] for a in res_mc], dtype=object))
# print('########################')
# print(res_otcs)
# print('########################')
# print(res_cs)
# print('########################')
# print(np.array([a['ps'] for a in res_mc]))

cp = np.array([0.25, 0.5, 0.75]) * T
cp = cp.astype(int)
nus = [0, 0.05, 0.1, 0.25, 0.5]
etas = [0.25, 0.5, 0.75, 1]
ns = [250, 500, 750, 1000]
seed = 99999

df_sum = test_util.sum_table(res_otcs, res_cs, real_cp=cp, index=['OT', 'CS'])
pd.set_option('display.max_columns', None)
print(df_sum)
    
print("--- %s seconds ---" % (time.time() - start_time))

