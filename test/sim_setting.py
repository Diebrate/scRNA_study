import numpy as np

np.random.seed(12345)

d = 20
T = 100
n = 250
M = 20

nu = 0.25
eta = 0.25

reg = 1
reg1 = 1
reg2 = 1

data_mc = []
sep = int(d / 2)
B = np.zeros((d, d))

real_cp = [int(per * T) for per in [0.2, 0.4, 0.6, 0.8]]

# B1[:, sep:] = eta
# B2[:, :sep] = eta
# B = np.diag(np.ones(d))

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
    
    # if i < sep:
    #     B[i, (i + 1) % sep] = 0.5 * eta
    #     B[i, min(i + sep, d - 1)] = 0.5 * eta
    # else:
    #     B[i, sep + (i + 1 - sep) % (d - sep)] = 0.5 * eta
    #     B[i, max(0, i - sep)] = 0.5 * eta
    
    B[i, i] = 1 - eta
    
    # if i < sep:
    #     temp = np.repeat([0, eta / (d - sep - 1)], (sep, d - sep))
    #     B[i, ] = temp
    #     B[i, i + sep] = 1 - eta
    # else:
    #     temp = np.repeat([eta / (sep - 1), 0], (sep, d - sep))
    #     B[i, ] = temp
    #     B[i, i - sep] = 1 - eta
    
    # if i == sep - 1:
    #     B[i, :sep] = eta / (sep - 1)
    #     B[i, i] = 1 - eta
    # elif i == d - 1:
    #     B[i, sep:d - 1] = eta / (d - sep - 1)
    #     B[i, i] = 1 - eta
        
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

p = np.zeros((T + 1, d))
p[0, ] = np.repeat(1 / d, d)
g_rate = np.ones((T, d))
g = np.ones((T, d))
for t in range(T):
    ind = t
    g[t, ] = np.sin(np.arange(ind, ind + d) * np.pi / d)
    # g[t, ] = np.concatenate((np.sin(2 * np.arange(ind, ind + sep) * np.pi / sep), 
    #                          np.cos(2 * np.arange(ind, ind + d - sep) * np.pi / (d - sep))))
for t in range(T):
    # if t % 2 == 0:
    #     g_rate[t, ] = g ** nu
    # else:
    #     g_rate[t, ] = np.ones(d)
    g_rate[t, ] = np.exp(g[t, ] * nu)
for t in range(1, T + 1):
    p[t, ] = p[t - 1, ] * g_rate[t - 1, ]
    p[t, ] = p[t, ] / np.sum(p[t, ])
    if t - 1 in real_cp:
        # if theta(t) == 1:
        #     p[t, ] = B1.transpose() @ p[t, ]
        # else:
        #     p[t, ] = B2.transpose() @ p[t, ]
        p[t, ] = B.T @ p[t, ]
        # p[t, ] = np.repeat(1 / d, d)
        # p_ind = np.argmax(p[t, ])
        # p[t, :p_ind] += p[t, p_ind] * eta / (d - 1)
        # p[t, p_ind + 1:] += p[t, p_ind] * eta / (d - 1)
        # p[t, p_ind] *= 1 - eta

data_mc = []        
for m in range(M):
    data_temp = np.zeros((T + 1, n), dtype=int)
    for t in range(T + 1):
        data_temp[t, ] = np.random.choice(np.arange(d), size=n, p=p[t, ])
    data_mc.append(data_temp)
                    
data_mc = np.array(data_mc)

import test_util
# res = test_util.est_check_unbalanced_fast(data_mc[5, 0, :], data_mc[5, 1, :], p[5, :], g_rate[5, :], B, costm, reg, reg1, reg2, d, n_conv=5)
# check = test_util.power_test_unbalanced_fast(p[5, :], g_rate[5, :], B, costm, reg, reg1, reg2, d)
# ot_cusum = test_util.run_cusum_ot(data_mc[10], d, costm, reg, reg1, reg2, balanced=False, sink=True, full_return=True)
cp, cost = test_util.run_cusum_ot(data_mc[18], d, costm, reg, reg1, reg2, balanced=False, sink=True, n_conv=None, full_return=True)
cp_cs, st, gt, diff = test_util.run_cusum(data_mc[18], d, return_ll=True)

# print(res)
# print(check['power'])
# print(ot_cusum)
print(cp)
print(cp_cs)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(cost)
plt.figure()
plt.plot(st, label='st')
plt.plot(gt, label='gt')
plt.plot(diff, label='diff')
plt.legend()





















