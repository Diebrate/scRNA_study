import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import solver
import test_util

import time
start_time = time.time()

np.random.seed(1234)

# test data
n = 100
m = 100
n_null = 100
n_sim = 1000

k = 5
r = np.arange(1, k + 1)
c = r.copy()
r, c = r / sum(r), c / sum(c)
a = np.zeros((n_null, n))
b = np.zeros((n_null, m))

px = np.repeat(1, k)
py = np.arange(1, k + 1)
px, py = px / sum(px), py / sum(py)
txy = np.random.rand(k, k)
txy = np.diag(1 / np.sum(txy, axis=1)) @ txy
x = np.zeros((n_null, n))
y = np.zeros((n_null, m))

for i in range(n_null):
    a[i] = np.random.choice(range(k), size=n, p=r)
    b[i] = np.random.choice(a[i], size=m)
for i in range(n_null):
    x[i] = np.random.choice(range(k), size=n, p=px)
    y_temp = np.zeros(n)
    for j in range(n):
        y_temp[j] = np.random.choice(range(k), size=1, p=txy[int(x[i, j])])
    y[i] = np.random.choice(y_temp, size=m)

a = np.array(a, dtype=int)
b = np.array(b, dtype=int)
x = np.array(x, dtype=int)
y = np.array(y, dtype=int)

costm = np.random.rand(k, k) 
costm = costm @ costm.transpose()
np.fill_diagonal(costm, 0)
reg = 1
reg1 = 1
reg2 = 50

index = 11

balanced = False
sink = True
fast = True

if fast:
    d1 = np.zeros((k, n_null))
    d2 = np.zeros((k, n_null))
    for i in range(n_null):
        d1[:, i] = test_util.get_weight(a[i], k)
        d2[:, i] = test_util.get_weight(b[i], k)
    if balanced:
        if sink:
            zs_null = solver.sink_loss_balanced_all(d1, d2, costm, reg)
        else:
            zs_null = solver.wass_loss_balanced_all(d1, d2, costm, reg)
    else:
        if sink:
            zs_null = solver.sink_loss_unbalanced_all(d1, d2, costm, reg, reg1, reg2)
        else:
            zs_null = solver.wass_loss_unbalanced_all(d1, d2, costm, reg, reg1, reg2)
    
    ind = 1
    ps_null = []
    for i, j in zip(a, b):
        if balanced:
            ps_null.append(test_util.perm_test_balanced_fast(i, j, costm, reg, sink=sink, n_sim=n_sim))
        else:
            ps_null.append(test_util.perm_test_unbalanced_fast(i, j, costm, reg, reg1, reg2, sink=sink, n_sim=n_sim))
        print('finished iteration ' + str(ind))
        ind += 1
    
    ind = 1
    ps_alt = []
    for i, j in zip(x, y):
        if balanced:    
            ps_alt.append(test_util.perm_test_balanced_fast(i, j, costm, reg, sink=sink, n_sim=n_sim))
        else:
            ps_alt.append(test_util.perm_test_unbalanced_fast(i, j, costm, reg, reg1, reg2, sink=sink, n_sim=n_sim))
        print('finished iteration ' + str(ind))
        ind += 1
    
    if balanced:
        res = test_util.perm_test_balanced_fast(a[index], b[index], costm, reg, sink=sink, n_sim=n_sim, fullreturn=True)
    else:
        res = test_util.perm_test_unbalanced_fast(a[index], b[index], costm, reg, reg1, reg2, sink=sink, n_sim=n_sim, fullreturn=True)
    zs_perm = res['zs']
    
    if balanced:
        res_alt = test_util.perm_test_balanced_fast(x[index], y[index], costm, reg, sink=sink, n_sim=n_sim, fullreturn=True)
    else:
        res_alt = test_util.perm_test_unbalanced_fast(x[index], y[index], costm, reg, reg1, reg2, sink=sink, n_sim=n_sim, fullreturn=True)
    zs_perm_alt = res_alt['zs']

else:
    # null distribution
    zs_null = []
    for i, j in zip(a, b):
        p1_temp = test_util.get_weight(i, k)
        p2_temp = test_util.get_weight(j, k)
        if balanced:
            zs_null.append(solver.sink_loss(p1_temp, p2_temp, costm, reg))
        else:
            zs_null.append(solver.sink_loss_unbalanced(p1_temp, p2_temp, costm, reg, reg1, reg2))
        
    # type 1 error check
    check_end = 10
    ps_null = []
    ind = 0
    for i, j in zip(a[:check_end], b[:check_end]):
        if balanced:
            ps_null.append(test_util.perm_test1(i, j, costm, reg))
        else:
            ps_null.append(test_util.perm_test1_unbalanced(i, j, costm, reg, reg1, reg2))
        print('finished iteration ' + str(ind))
        ind += 1
    ps_null = np.array(ps_null)
        
    ps_alt = []
    ind = 0
    for i, j in zip(x[:check_end], y[:check_end]):
        if balanced:    
            ps_alt.append(test_util.perm_test1(i, j, costm, reg))
        else:
            ps_alt.append(test_util.perm_test1_unbalanced(i, j, costm, reg, reg1, reg2))
        print('finished iteration ' + str(ind))
        ind += 1
    ps_alt = np.array(ps_alt)
    
    if balanced:
        res = test_util.perm_test1(a[index], b[index], costm, reg, n_sim=n_null, fullreturn=True)
    else:
        res = test_util.perm_test1_unbalanced(a[index], b[index], costm, reg, reg1, reg2, n_sim=n_null, fullreturn=True)
    zs_perm = res['zs']
    
    if balanced:
        res_alt = test_util.perm_test1(x[index], y[index], costm, reg, n_sim=n_null, fullreturn=True)
    else:
        res_alt = test_util.perm_test1_unbalanced(x[index], y[index], costm, reg, reg1, reg2, n_sim=n_null, fullreturn=True)
    zs_perm_alt = res_alt['zs']

df = {'value': np.concatenate((zs_null, zs_perm)),
      'source': np.concatenate((np.repeat('null', n_null), np.repeat('perm', n_sim)))}

df_alt = {'value': np.concatenate((zs_null, zs_perm_alt)),
          'source': np.concatenate((np.repeat('null', n_null), np.repeat('perm', n_sim)))}

plt.figure(1)
sns.kdeplot(data=df, x='value', hue='source', common_norm=False, alpha=.5, fill=True)

plt.figure(2)
sns.kdeplot(data=df_alt, x='value', hue='source', common_norm=False, alpha=.5, fill=True)

plt.figure(3)
sns.kdeplot(ps_null, clip=(0, 1))

plt.figure(4)
sns.kdeplot(ps_alt, clip=(0, 1))


print("--- %s seconds ---" % (time.time() - start_time))

