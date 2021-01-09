import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import solver
import test_util

import time
start_time = time.time()

# test data
n = 200
m = 200
n_null = 1000

k = 5
r = np.random.rand(k)
c = r.copy()
r, c = r / sum(r), c / sum(c)
a = np.zeros((n_null, n))
b = np.zeros((n_null, m))

px = np.random.rand(k)
py = px[::-1]
px, py = px / sum(px), py / sum(py)
x = np.zeros((n_null, n))
y = np.zeros((n_null, m))

for i in range(n_null):
    a[i] = np.random.choice(range(k), size=n, p=r)
    b[i] = np.random.choice(range(k), size=m, p=c)
    x[i] = np.random.choice(range(k), size=n, p=px)
    y[i] = np.random.choice(range(k), size=m, p=py)

a = np.array(a, dtype=int)
b = np.array(b, dtype=int)
x = np.array(x, dtype=int)
y = np.array(y, dtype=int)

costm = np.random.rand(k, k) * 10
costm = costm @ costm.transpose()
np.fill_diagonal(costm, 0)
reg = 10
reg1 = 1
reg2 = 1

index = 20

balanced = True

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
    
ps_alt = []
ind = 0
for i, j in zip(x[:check_end], y[:check_end]):
    if balanced:    
        ps_alt.append(test_util.perm_test1(i, j, costm, reg))
    else:
        ps_alt.append(test_util.perm_test1_unbalanced(i, j, costm, reg, reg1, reg2))
    print('finished iteration ' + str(ind))
    ind += 1

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

if balanced:
    df = {'loss_null': zs_null, 'loss_perm': zs_perm}
else:
    df = {'loss_null': zs_null, 'loss_perm': zs_perm}
df = pd.DataFrame(data=df)

plt.figure(1)
sns.kdeplot(data=df)

df_alt = {'loss_null': zs_null, 'loss_perm': zs_perm_alt}
df_alt = pd.DataFrame(data=df_alt)

plt.figure(2)
sns.kdeplot(data=df_alt)


print("--- %s seconds ---" % (time.time() - start_time))

