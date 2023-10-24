import numpy as np
import pandas as pd
import pyreadr
from scipy.io import loadmat

M = 100
B = 50
T = 30

cp = np.zeros(30)
cp[[9, 19]] = 1

res_ot = []
res_ecp = []
res_mn = []

for m in range(1, M + 1):
    res_ot.append(np.load('../results/simulation/compile/res_cp_id' + str(m) + '.npy', allow_pickle=True))
    res_ecp.append(np.array(pyreadr.read_r('../results/simulation/compile/test_ecp_id' + str(m) + '.RDS')[None]))
    res_mn.append(loadmat('../results/simulation/compile/test_mn_id' + str(m) + '.mat')['res'])

res = {'ot': np.vstack(res_ot),
       'ecp': np.vstack(res_ecp),
       'mn': np.vstack(res_mn)}

perf = {method: {'precision': 0, 'recall': 0, 'f-score': 0} for method in res.keys()}


def precision(method):
    pred = res[method]
    x = (pred @ cp) / np.sum(pred, axis=1)
    x[np.sum(pred, axis=1) == 0] = 0
    return x


def recall(method):
    pred = res[method]
    return (pred @ cp) / np.sum(cp)


def f_score(method):
    p = perf[method]['precision']
    r = perf[method]['recall']
    x = (2 * p * r) / (p + r)
    x[(p == 0) & (r == 0)] = 0
    return x


for method in res.keys():
    perf[method]['precision'] = precision(method)
    perf[method]['recall'] = recall(method)
    perf[method]['f-score'] = f_score(method)

perf_mean = pd.DataFrame(index=res.keys(), columns=['precision', 'recall', 'f-score'])
perf_std = pd.DataFrame(index=res.keys(), columns=['precision', 'recall', 'f-score'])

for method in res.keys():
    for metric in ['precision', 'recall', 'f-score']:
        perf_mean.loc[method, metric] = np.mean(perf[method][metric])
        perf_std.loc[method, metric] = np.std(perf[method][metric])
