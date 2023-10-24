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

for m in range(M):
    res0_ot = np.load('../results/simulation/compile/res_cp_id1.npy', allow_pickle=True)
    tmp = np.zeros((B, T))
    for b in range(B):
        tmp[b, [res0_ot[b] - 1]] = 1
    res_ot.append(tmp)
    res_ecp.append(np.array(pyreadr.read_r('../results/simulation/compile/test_ecp_id1.RDS')[None]))
    res_mn.append(loadmat('../results/simulation/compile/test_mn_id1.mat')['res'])

res = {'ot': np.vstack(res_ot),
       'ecp': np.vstack(res_ecp),
       'mn': np.vstack(res_mn)}

perf = {method: {'precision': 0, 'recall': 0, 'f-score': 0} for method in res.keys()}


def precision(method):
    pred = res[method]
    if np.sum(pred) > 0:
        return (pred @ cp) / np.sum(pred)
    else:
        return 0


def recall(method):
    pred = res[method]
    return (pred @ cp) / np.sum(cp)


def f_score(method):
    p = perf[method]['precision']
    r = perf[method]['recall']
    return (2 * p * r) / (p + r)


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
