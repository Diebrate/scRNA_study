import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import phate

import test_util
import solver

rng = np.random.default_rng(999)
B = 5

n = 1000
T = 30
d = 4
G = 50
means = [-2, -1, 1, 2]

nu = 0.1
eta = 1
g = []
change1 = np.array([np.exp(eta), np.exp(-eta), np.exp(eta), np.exp(-eta)])
change2 = np.array([np.exp(-eta), np.exp(eta), np.exp(-eta), np.exp(eta)])

data_all = []

for k in range(B):
    # Q = [np.arange(1, d + 1) / np.arange(1, d + 1).sum()]
    Q = [np.ones(d) / d]
    X = np.zeros((T + 1, n, G + 1))
    n0 = 0
    n_mn = rng.multinomial(n, Q[0])
    for i in range(d):
        X[0, n0:(n0 + n_mn[i]), :G] = rng.multivariate_normal(mean=np.repeat(means[i], G), cov=np.diag(np.ones(G)), size=n_mn[i])
        X[0, n0:(n0 + n_mn[i]), G] = i
        n0 += n_mn[i]

    for t in range(T):
        g0 = np.exp(nu * np.sin(np.pi * (t + np.arange(d)) / d))
        g.append(g0)
        q0 = Q[-1] * g0 / np.sum(Q[-1] * g0)
        if t + 1 in [10, 20]:
            change = change1 if (t + 1) == 10 else change2
            q0 = (change * q0) / np.sum(change * q0)
        Q.append(q0)
        n0 = 0
        n_mn = rng.multinomial(n, Q[t])
        for i in range(d):
            X[t+1, n0:(n0 + n_mn[i]), :G] = rng.multivariate_normal(mean=np.repeat(means[i], G), cov=np.diag(np.ones(G)), size=n_mn[i])
            X[t+1, n0:(n0 + n_mn[i]), G] = i
            n0 += n_mn[i]

    columns = ['x' + str(i + 1) for i in range(G)] + ['type']
    data = pd.DataFrame(columns=columns)

    df = []
    for t in range(T + 1):
        df.append(pd.DataFrame(X[t, :, :], columns=columns))
    data = pd.concat(df, ignore_index=True)

    phate_op = phate.PHATE(n_jobs=-2, n_pca=20)
    Y_phate = phate_op.fit_transform(data[['x' + str(i + 1) for i in range(G)]])

    data[['phate1', 'phate2']] = Y_phate
    data['time'] = np.repeat(np.arange(T + 1), n)
    data['batch'] = k

    data_all.append(data)

    print('finished batch ' + str(k))

data_all = pd.concat(data_all, ignore_index=True)

res = np.zeros((B, T))

eps = 0.001
reg = 1

for k in range(B):

    data = data_all[data_all.batch == k]

    centroids = data[['phate1', 'phate2', 'type']].groupby('type').mean().to_numpy()
    costm = test_util.get_cost_matrix(centroids, centroids, dim=2)
    probs = np.zeros((T + 1, d))
    for t in range(T + 1):
        p = data['type'][data['time'] == t].value_counts(normalize=True).sort_index().to_numpy()
        probs[t, :] = p

    tmap = solver.ot_unbalanced_all(probs[:-1, ].T, probs[1:, ].T, costm, reg=eps, reg1=reg, reg2=50)
    cost = []
    for t in range(T):
        phat = tmap[t].sum(axis=1)
        c = np.sum(tmap[t] * costm) + reg * np.sum(phat * np.log(phat / probs[t]))
        cost.append(c)
    cp = test_util.get_cp_from_cost(cost, win_size=1)
    if len(cp) > 0:
        res[k, cp - 1] = 1

    print('b = ' + str(k))

    # sanity check
    plt.figure()
    plt.plot(cost)
