import numpy as np
import pandas as pd
import phate

use_sys = True
if use_sys:
    import sys
    m = int(sys.argv[1])
else:
    m = 0

rng = np.random.default_rng(m + 12345)
B = 5

n = 1000
T = 30
d = 4
G = 50
means = [-2, -1, 1, 2]

nu = 0.25
eta = 0.5
g = []
change = np.array([np.exp(eta), np.exp(-eta), np.exp(eta), np.exp(-eta)])

data_all = []

for k in range(B):
    Q = [np.arange(1, d + 1) / np.arange(1, d + 1).sum()]
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

    print('finished batch ' + str(k) + ' for m = ' + str(m))

data_all = pd.concat(data_all, ignore_index=True)
data_all.to_csv('../data/simulation_data/simulation_id' + str(m) + '.csv')
