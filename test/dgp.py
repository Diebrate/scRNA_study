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
B = 10

n = 1000
T = 20
d = 10
G = 50
means = np.arange(d) - (d / 2)

nu = 0.25
eta = 1
g = []
change = np.array([-1] * (d // 2) + [1] * (d - d // 2))
change1 = np.exp(eta * change)
change2 = np.exp(eta * change[::-1])

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
        if t + 1 in [5, 10, 15]:
            change = change2 if (t + 1) == 10 else change1
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
