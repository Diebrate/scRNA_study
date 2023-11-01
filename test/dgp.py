import numpy as np
import pandas as pd
import phate

test_mode = True
if test_mode:
    m = 0
    B = 3
else:
    import sys
    m = int(sys.argv[1])
    B = 10

rng = np.random.default_rng(m + 12345)

n = 1000
T = 50
d = 10
G = 50
means = np.arange(d) - (d / 2)

nus = [0.1, 0.25]
etas = [0.5, 1]
# nu = 0.1
# eta = 1
g = []
change = np.array([-1] * (d // 2) + [1] * (d - d // 2))

data_all = []

for k in range(B):
    for nu in nus:
        for eta in etas:
            change1 = np.exp(eta * change)
            change2 = np.exp(eta * change[::-1])
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
                if t + 1 in [10, 20, 30, 40]:
                    change = change2 if (t + 1) % 20 == 0 else change1
                    q0 = (change * q0) / np.sum(change * q0)
                Q.append(q0)
                n0 = 0
                n_mn = rng.multinomial(n, Q[t])
                for i in range(d):
                    X[t+1, n0:(n0 + n_mn[i]), :G] = rng.multivariate_normal(mean=np.repeat(means[i], G), cov=np.diag(np.ones(G)), size=n_mn[i])
                    X[t+1, n0:(n0 + n_mn[i]), G] = i
                    n0 += n_mn[i]

            columns = ['x' + str(i + 1) for i in range(G)] + ['type']

            df = []
            for t in range(T + 1):
                df.append(pd.DataFrame(X[t, :, :], columns=columns))
            data = pd.concat(df, ignore_index=True)

            phate_op = phate.PHATE(n_jobs=-2, n_pca=20)
            Y_phate = phate_op.fit_transform(data[['x' + str(i + 1) for i in range(G)]])

            data[['phate1', 'phate2']] = Y_phate
            data['time'] = np.repeat(np.arange(T + 1), n)
            data['batch'] = k
            data['nu'] = 'low' if nu == nus[0] else 'high'
            data['eta'] = 'low' if eta == etas[0] else 'high'

            data_all.append(data)

            print('finished batch ' + str(k) + ' for m = ' + str(m) + '\nnu = ' + str(nu) + ' eta = ' + str(eta))

data_all = pd.concat(data_all, ignore_index=True)
if test_mode:
    data_all.to_csv('../data/simulation_data/test_sample.csv')
else:

    data_all.to_csv('../data/simulation_data/simulation_id' + str(m) + '.csv')
