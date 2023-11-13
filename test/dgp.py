import numpy as np
import pandas as pd
import phate

test_mode = False
if test_mode:
    m = 0
    B = 2
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

ns = [1000, 2000]
nus = [0.1, 0.25]
etas = [0.5, 1]
cp_vec = np.array([-1] * (d // 2) + [1] * (d - d // 2))

for n in ns:

    for nu in nus:

        for eta in etas:

            print(f'starting n = {n}, nu = {nu}, eta = {eta}')

            change1 = np.exp(eta * cp_vec)
            change2 = np.exp(eta * cp_vec[::-1])
            Q = [np.ones(d) / d]
            for t in range(T):
                g0 = np.exp(nu * np.sin(np.pi * (t + np.arange(d)) / d))
                q0 = Q[-1] * g0 / np.sum(Q[-1] * g0)
                if t + 1 in [10, 20, 30, 40]:
                    change = change2 if (t + 1) % 20 == 0 else change1
                    q0 = (change * q0) / np.sum(change * q0)
                Q.append(q0)

            n_type = 'low' if n == ns[0] else 'high'
            nu_type = 'low' if nu == nus[0] else 'high'
            eta_type = 'low' if eta == etas[0] else 'high'

            data_all = []

            for k in range(B):

                X = np.zeros((T + 1, n, G + 1))
                n0 = 0
                n_mn = rng.multinomial(n, Q[0])
                for i in range(d):
                    X[0, n0:(n0 + n_mn[i]), :G] = rng.multivariate_normal(mean=np.repeat(means[i], G), cov=np.diag(np.ones(G)), size=n_mn[i])
                    X[0, n0:(n0 + n_mn[i]), G] = i
                    n0 += n_mn[i]

                for t in range(T):
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

                data_all.append(data)

                print(f'finished batch {k}, n = {n}, nu = {nu}, eta = {eta}')

            data_all = pd.concat(data_all, ignore_index=True)

            data_all['n'] = n_type
            data_all['nu'] = nu_type
            data_all['eta'] = eta_type

            if test_mode:
                data_all.to_csv(f'../data/simulation_data/test_sample_{n_type}_n_{nu_type}_nu_{eta_type}_eta.csv')
            else:
                data_all.to_csv(f'../data/simulation_data/simulation_id{m}_{n_type}_n_{nu_type}_nu_{eta_type}_eta.csv')

print(f'finished m = {m}')