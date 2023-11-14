import numpy as np
import phate
import test_util

rng = np.random.default_rng(2023)

T = 50
G = 50
d = 10
n = 3000

nus = [0.1, 0.25]
etas = [0.5, 1]

means = np.arange(d) - (d / 2)
cp_vec = np.array([-1] * (d // 2) + [1] * (d - d // 2))

for nu in nus:

    for eta in etas:

        nu_type = 'low' if nu == nus[0] else 'high'
        eta_type = 'low' if eta == etas[0] else 'high'

        Q = np.ones((T + 1, d)) / d
        change1 = np.exp(eta * cp_vec)
        change2 = np.exp(eta * cp_vec[::-1])
        for t in range(T):
            g0 = np.exp(nu * np.sin(np.pi * (t + np.arange(d)) / d))
            q0 = Q[t] * g0 / np.sum(Q[t] * g0)
            if t in [10, 20, 30, 40]:
                change = change2 if t % 20 == 0 else change1
                q0 = (change * q0) / np.sum(change * q0)
            Q[t + 1, :] = q0

        X = np.zeros(((T + 1) * n, G + 1))
        n0 = 0
        n_mn = rng.multinomial(n, Q[0])
        for i in range(d):
            X[n0:(n0 + n_mn[i]), :G] = rng.multivariate_normal(mean=np.repeat(means[i], G),
                                                               cov=np.diag(np.ones(G)),
                                                               size=n_mn[i])
            X[n0:(n0 + n_mn[i]), G] = i
            n0 += n_mn[i]

        for t in range(T):
            n_mn = rng.multinomial(n, Q[t])
            for i in range(d):
                X[n0:(n0 + n_mn[i]), :G] = rng.multivariate_normal(mean=np.repeat(means[i], G),
                                                                   cov=np.diag(np.ones(G)),
                                                                   size=n_mn[i])
                X[n0:(n0 + n_mn[i]), G] = i
                n0 += n_mn[i]

        phate_op = phate.PHATE(n_jobs=-2, n_pca=20)
        Y_phate = phate_op.fit_transform(X[:, :-1])

        centroids = np.zeros((d, 2))
        for i in range(d):
            centroids[i, :] = Y_phate[X[:, -1] == i].mean(axis=0)
        costm = test_util.get_cost_matrix(centroids, centroids, dim=2)

        np.save(f'../data/simulation_data/cost_{nu_type}_nu_{eta_type}_eta', costm)
