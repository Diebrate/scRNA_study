import numpy as np
import pandas as pd
import solver

rng = np.random.default_rng(2023)

M = 100
J = 10000
T = 50
d = 10
G = 50

eps = 0.001
reg = 1

means = np.arange(d) - (d / 2)

ns = {'low': 1000, 'high': 2000}
nus = {'low': 0.1, 'high': 0.25}
etas = {'low': 0.5, 'high': 1}

cp_vec = np.array([-1] * (d // 2) + [1] * (d - d // 2))

cp = [10, 20, 30, 40]

perf_cp = []
perf_ncp = []

for n_key in ['low', 'high']:

    summary_cp = pd.DataFrame(index=['low', 'high'], columns=['low', 'high'])
    summary_ncp = pd.DataFrame(index=['low', 'high'], columns=['low', 'high'])

    for eta_key in ['low', 'high']:

        for nu_key in ['low', 'high']:

            n = ns[n_key]
            eta = etas[eta_key]
            nu = nus[nu_key]

            costm = np.load(f'../data/simulation_data/cost_{nu_key}_nu_{eta_key}_eta.npy')

            diff_cp = []
            diff_ncp = []

            change1 = np.exp(eta * cp_vec)
            change2 = np.exp(eta * cp_vec[::-1])
            Q = np.ones((T + 1, d)) / d
            for t in range(T):
                g0 = np.exp(nu * np.sin(np.pi * (t + np.arange(d)) / d))
                q0 = Q[t] * g0 / np.sum(Q[t] * g0)
                if t in [10, 20, 30, 40]:
                    change = change2 if t % 20 == 0 else change1
                    q0 = (change * q0) / np.sum(change * q0)
                Q[t + 1, :] = q0

            tmap = solver.ot_unbalanced_all(Q[:-1, ].T, Q[1:, ].T, costm, reg=eps, reg1=reg, reg2=50)

            res = []

            for m in range(1, M + 1):
                res.append(np.load(f'../results/simulation/compile/{n_key}_n/{nu_key}_nu_{eta_key}_eta/tmap_id{m}.npy'))

            for i in range(len(res)):
                for j in range(len(res[i])):
                    diff0_cp = 0
                    diff0_ncp = 0
                    for t in range(T):
                        if t in cp:
                            diff0_cp += np.power(tmap[t] - res[i][j, t, :, :], 2).sum()
                        else:
                            diff0_ncp += np.power(tmap[t] - res[i][j, t, :, :], 2).sum()
                    diff_cp.append(diff0_cp / len(cp))
                    diff_ncp.append(diff0_ncp / (T - len(cp)))

            txt = '{:.2f}({:.2f})'

            summary_cp.loc[eta_key, nu_key] = txt.format(10000 * np.mean(diff_cp),
                                                         10000 * np.std(diff_cp) / np.sqrt(M * len(res)))
            summary_ncp.loc[eta_key, nu_key] = txt.format(10000 * np.mean(diff_ncp),
                                                          10000 * np.std(diff_ncp) / np.sqrt(M * len(res)))

    summary_cp.columns = [f'$\\nu={k}$' for k in nus.values()]
    summary_cp.index = [f'$\\eta={k}$' for k in etas.values()]
    summary_ncp.columns = [f'$\\nu={k}$' for k in nus.values()]
    summary_ncp.index = [f'$\\eta={k}$' for k in etas.values()]

    perf_cp.append(summary_cp)
    perf_ncp.append(summary_ncp)

perf_cp = pd.concat(perf_cp, axis='rows', keys=[f'$n={k}$' for k in ns.values()])
perf_ncp = pd.concat(perf_ncp, axis='rows', keys=[f'$n={k}$' for k in ns.values()])
perf_cp.to_csv('../results/perf/summary_conv_cp.csv')
perf_ncp.to_csv('../results/perf/summary_conv_ncp.csv')
