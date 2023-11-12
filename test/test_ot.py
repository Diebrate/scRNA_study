import numpy as np
import pandas as pd
import test_util
import solver

use_sys = True
if use_sys:
    import sys
    m = int(sys.argv[1])
else:
    m = 0

eps = 0.001
reg = 1

for n in ['low', 'high']:

    for nu in ['low', 'high']:

        for eta in ['low', 'high']:

            # data_all = pd.read_csv(f'../data/simulation_data/test_sample_{n}_n_{nu}_nu_{eta}_eta.csv')
            data_all = pd.read_csv(f'../data/simulation_data/simulation_id{m}_{n}_n_{nu}_nu_{eta}_eta.csv')
            B = int(data_all.batch.max()) + 1
            T = int(data_all.time.max())
            d = int(data_all.type.max()) + 1

            res = np.zeros((B, T))
            tmap_all = np.zeros((B, T, d, d))

            for k in range(B):

                data = data_all[(data_all.n == n) & (data_all.nu == nu) & (data_all.eta == eta) & (data_all.batch == k)]

                centroids = data[['phate1', 'phate2', 'type']].groupby('type').mean().to_numpy()
                costm = test_util.get_cost_matrix(centroids, centroids, dim=2)
                probs = np.zeros((T + 1, d))
                for t in range(T + 1):
                    counts = data['type'][data['time'] == t].value_counts().reindex(index=np.arange(d), fill_value=0).to_numpy()
                    for count in counts:
                        count += 1
                    p = counts / np.sum(counts)
                    probs[t, :] = p

                tmap = solver.ot_unbalanced_all(probs[:-1, ].T, probs[1:, ].T, costm, reg=eps, reg1=reg, reg2=50)
                cost = []
                for t in range(T):
                    phat = tmap[t].sum(axis=1)
                    c = np.sum(tmap[t] * costm) + reg * np.sum(phat * np.log(phat / probs[t]))
                    cost.append(c)
                    tmap_all[k, t, :, :] = tmap[t]
                cp = test_util.get_cp_from_cost(cost, win_size=1, snr=0.001)
                if len(cp) > 0:
                    res[k, cp - 1] = 1

                print(f'finished m = {m}, b = {k}, n = {n}, nu = {nu}, eta = {eta}')

                # # sanity check
                # import matplotlib.pyplot as plt
                # plt.plot(cost)

            np.save(f'../results/simulation/{n}_n/{nu}_nu_{eta}_eta/test_ot_id{m}', res)
            np.save(f'../results/simulation/{n}_n/{nu}_nu_{eta}_eta/tmap_id{m}', res)
