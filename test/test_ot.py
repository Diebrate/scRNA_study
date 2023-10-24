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

B = 50
data = pd.read_csv('../data/simulation_data/simulation_id1_0.csv')
T = int(data.time.max())
d = int(data.type.max()) + 1
res = np.zeros((B, T))

for k in range(B):
    data = pd.read_csv('../data/simulation_data/simulation_id' + str(m) + '_' + str(k) + '.csv')

    centroids = data[['phate1', 'phate2', 'type']].groupby('type').mean().to_numpy()
    costm = test_util.get_cost_matrix(centroids, centroids, dim=2)
    probs = np.zeros((T + 1, d))
    for t in range(T + 1):
        p = data['type'][data['time'] == t].value_counts(normalize=True).sort_index().to_numpy()
        probs[t, :] = p

    eps = 0.001
    reg = 1

    tmap = solver.ot_unbalanced_all(probs[:-1, ].T, probs[1:, ].T, costm, reg=eps, reg1=reg, reg2=50)
    cost = []
    for t in range(T):
        phat = tmap[t].sum(axis=1)
        c = np.sum(tmap[t] * costm) + reg * np.sum(phat * np.log(phat / probs[t]))
        cost.append(c)
    cp = test_util.get_cp_from_cost(cost, win_size=1)
    res[k, cp - 1] = 1

    # # sanity check
    # import matplotlib.pyplot as plt
    # plt.plot(cost)

np.save('../results/simulation/test_ot_id' + str(m), res)
