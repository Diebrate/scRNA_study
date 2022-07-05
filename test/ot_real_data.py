import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0,'..')
import test_util
import solver
os.chdir('..\preproc')

import time
start_time = time.time()

batch = False
metric = 'phate'
sink = True
win_size = 2
weight = None

cost_disc = True
cost_cont = False

load_cost = True

reg = 0.05
reg1 = 1
reg2 = 1

if batch:
    df = pd.read_csv(r'..\data\proc_data\proc_df.csv')
else:
    df = pd.read_csv(r'..\data\proc_data\proc_df_no_batch.csv')
time_names = pd.read_csv(r'..\data\time_names.csv', index_col=0)
time_names = time_names['time'].to_numpy()
time_labels = df.time_index

T = len(time_names)
k = df['cluster'].max() + 1

### high dimensional data
if metric in ['pc', 'tsne', 'umap', 'phate']:
    centroids = df[[metric + '1', metric + '2', 'cluster']].groupby('cluster').mean().to_numpy()
    xy_sc = df[[metric + '1', metric + '2']].to_numpy() # continuous data
elif metric == 'raw':
    X = anndata.read_h5ad(r'..\data\proc_data\proc_data').X 
    X = pd.DataFrame(X)
    X['cluster'] = df.cluster
    centroids = X.groupby('cluster').mean().to_numpy()    
    xy_sc = X.to_numpy() # continuous data
dim = centroids.shape[1]   

### convert to multinomial or dirac measure
probs = []
probs_exact = []
for t in range(T):
    p_temp = test_util.get_weight(df.loc[df.time_index==t, 'cluster'].to_numpy(), k=k)
    probs.append(p_temp)
    p_temp = test_util.get_weight_no_ignore(df.loc[df.time_index==t, 'cluster'].to_numpy(), k=k)
    probs_exact.append(p_temp)
probs = np.array(probs)
probs_exact = np.array(probs_exact)
costm = test_util.get_cost_matrix(centroids, centroids, dim=dim)
costm = costm / np.median(costm)
cost = solver.loss_unbalanced_all_local(probs, costm, reg, reg1, reg2, sink=sink, win_size=win_size, weight=weight)
cost_partial = solver.loss_unbalanced_all_local(probs_exact, costm, reg, reg1, reg2, sink=sink, win_size=win_size, weight=weight, partial=True)

### normalization
# cost = cost / np.sum(cost)
cost = np.append(cost, np.nan)

cp = test_util.get_cp_from_cost(cost, win_size=win_size)
cp_partial = test_util.get_cp_from_cost(cost_partial, win_size=win_size)

if cost_disc:
    local_cost_disc = solver.compute_cost_disc(probs, costm, reg, reg1, reg2, sink=sink, partial=False, max_win_size=4)
    local_cost_disc_partial = solver.compute_cost_disc(probs, costm, reg, reg1, reg2, sink=sink, partial=True, max_win_size=4)
    np.save(r'..\results\cost_local_disc.npy', local_cost_disc)
    np.save(r'..\results\cost_local_disc_partial.npy', local_cost_disc_partial)
if cost_cont:
    local_cost_cont = solver.compute_cost_cont(xy_sc, time_labels, reg, reg1, reg2, sink=sink, max_win_size=4)
    np.save(r'..\results\cost_local_cont.npy', local_cost_cont)

if load_cost:
    local_cost_disc = np.load(r'..\results\cost_local_disc.npy')
    local_cost_disc_partial = np.load(r'..\results\cost_local_disc_partial.npy')
    local_cost_cont = np.load(r'..\results\cost_local_cont.npy')
    
cost_disc_m = [solver.get_weighted_cost(local_cost_disc, weight=weight, win_size=i) for i in [1, 2, 3, 4]]
# cost_cont_m = [solver.get_weighted_cost(local_cost_cont, weight=weight, win_size=i) for i in [1, 2, 3, 4]]

### normalizing the cost
for i in range(4):
    # cost_disc_m[i] = cost_disc_m[i] / np.sum(cost_disc_m[i])
    cost_disc_m[i][:i + 1] = 0
    # cost_cont_m[i] = cost_cont_m[i] / np.sum(cost_cont_m[i])

x = list(range(T - 1))
# z = np.polyfit(x, [float(i) for i in cost], 7)
# f = np.poly1d(z)

fig, ax = plt.subplots(figsize=(10,8))

# ax.plot(x, f(x))
# ax.plot(x, cost, label='full')
# ax.plot(x, cost_partial, label='partial')
# ax.legend()
for i in range(4):
    ax.plot(x, cost_disc_m[i], label='clustered with win-size=' + str(i + 1))
    # ax.plot(x, cost_cont_m[i], label='original with win-size=' + str(i + 1))
ax.set_xticks(x)
ax.set_xticklabels(time_names[:-1], rotation=70, size=10)
ax.set_xlabel('Time', size=20)
ax.set_ylabel('Loss', size=20)
ax.set_title(r'Time vs Loss with ' + metric + ' embedding', size = 20)
plt.legend()

print("--- %s seconds ---" % (time.time() - start_time))

