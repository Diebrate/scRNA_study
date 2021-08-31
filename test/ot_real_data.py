import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import util
import solver

import os
import test_util
os.chdir('..\preproc')

import time
start_time = time.time()

batch = False
do_iter = False
graph = False
save_fig = False
method = 'phate'
sink = True
win_size = 2
weight = None

reg = 1
reg1 = 1
reg2 = 1

if batch:
    df = pd.read_csv(r'..\data\proc_data\proc_df.csv')
else:
    df = pd.read_csv(r'..\data\proc_data\proc_df_no_batch.csv')
time_names = pd.read_csv(r'..\data\time_names.csv', index_col=0)
time_names = time_names['time'].to_numpy()

T = len(time_names)
k = df['cluster'].max() + 1

if do_iter:
    metric_list = ['raw', 'pc', 'tsne', 'umap', 'phate']
else:    
    metric_list = [method]
for metric in metric_list:
    if metric in ['pc', 'tsne', 'umap', 'phate']:
        centroids = df[[metric + '1', metric + '2', 'cluster']].groupby('cluster').mean().to_numpy()
    elif metric == 'raw':
        X = anndata.read_h5ad(r'..\data\proc_data\proc_data').X 
        X = pd.DataFrame(X)
        X['cluster'] = df.cluster
        centroids = X.groupby('cluster').mean().to_numpy()      
    probs = []
    probs_exact = []
    for t in range(T):
        p_temp = test_util.get_weight(df.loc[df.time_index==t, 'cluster'].to_numpy(), k=k)
        probs.append(p_temp)
        p_temp = test_util.get_weight_no_ignore(df.loc[df.time_index==t, 'cluster'].to_numpy(), k=k)
        probs_exact.append(p_temp)
    probs = np.array(probs)
    probs_exact = np.array(probs_exact)
    dim = centroids.shape[1]
    costm = test_util.get_cost_matrix(centroids, centroids, dim=dim)
    cost = solver.loss_unbalanced_local(probs, costm, reg, reg1, reg2, sink=sink, win_size=win_size, weight=weight)
    cost_partial = solver.loss_unbalanced_local(probs_exact, costm, reg, reg1, reg2, sink=sink, win_size=win_size, weight=weight, partial=True)
cp = test_util.get_cp_from_cost(cost, win_size=win_size)
cp_partial = test_util.get_cp_from_cost(cost, win_size=win_size)

x = list(range(T - 1))
z = np.polyfit(x, [float(i) for i in cost], 7)
f = np.poly1d(z)

fig, ax = plt.subplots(figsize=(10,8))

# ax.plot(x, f(x))
# ax.plot(x, cost, label='full')
# ax.plot(x, cost_partial, label='partial')
# ax.legend()
ax.plot(x, cost)
ax.set_xticks(x)
ax.set_xticklabels(time_names[:-1], rotation=70, size=10)
ax.set_xlabel('Time', size=20)
ax.set_ylabel('Loss', size=20)
ax.set_title(r'Time vs Loss with ' + method + ' embedding', size = 20)

print("--- %s seconds ---" % (time.time() - start_time))

