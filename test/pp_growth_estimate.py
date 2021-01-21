import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import util
# import solver

import os
os.chdir(r'..\test\\')
import test_util
os.chdir('..\preproc')

import time
start_time = time.time()

batch = False
do_iter = False
graph = False
method = 'phate'
save_fig = False
conv = True

reg = 0.1
reg1 = 1
reg2 = 50

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
    dim = centroids.shape[1]
    
costm = util.get_cost_matrix(centroids, centroids, dim)

growth = []

for t in range(T-1):
    x1_temp = df.loc[df.time_index == t, 'cluster'].to_numpy(dtype=int)
    x2_temp = df.loc[df.time_index == t + 1, 'cluster'].to_numpy(dtype=int)
    growth.append(test_util.growth_CI1(x1_temp, x2_temp, costm, reg, reg1, reg2, k=k, ignore_empty=False, conv=conv))
    print('Finished time index ' + str(t))
    
growth_df = np.zeros((T-1, k))
for t in range(T-1):
    label_non_zero = growth[t]['valid_group1']
    l = len(label_non_zero)
    for i in range(l):
        growth_df[t, label_non_zero[i]] = growth[t]['est'][i]

growth_df = pd.DataFrame(data=growth_df)

growth_df.columns = ['cluster ' + str(i) for i in range(k)]
growth_df.index = time_names[:-1]

x = list(range(T - 1))

fig1, ax1 = plt.subplots(figsize=(12, 8))
for i in range(k):
    ax1.plot(growth_df['cluster ' + str(i)], label='cluster ' + str(i))

ax1.legend(bbox_to_anchor=(0.98, 1), loc='upper left')
ax1.set_xticks(x)
ax1.set_xticklabels(time_names[:-1], rotation=70, size=10)

fig2, ax2 = plt.subplots(figsize=(12, 8))
for i in range(k):
    z = np.polyfit(x, growth_df['cluster ' + str(i)], 5)
    f = np.poly1d(z)
    ax2.plot(x, f(x), label='cluster ' + str(i))

ax2.legend(bbox_to_anchor=(0.98, 1), loc='upper left')
ax2.set_xticks(x)
ax2.set_xticklabels(time_names[:-1], rotation=70, size=10)

    
print("--- %s seconds ---" % (time.time() - start_time))

