import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import util
import solver

import os
os.chdir(r'..\test\\')
import test_util
os.chdir('..\preproc')

import time
start_time = time.time()

batch = False
do_iter = False
method = 'phate'
conv = True

reg = 1
reg1 = 1.5
reg2 = 50

t = 25
n = 500
m = 500

if batch:
    df = pd.read_csv(r'..\data\proc_data\proc_df.csv')
else:
    df = pd.read_csv(r'..\data\proc_data\proc_df_no_batch.csv')
df = df.iloc[:, 1:]
time_names = pd.read_csv(r'..\data\time_names.csv', index_col=0)
time_names = time_names['time'].to_numpy()

T = len(time_names)
k = df['cluster'].max() + 1

if method in ['pc', 'tsne', 'umap', 'phate']:
    centroids = df[[method + '1', method + '2', 'cluster']].groupby('cluster').mean().to_numpy()
elif method == 'raw':
    X = anndata.read_h5ad(r'..\data\proc_data\proc_data').X 
    X = pd.DataFrame(X)
    X['cluster'] = df.cluster
    centroids = X.groupby('cluster').mean().to_numpy()      
dim = centroids.shape[1]
costm = util.get_cost_matrix(centroids, centroids, dim)

df1 = df[df.time_index == t]
df2 = df[df.time_index == t + 1]
p1 = test_util.get_weight_no_ignore(df1['cluster'], k)
p2 = test_util.get_weight_no_ignore(df2['cluster'], k)
valid_cluster1 = np.arange(k)[p1 > 0]
valid_cluster2 = np.arange(k)[p2 > 0]

tmap = solver.ot_unbalanced_iter(p1[p1 > 0], p2[p2 > 0], costm[p1 > 0][:, p2 > 0], reg, reg1, reg2)
# growth_total = test_util.growth_CI1(df1['cluster'], df2['cluster'], costm, reg, reg1, reg2, k=k, ignore_empty=True, conv=True)

valid_i = np.random.choice(np.arange(len(valid_cluster1)))
valid_c = valid_cluster1[valid_i]

p_cond = tmap[valid_i] / tmap[valid_i].sum()
x1 = df1[df1.cluster == valid_c].sample(n=n, replace=True).loc[:, [method + '1', method + '2']].to_numpy()
weight_sim = np.random.multinomial(n=m, pvals=p_cond)

x2 = np.zeros((1, 2))
for i in range(len(weight_sim)):
    if weight_sim[i] > 0:
        x2 = np.vstack((x2, df2[df2.cluster == valid_cluster2[i]].sample(n=weight_sim[i], replace=True).loc[:, [method + '1', method + '2']].to_numpy()))
x2 = x2[1:]

costm_single = util.get_cost_matrix(x1, x2, dim)
tmap_single = solver.ot_unbalanced_iter(np.repeat(1 / n, n), np.repeat(1 / m, m), costm_single, reg, reg1, reg2)

for i in np.random.choice(np.arange(len(x1)), size=20):
    ind = tmap_single[i, :].argsort()[-int(m / 10):]
    for j in ind:
        xdata = np.array([x1[i, 0], x2[j, 0]])
        ydata = np.array([x1[i, 1], x2[j, 1]])
        plt.plot(xdata, ydata, c='black', linewidth=0.3)
plt.scatter(x1[:, 0], x1[:, 1], label=time_names[t], s=3)
plt.scatter(x2[:, 0], x2[:, 1], label=time_names[t + 1], s=3)
plt.legend(bbox_to_anchor=(0.9, 1), loc='upper left')












































