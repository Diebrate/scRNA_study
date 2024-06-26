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
graph = False
save_fig = False
method = 'phate'
balanced = False
fast = True
sink = False

reg = 1
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
    probs = []
    for t in range(T):
        p_temp = util.get_prob_cluster(df.loc[df.time_index==t, 'cluster'].to_numpy(), k=k)
        probs.append(p_temp)
    probs = np.array(probs)
    dim = centroids.shape[1]
    if balanced:
        if fast:
            costm = util.get_cost_matrix(centroids, centroids, dim)
            probs_t1 = probs.transpose()[:, :-1]
            probs_t2 = probs.transpose()[:, 1:]
            cluster_ot = solver.ot_balanced_all(probs_t1, probs_t2, costm, reg=reg)
        else:
            cluster_ot = util.compute_all_ot_cluster2(centroids, probs, dim=dim, reg=reg) 
    else:
        if fast:
            costm = util.get_cost_matrix(centroids, centroids, dim)
            probs_t1 = probs.transpose()[:, :-1]
            probs_t2 = probs.transpose()[:, 1:]
            cluster_ot = solver.ot_unbalanced_all(probs_t1, probs_t2, costm, reg=reg, reg1=reg1, reg2=reg2)
        else:
            cluster_ot = util.compute_all_ot_cluster2_unbalanced(centroids, probs, dim=dim, reg=reg, reg1=reg1, reg2=reg2)
    if graph:
        for i in range(len(cluster_ot)):
            # cluster_ot[i] = cluster_ot[i] / cluster_ot[i].sum()
            # cluster_ot[i] = np.reshape(util.renormalize_matrix(cluster_ot[i], 0), (k, k))
            plt.figure(i)
            plt.imshow(cluster_ot[i], 'magma')
            plt.title('(total) pp_louvain with ' + metric + ' ' + time_names[i] + ' to ' + time_names[i+1] + [' (no_batch)', ' (batch)'][batch])
            plt.xlabel(metric + '1')
            plt.ylabel(metric + '2')
            plt.colorbar()
            reform_time_name = time_names[i].replace('.', '_') + 'to' + time_names[i+1].replace('.', '_')
            if save_fig:
                plt.savefig(r'..\image\pp_datavis\\' + ['no_batch', 'batch'][batch] + '\ot_map\\' + metric + '\pp_louvain_cluster_map_' + metric + '_' + reform_time_name + ['_no_batch', '_batch'][batch])
        if do_iter or save_fig:    
            plt.close('all')
n_seed = 1
M = test_util.get_cost_matrix(centroids, centroids, dim=dim)
p_values = np.zeros((n_seed, T - 1))
rank_sum = np.zeros(T - 1)
for i in range(n_seed):
    print('-----------------------------------------------' + '\nStarting seed ' + str(i + 1))
    np.random.seed(seed=i + 2000)    
    for t in range(T - 1):
        print('#####################################' + '\nStarting time index ' + str(t))
        x_temp = df.loc[df.time_index == t, 'cluster'].to_numpy()
        y_temp = df.loc[df.time_index == t+1, 'cluster'].to_numpy()
        if balanced:
            if fast:
                p_values[i, t] = test_util.perm_test_balanced_fast(x_temp, y_temp, costm=M, reg=reg, k=k, sink=sink)
            else:
                p_values[i, t] = test_util.perm_test(test_util.ot_map_test1, x_temp, y_temp, tail='right', n_times=500, timer=True, M=M, k=k, reg=reg)['p_value']
        else:
            if fast:
                p_values[i, t] = test_util.perm_test_unbalanced_fast(x_temp, y_temp, costm=M, reg=reg, reg1=reg1, reg2=reg2, k=k, sink=sink)
            else:
                p_values[i, t] = test_util.perm_test1_unbalanced(x_temp, y_temp, M, reg, reg1, reg2, k=k)
        print('Finished time index ' + str(t) + '\n#####################################')
    rank_sum += np.array(pd.DataFrame(p_values[i]).rank(axis=0).iloc[:, 0])
    # rank_sum += T - 1 - np.argsort(p_values[i])
    print('\nFinished seed ' + str(i + 1) + '\n-----------------------------------------------')

df_res = pd.DataFrame()
df_res['time'] = range(T - 1)
df_res['r'] = rank_sum
for i in range(n_seed):
    df_res['p' + str(i)] = p_values[i, :]

x = list(range(T - 1))
z = np.polyfit(x, [float(i) for i in df_res.r], 7)
f = np.poly1d(z)

fig, ax = plt.subplots(figsize=(10,8))

ax.plot(x, f(x))
ax.set_xticks(x)
ax.set_xticklabels(time_names[:-1], rotation=70, size=10)
ax.set_xlabel('Time', size=20)
ax.set_ylabel('Rank', size=20)
ax.set_title(r'Time vs Rank with ' + method + ' embedding', size = 20)

print("--- %s seconds ---" % (time.time() - start_time))


















