import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util

df = pd.read_csv(r'..\data\proc_data\proc_df.csv')
time_names = pd.read_csv(r'..\data\time_names.csv', index_col=0)
time_names = time_names['time'].to_numpy()

T = len(time_names)
k = df['cluster'].max() + 1
metric = 'phate'
centroids = df[[metric + '1', metric + '2', 'cluster']].groupby('cluster').mean().to_numpy()

probs = []
for t in range(T):
    p_temp = util.get_prob_cluster(df.loc[df.time_index==t, 'cluster'].to_numpy(), k=k)
    probs.append(p_temp)
probs = np.array(probs)
    
cluster_ot = util.compute_all_ot_cluster2(centroids, probs, dim=2, reg=10)

for i in range(len(cluster_ot)):
    # cluster_ot[i] = cluster_ot[i] / cluster_ot[i].sum()
    # cluster_ot[i] = np.reshape(util.renormalize_matrix(cluster_ot[i], 0), (k, k))
    plt.figure(i)
    plt.imshow(cluster_ot[i], 'magma')
    plt.title('(total) pp_louvain with ' + metric + ' ' + time_names[i] + ' to ' + time_names[i+1])
    plt.xlabel(metric + '1')
    plt.ylabel(metric + '2')
    plt.colorbar()
    reform_time_name = time_names[i].replace('.', '_') + 'to' + time_names[i+1].replace('.', '_')
    plt.savefig(r'..\image\pp_datavis\ot_map\\' + metric + '\pp_louvain_cluster_map_' + metric + '_' + reform_time_name)