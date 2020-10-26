import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import util

batch = True
if batch:
    df = pd.read_csv(r'..\data\proc_data\proc_df.csv')
else:
    df = pd.read_csv(r'..\data\proc_data\proc_df_no_batch.csv')
time_names = pd.read_csv(r'..\data\time_names.csv', index_col=0)
time_names = time_names['time'].to_numpy()

T = len(time_names)
k = df['cluster'].max() + 1
metric_list = ['raw', 'pc', 'tsne', 'umap', 'phate']
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
    cluster_ot = util.compute_all_ot_cluster2(centroids, probs, dim=dim, reg=10)
    
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
        plt.savefig(r'..\image\pp_datavis\\' + ['no_batch', 'batch'][batch] + '\ot_map\\' + metric + '\pp_louvain_cluster_map_' + metric + '_' + reform_time_name + ['_no_batch', '_batch'][batch])
    plt.close('all')