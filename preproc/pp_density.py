import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata
import util

batch = False
do_iter = True
method = 'tsne'
save_graph = True

if batch:
    df = pd.read_csv(r'..\data\proc_data\proc_df.csv')
else:
    df = pd.read_csv(r'..\data\proc_data\proc_df_no_batch.csv')
time_names = pd.read_csv(r'..\data\time_names.csv', index_col=0)
time_names = time_names['time'].to_numpy()

n = len(time_names)

if do_iter:
    method_list = ['pc', 'tsne', 'umap', 'phate']
else:
    method_list = [method]

for m in method_list: 
    for i in range(n):
        data_temp = df[[m + '1', m + '2']].loc[df.time_index == i]
        prob_temp = util.get_kernel_weight_graph(data_temp)
        fig = plt.figure(num=i, figsize=(10, 8))
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(data_temp.iloc[:,0], data_temp.iloc[:,1], prob_temp, cmap='Reds',linewidth=0, antialiased=True, zorder=0.5)
        ax.set_zlabel('Density')
        ax.set_title('Density with ' + m + ' embedding at ' + time_names[i])
        reform_time_name = time_names[i].replace('.', '_')
        if save_graph or do_iter:
            fig.savefig(r'..\image\pp_datavis\density\pp_' + m + '_density_' + reform_time_name)
            plt.close()