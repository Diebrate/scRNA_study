import anndata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import util

data = anndata.read_h5ad(r'..\data\proc_data\proc_data_no_batch')
time_names = pd.read_csv(r'..\data\time_names.csv', index_col=0)

s = 20
t = 22
k = data.obs['louvain'].astype(int).max() + 1

method = 'tsne'

index_s = data.obs['batch'].astype(int) == s
index_t = data.obs['batch'].astype(int) == t

x_s = data.obsm['X_' + method][index_s]
x_t = data.obsm['X_' + method][index_t]

l_s = data.obs['louvain'][index_s].astype(int)
l_t = data.obs['louvain'][index_t].astype(int)

fig1 = plt.figure(figsize=(10, 8), tight_layout=True)
ax1 = fig1.add_subplot()
scatter1 = ax1.scatter(x_s[:, 0], x_s[:, 1], c=l_s, cmap='Spectral', s=7)
ax1.legend(*scatter1.legend_elements(num=k), title='cluster',
           bbox_to_anchor=(1, 1), loc='upper left')
ax1.set_title('Cluster Map at ' + time_names.time[s])
ax1.set_xlabel(method + '1')
ax1.set_ylabel(method + '2')

fig2 = plt.figure(figsize=(10, 8), tight_layout=True)
ax2 = fig2.add_subplot()
scatter2 = ax2.scatter(x_t[:, 0], x_t[:, 1], c=l_t, cmap='Spectral', s=7)
ax2.legend(*scatter1.legend_elements(num=k), title='cluster',
           bbox_to_anchor=(1, 1), loc='upper left')
ax2.set_title('Cluster Map at ' + time_names.time[t])
ax2.set_xlabel(method + '1')
ax2.set_ylabel(method + '2')

fig = plt.figure(figsize=(10, 8), tight_layout=True)
ax = fig.add_subplot()
scatter = ax.scatter(data.obsm['X_' + method][:, 0], data.obsm['X_' + method][:, 1],
                     c=data.obs['louvain'].astype(int), s=1, cmap='Spectral')
ax.legend(*scatter.legend_elements(num=k), title='cluster',
          bbox_to_anchor=(1, 1), loc='upper left')
ax.set_title('Cluster on Whole Data Set')
ax.set_xlabel(method + '1')
ax.set_ylabel(method + '2')

p_s = util.get_kernel_weight_graph(x_s)
p_t = util.get_kernel_weight_graph(x_t)

fig11 = plt.figure(figsize=(10, 8), tight_layout=True)
ax11 = fig11.add_subplot(projection='3d')
scatt3d_11 = ax11.scatter(x_s[:, 0], x_s[:, 1], p_s, c=l_s, cmap='Spectral', s=7)
ax11.legend(*scatt3d_11.legend_elements(num=k), title='cluster',
            bbox_to_anchor=(-0.05, 1), loc='upper left')
ax11.set_title('Cluster Density Map at ' + time_names.time[s])
ax11.set_xlabel(method + '1')
ax11.set_ylabel(method + '2')
ax11.set_zlabel('density')

fig22 = plt.figure(figsize=(10, 8), tight_layout=True)
ax22 = fig22.add_subplot(projection='3d')
scatt3d_22 = ax22.scatter(x_t[:, 0], x_t[:, 1], p_t, c=l_t, cmap='Spectral', s=7)
ax22.legend(*scatt3d_22.legend_elements(num=k), title='cluster',
            bbox_to_anchor=(-0.05, 1), loc='upper left')
ax22.set_title('Cluster Density Map at ' + time_names.time[t])
ax22.set_xlabel(method + '1')
ax22.set_ylabel(method + '2')
ax22.set_zlabel('density')

























