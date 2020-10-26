import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib import cm

batch = False

if batch:
    data = anndata.read(r'..\data\proc_data\proc_data')
else:
    data = anndata.read(r'..\data\proc_data\proc_data_no_batch')
df = pd.DataFrame()

import os
import re

path_dir = '..\data\GSE122662_RAW'

file_names = []

pattern = re.compile(r'^.*_D[0-9.]{1,4}_(?!2i).*_C1_.*\.h5$')

for name in os.listdir(path_dir):
    if pattern.match(name) is not None and not re.findall('exp|ctrl', name):
        file_names.append(name)

path_names = [path_dir + '\\' + n for n in file_names]

time_names = []

for name in file_names:
    time_names.append(re.search('D[0-9.]*', name).group())
    
T = len(time_names)

df['pc1'] = data.obsm['X_pca'][:, 0]
df['pc2'] = data.obsm['X_pca'][:, 1]
df['tsne1'] = data.obsm['X_tsne'][:, 0]
df['tsne2'] = data.obsm['X_tsne'][:, 1]
df['umap1'] = data.obsm['X_umap'][:, 0]
df['umap2'] = data.obsm['X_umap'][:, 1]
df['phate1'] = data.obsm['X_phate'][:, 0]
df['phate2'] = data.obsm['X_phate'][:, 1]
df['cluster'] = np.array(data.obs['louvain'], dtype=int)
df['time_index'] = np.array(data.obs['batch'], dtype=int)
df['time'] = np.array([time_names[int(i)] for i in data.obs['batch']])

fig = plt.figure(figsize=(12, 12), constrained_layout=True)
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.03])
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
cax = fig.add_subplot(gs[:, 2])

scatter0 = ax0.scatter(df['pc1'], df['pc2'], c=np.array(df['time_index'], dtype=int), 
                       s=0.3,cmap='Spectral')
scatter1 = ax1.scatter(df['tsne1'], df['tsne2'], c=np.array(df['time_index'], dtype=int), 
                       s=0.3, cmap='Spectral')
scatter2 = ax2.scatter(df['umap1'], df['umap2'], c=np.array(df['time_index'], dtype=int), 
                       s=0.3, cmap='Spectral')
scatter3 = ax3.scatter(df['phate1'], df['phate2'], c=np.array(df['time_index'], dtype=int), 
                       s=0.3, cmap='Spectral')
ax0.set_xlabel('pc1')
ax0.set_ylabel('pc2')
ax1.set_xlabel('tsne1')
ax1.set_ylabel('tsne2')
ax2.set_xlabel('umap1')
ax2.set_ylabel('umap2')
ax3.set_xlabel('phate1')
ax3.set_ylabel('phate2')

cbar = fig.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=T),
                                      cmap='Spectral'), cax=cax)
cbar.set_label('time index')

# fig.tight_layout()

plt.show()



































































