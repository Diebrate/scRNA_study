import numpy as np
import anndata
import scanpy
import phate
import os
import re

import time
start_time = time.time()

path_dir = '..\data\GSE122662_RAW'

file_names = []

# pattern = re.compile(r'^.*_D[0-9.]{1,4}_(?!2i).*_C1_.*\.h5$')
pattern = re.compile(r'^.*_D[0-9.]{1,4}_.*_C[12]_.*\.h5$')

for name in os.listdir(path_dir):
    if pattern.match(name) is not None and not re.findall('exp|ctrl', name):
        file_names.append(name)

path_names = [path_dir + '\\' + n for n in file_names]

time_names = []

for name in file_names:
    time_names.append(float(re.search('(?<=D)[0-9.]*', name).group()))
time_names = np.array(['D'+str(i) for i in np.sort(np.unique(time_names))])
    
T = len(time_names)
    
data_raw = []
for path in path_names:
    data_temp = scanpy.read_10x_h5(path)
    data_temp.var_names_make_unique()
    data_temp.obs['time'] = np.repeat(float(re.search('(?<=D)[0-9.]*', path).group()), data_temp.n_obs)
    data_raw.append(data_temp)
    
# data = data_raw[0].copy()
# data.var_names_make_unique()
data = anndata.AnnData.concatenate(*data_raw)
  
do_preproc = True
if do_preproc:
    scanpy.pp.downsample_counts(data, counts_per_cell=15000)
    scanpy.pp.filter_cells(data, min_counts=2000)
    scanpy.pp.filter_genes(data, min_cells=50)
    scanpy.pp.normalize_total(data, target_sum=10000, inplace=True)
    scanpy.pp.log1p(data)
    # scanpy.pp.recipe_seurat(data, log=False)
    # scanpy.pp.combat(data)
    # scanpy.tl.pca(data, n_comps=5)
    # scanpy.pp.neighbors(data)
    # scanpy.external.pp.bbknn(data, batch_key='batch')
    # scanpy.tl.tsne(data)
    # scanpy.tl.umap(data)
    # scanpy.tl.louvain(data)
    
    phate_op = phate.PHATE(n_jobs=-2, n_pca=20)
    Y_phate = phate_op.fit_transform(data.X)
    
    data.obsm['X_phate'] = Y_phate
    # data.obs['time'] = np.array([time_names[int(i)] for i in data.obs['batch']], dtype=str)
    
print("--- %s seconds ---" % (time.time() - start_time))
