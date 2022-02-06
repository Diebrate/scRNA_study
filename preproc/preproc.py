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

cond1 = 'all'
cond2 = 'all'

if cond1 == 'serum':
    if cond2 == 'C1':
        pattern = re.compile(r'^.*_D[0-9.]{1,4}_(?!2i).*_C1_.*\.h5$')
    elif cond2 == 'C2':
        pattern = re.compile(r'^.*_D[0-9.]{1,4}_(?!2i).*_C2_.*\.h5$')
    elif cond2 == 'all':
        pattern = re.compile(r'^.*_D[0-9.]{1,4}_(?!2i).*_C[12]_.*\.h5$')
elif cond1 == '2i':
    if cond2 == 'C1':
        pattern = re.compile(r'^.*_D[0-9.]{1,4}_(?!serum).*_C1_.*\.h5$')
    elif cond2 == 'C2':
        pattern = re.compile(r'^.*_D[0-9.]{1,4}_(?!serum).*_C2_.*\.h5$')
    elif cond2 == 'all':
        pattern = re.compile(r'^.*_D[0-9.]{1,4}_(?!serum).*_C[12]_.*\.h5$')
elif cond1 == 'all':
    if cond2 == 'C1':
        pattern = re.compile(r'^.*_D[0-9.]{1,4}.*_C1_.*\.h5$')
    elif cond2 == 'C2':
        pattern = re.compile(r'^.*_D[0-9.]{1,4}.*_C2_.*\.h5$')
    elif cond2 == 'all':
        pattern = re.compile(r'^.*_D[0-9.]{1,4}.*_C[12]_.*\.h5$')

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
seurat = True
if do_preproc:
    if seurat:
        scanpy.pp.recipe_seurat(data)
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
        
        data.write(r'..\data\proc_data\proc_df_full' + '_' + cond1 + '_' + cond2 + '.h5ad')
    else:
        scanpy.pp.downsample_counts(data, counts_per_cell=15000)
        scanpy.pp.filter_cells(data, min_counts=2000)
        scanpy.pp.filter_genes(data, min_cells=50)
        scanpy.pp.normalize_total(data, target_sum=10000, inplace=True)
        
        X_hv = data.X.copy()
        scanpy.pp.log1p(data)
        scanpy.pp.highly_variable_genes(data, min_disp=1)
        data_hv = anndata.AnnData(X=X_hv[:, data.var['highly_variable'].to_numpy()],
                                  obs=data.obs,
                                  var=data.var[data.var['highly_variable'].to_numpy()])
        scanpy.pp.normalize_total(data_hv, target_sum=10000, inplace=True)
        scanpy.pp.log1p(data_hv)
    
        Y_phate_hv = phate_op.fit_transform(data_hv.X)
        data_hv.obsm['X_phate'] = Y_phate_hv
        
        data.write(r'..\data\proc_data\proc_df_full_no_seurat' + '_' + cond1 + '_' + cond2 + '.h5ad')
        data_hv.write(r'..\data\proc_data\proc_df_full_no_seurat_hv' + '_' + cond1 + '_' + cond2 + '.h5ad')
    
print("--- %s seconds ---" % (time.time() - start_time))
