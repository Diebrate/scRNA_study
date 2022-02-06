import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import wot
import anndata

import re
import seaborn as sns

# Path to input files
FLE_COORDS_PATH ='data/fle_coords.txt'
FULL_DS_PATH = 'data/ExprMatrix.h5ad'
VAR_DS_PATH = 'data/ExprMatrix.var.genes.h5ad'
CELL_DAYS_PATH = 'data/cell_days.txt'
GENE_SETS_PATH = 'data/gene_sets.gmx'
GENE_SET_SCORES_PATH = 'data/gene_set_scores.csv'
CELL_SETS_PATH = 'data/cell_sets.gmt'



coord_df = pd.read_csv(FLE_COORDS_PATH, index_col='id', sep='\t')
days_df = pd.read_csv(CELL_DAYS_PATH, index_col='id', sep='\t')

# # Read expression matrix, cell days, and 2-d coordinates
adata = wot.io.read_dataset(FULL_DS_PATH, obs=[days_df,coord_df])
unique_days = adata.obs['day'].unique()
unique_days = unique_days[np.isnan(unique_days) == False]



gs = wot.io.read_sets(GENE_SETS_PATH, adata.var.index.values)
gene_set_scores_df = pd.DataFrame(index=adata.obs.index)
for j in range(gs.shape[1]):
    gene_set_name = str(gs.var.index.values[j])
    result = wot.score_gene_sets(ds=adata, gs=gs[:, [j]], permutations=0, method='mean_z_score')
    gene_set_scores_df[gene_set_name] = result['score']
gene_set_scores_df.to_csv(GENE_SET_SCORES_PATH, index_label='id')



gene_set_scores_df = pd.read_csv(GENE_SET_SCORES_PATH,index_col='id')


gene_set_scores_df = gene_set_scores_df.join(coord_df).join(days_df)



# # Load cell sets
cell_sets = wot.io.read_sets(CELL_SETS_PATH)
full_id = cell_sets.var_names[cell_sets.X.argmax(axis=1)]
full_id = pd.DataFrame({'cell type': full_id}, index=cell_sets.obs_names)

get_df = False
if get_df:
    full_df = cell_sets.obs.join(coord_df).join(days_df)
    full_df = coord_df.join(days_df).join(full_id)
    full_df = full_df[~np.isnan(full_df['day'])]
    full_df['cell type'].fillna('MEF', inplace=True)
    full_df.to_csv(r'data\full_df.csv')

full_df = pd.read_csv(r'data\full_df.csv')
full_df = full_df[full_df['day'] != 8.25]
full_df = full_df[full_df['day'] != 8.75]

get_ot = True
seurat = True
do_hv = False
cond1 = 'serum'
cond2 = 'all'
suffix = '_' + cond1 + '_' + cond2
if get_ot:
    if seurat:
        ot_df = wot.io.read_dataset(r'..\data\proc_data\proc_df_full' + suffix + '.h5ad')
    else:
        if do_hv:
            ot_df = wot.io.read_dataset(r'..\data\proc_data\proc_df_full_no_seurat_hv' + suffix + '.h5ad')
        else:
            ot_df = anndata.read_h5ad(r'..\data\proc_data\proc_df_full_no_seurat' + suffix + '.h5ad')
    
    cell_id = ot_df.obs_names.to_numpy()
    cell_id_list = np.array([re.search(r'^.*-1(?=-)', name).group() for name in cell_id])
    
    # cell_sets_new = cell_sets.copy()
    # cell_sets_new.obs_names = [re.search(r'(?<=C[12]_).*$', name).group(0) for name in cell_sets.obs_names]
    
    ot_df = pd.DataFrame({'x': ot_df.obsm['X_phate'][:, 0],
                          'y': ot_df.obsm['X_phate'][:, 1],
                          'day': ot_df.obs['time'].to_numpy()}, index=cell_id_list)
    full_id_new = full_id.copy()
    full_id_new.index = [re.search(r'(?<=C[12]_).*$', name).group(0) for name in full_id_new.index]
    
    ot_df = ot_df.join(full_id_new)
    ot_df['cell type'].fillna('MEF', inplace=True)
    
    ot_df.to_csv(r'..\data\proc_data\ot_df.csv')
    
ot_df = pd.read_csv(r'..\data\proc_data\ot_df.csv')
ot_df = ot_df[ot_df['day'] != 8.25]
ot_df = ot_df[ot_df['day'] != 8.75]

plot = None

if plot == 'schiebinger':
    
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(15, 8))
    scatter = ax[0].scatter(x=full_df['x'], y=full_df['y'], s=0.25, c=full_df['day'])
    ax[0].legend(*scatter.legend_elements(num=19), title='day')

    sns.scatterplot(x='x', y='y', data=full_df, hue='cell type', linewidth=0, s=1, ax=ax[1])

elif plot == 'seurat':
    
    fig, ax = plt.subplots(nrows=1, ncols=2, constrained_layout=True, figsize=(15, 8))
    scatter = ax[0].scatter(x=ot_df['x'], y=ot_df['y'], s=0.25, c=ot_df['day'])
    ax[0].legend(*scatter.legend_elements(num=19), title='day')

    sns.scatterplot(x='x', y='y', data=ot_df, hue='cell type', linewidth=0, s=1, ax=ax[1])

check_gene = True
gene_list = ['Fut9', 'Shisa8', 'Dmrtc2', 'Zic3', 'Pou3f1']
if check_gene:
    raw_df = wot.io.read_dataset(r'..\data\proc_data\proc_df_full_no_seurat.h5ad')
    gene_df = pd.DataFrame(data=raw_df[:, gene_list].X.todense(),
                           index=raw_df.obs_names,
                           columns=gene_list)
    gene_df['day'] = raw_df.obs['time']
    gene_df = gene_df.groupby(by='day').mean()
    gene_df = gene_df.apply(lambda x: x / np.sum(x), axis=0)