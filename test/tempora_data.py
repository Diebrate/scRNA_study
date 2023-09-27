import numpy as np
import pandas as pd

import wot
import anndata

import re

# Path to input files
FLE_COORDS_PATH = '../wot/data/fle_coords.txt'
FULL_DS_PATH = '../wot/data/ExprMatrix.h5ad'
VAR_DS_PATH = '../wot/data/ExprMatrix.var.genes.h5ad'
CELL_DAYS_PATH = '../wot/data/cell_days.txt'
GENE_SETS_PATH = '../wot/data/gene_sets.gmx'
GENE_SET_SCORES_PATH = '../wot/data/gene_set_scores.csv'
CELL_SETS_PATH = '../wot/data/cell_sets.gmt'

coord_df = pd.read_csv(FLE_COORDS_PATH, index_col='id', sep='\t')
days_df = pd.read_csv(CELL_DAYS_PATH, index_col='id', sep='\t')

# # Read expression matrix, cell days, and 2-d coordinates
adata = wot.io.read_dataset(FULL_DS_PATH, obs=[days_df, coord_df])
unique_days = adata.obs['day'].unique()
unique_days = unique_days[np.isnan(unique_days) == False]

gs = wot.io.read_sets(GENE_SETS_PATH, adata.var.index.values)
gene_set_scores_df = pd.DataFrame(index=adata.obs.index)
for j in range(gs.shape[1]):
    gene_set_name = str(gs.var.index.values[j])
    result = wot.score_gene_sets(ds=adata, gs=gs[:, [j]], permutations=0, method='mean_z_score')
    gene_set_scores_df[gene_set_name] = result['score']
gene_set_scores_df.to_csv(GENE_SET_SCORES_PATH, index_label='id')

gene_set_scores_df = pd.read_csv(GENE_SET_SCORES_PATH, index_col='id')

gene_set_scores_df = gene_set_scores_df.join(coord_df).join(days_df)

# # Load cell sets
cell_sets = wot.io.read_sets(CELL_SETS_PATH)
full_id = cell_sets.var_names[cell_sets.X.argmax(axis=1)]
full_id = pd.DataFrame({'cell type': full_id}, index=cell_sets.obs_names)

full_df = pd.read_csv(r'../wot/data/full_df.csv')
full_df = full_df[full_df['day'] != 8.25]
full_df = full_df[full_df['day'] != 8.75]

get_ot = False
seurat = True
do_hv = False
cond1 = 'serum'
cond2 = 'all'
suffix = '_' + cond1 + '_' + cond2

ot_df = wot.io.read_dataset(r'..\data\proc_data\proc_df_full_serum_all.h5ad')

cell_id = ot_df.obs_names.to_numpy()
cell_id_list = np.array([re.search(r'^.*-1(?=-)', name).group() for name in cell_id])

df = pd.DataFrame(ot_df.X)
df.columns = ['gene' + str(i) for i in range(ot_df.X.shape[1])]
df['day'] = ot_df.obs['time'].to_numpy()
df.index=cell_id_list

full_id_new = full_id.copy()
full_id_new.index = [re.search(r'(?<=C[12]_).*$', name).group(0) for name in full_id_new.index]

df = df.join(full_id_new)
df['cell type'].fillna('MEF', inplace=True)

df = df[df['day'] != 8.25]
df = df[df['day'] != 8.75]

df['cell type'] = df['cell type'].astype('category')
df['type numeric'] = df['cell type'].cat.codes

# df.to_csv(r'../data/proc_data/r_df.csv')