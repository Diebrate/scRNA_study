import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap

import os
import sys
sys.path.insert(0,'..')
import test_util
import solver
os.chdir('../preproc')

import time
start_time = time.time()

sink = False
win_size = 1
weight = None

do_cost = True
do_save = False
do_load = False

reg = 0.001 # 0.05
reg1 = 1
reg2 = 1

time_names = pd.read_csv(r'../data/time_names.csv', index_col=0)
time_names = time_names['time'].to_numpy()

time_names = time_names[time_names != 'D8.25']
time_names = time_names[time_names != 'D8.75']

# df = pd.read_csv(r'../wot/data/full_df.csv')
df = pd.read_csv(r'../data/proc_data/ot_df.csv')
df = df[df['day'] != 8.25]
df = df[df['day'] != 8.75]
time_labels = [float(i[1:]) for i in time_names]
sample_size = pd.DataFrame()
sample_size['day'] = time_labels
sample_size['sample size'] = df.groupby('day').size().to_numpy()
# sample_size.to_csv(r'../results/sample_size.csv')

T = len(time_labels)
k = len(np.unique(df['cell type']))

centroids = df[['x', 'y', 'cell type']].groupby('cell type').mean().to_numpy()
xy_sc = df[['x', 'y']].to_numpy() # continuous data
dim = centroids.shape[1]

### convert to multinomial or dirac measure
type_list = df['cell type'].unique()
tmaps = []
for t in range(T - 1):
    df1 = df[df['day'] == time_labels[t]]
    df2 = df[df['day'] == time_labels[t + 1]]
    x1 = df1[['x', 'y']].to_numpy()
    x2 = df2[['x', 'y']].to_numpy()
    p1 = np.ones(x1.shape[0]) / x1.shape[0]
    p2 = np.ones(x2.shape[0]) / x2.shape[0]
    costm = test_util.get_cost_matrix(x1, x2, dim=dim)
    tmap_cont = solver.ot_unbalanced(p1, p2, costm, reg=reg, reg1=reg1, reg2=50)
    label1 = pd.get_dummies(df1['cell type'])[type_list].to_numpy()
    label2 = pd.get_dummies(df2['cell type'])[type_list].to_numpy()
    tmaps.append(label1.T @ tmap_cont @ label2)
