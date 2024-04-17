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

time_names = pd.read_csv('../data/time_names.csv', index_col=0)
time_names = time_names['time'].to_numpy()

time_names = time_names[time_names != 'D8.25']
time_names = time_names[time_names != 'D8.75']

# load data
df = pd.read_csv('../data/proc_data/ot_df.csv')
df = df[df['day'] != 8.25]
df = df[df['day'] != 8.75]
time_labels = [float(i[1:]) for i in time_names]
sample_size = pd.DataFrame()
sample_size['day'] = time_labels
sample_size['sample size'] = df.groupby('day').size().to_numpy()
sample_size.to_csv('../results/sample_size.csv')

T = len(time_labels)
k = len(np.unique(df['cell type']))

centroids = df[['x', 'y', 'cell type']].groupby('cell type').mean().to_numpy()
xy_sc = df[['x', 'y']].to_numpy() # continuous data
dim = centroids.shape[1]

### convert to multinomial or dirac measure
type_list = df['cell type'].unique()
probs = []
for t in range(T):
    p_temp = df['cell type'][df['day'] == time_labels[t]].value_counts(normalize=True).reindex(type_list, fill_value=0).to_numpy()
    probs.append(p_temp)
probs = np.array(probs)
costm = test_util.get_cost_matrix(centroids, centroids, dim=dim)
# costm = costm / 100000
# costm = costm / np.median(costm)
tmap = solver.ot_unbalanced_all(probs[:-1, ].T, probs[1:, ].T, costm, reg=reg, reg1=reg1, reg2=50)
cost = solver.loss_unbalanced_all_local(probs, costm, reg, reg1, reg2=50, sink=sink, win_size=win_size, weight=weight, partial=True)

tmap_direct = []
for t in range(T-1):
    tmap_temp = solver.ot_unbalanced(probs[t, ], probs[t+1, ], costm, reg=reg, reg1=reg1, reg2=50)
    # append the normalized tmap by row
    tmap_direct.append(tmap_temp / tmap_temp.sum(axis=1)[:, None])

# calculate the coupled tmaps
tmap_couple = [np.eye(k)]

for t in range(T-1):
    # normalize tmap[t] by row
    tmap_temp = tmap[t] / tmap[t].sum(axis=1)[:, None]
    tmap_couple.append(tmap_couple[-1] @ tmap_temp)
tmap_couple = tmap_couple[1:]

# calculate distance between coupled and direct tmaps
dist = []
for t in range(T - 1):
    dist.append((tmap_couple[t] * (np.log(tmap_couple[t]) - np.log(tmap_direct[t]))).sum())

# plot the distance as a function of time lag
plt.plot(dist)
plt.show()
