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

reg = 0.001 # 0.05
reg1 = 1
reg2 = 50

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
tmap = solver.ot_unbalanced_all(probs[:-1, ].T, probs[1:, ].T, costm, reg=reg, reg1=reg1, reg2=reg2)
cost = solver.loss_unbalanced_all_local(probs, costm, reg, reg1, reg2=reg2, sink=sink, win_size=win_size, weight=weight, partial=True)

cp = test_util.get_cp_from_cost(cost, win_size=win_size)
cp_days = time_names[cp]
t_segments = []
for point in (list(cp) + [T - 1]):
    if t_segments:
        t_segments.append([t_segments[-1][1] + 1, point])
    else:
        t_segments.append([0, point])

cost_pair = np.zeros(T - 1)
for t in range(T - 1):
    cost_pair[t] = np.sum(costm * tmap[t])
    cost_pair[t] += reg1 * solver.kl_div(tmap[t].sum(axis=1), probs[t, ])

max_lag = 8
cost_couple = [[] for lag in range(max_lag)]
for lag in range(max_lag):
    for segment in t_segments:
        if segment[1] - segment[0] - lag > 0:
            for t in range(segment[0], segment[1] - lag):
                cost_couple[lag].append(np.sum(cost_pair[t:t+lag+1]))

cost_direct = [[] for lag in range(max_lag)]
for lag in range(max_lag):
    for segment in t_segments:
        if segment[1] - segment[0] - lag > 0:
            for t in range(segment[0], segment[1] - lag):
                tmap_temp = solver.ot_unbalanced(probs[t, ], probs[t + lag + 1, ], costm, reg=reg, reg1=reg1, reg2=reg2)
                cost_temp = np.sum(costm * tmap_temp)
                cost_temp += reg1 * solver.kl_div(tmap_temp.sum(axis=1), probs[t, ])
                cost_direct[lag].append(cost_temp)

cost_ratio = np.zeros(max_lag)
for lag in range(max_lag):
    cost_ratio[lag] = np.mean(np.array(cost_direct[lag]) / np.array(cost_couple[lag]))

# compare cost_couple and cost_direct
plt.plot(cost_ratio)
plt.title('ratio of direct cost over coupled cost')
plt.xlabel('number of omitted time points')
plt.ylabel('cost ratio')
# add a horizontal line showing threshold
plt.axhline(y=0.5, color='r', linestyle='--')
plt.show()
