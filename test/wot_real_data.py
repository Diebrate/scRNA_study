import numpy as np
import pandas as pd
import anndata
import matplotlib.pyplot as plt
import util
import solver

import os
import test_util
os.chdir('..\preproc')

import time
start_time = time.time()

sink = True
win_size = 3
weight = None

do_cost = True
do_load = True

reg = 0.01
reg1 = 1
reg2 = 1

time_names = pd.read_csv(r'..\data\time_names.csv', index_col=0)
time_names = time_names['time'].to_numpy()

time_names = time_names[time_names != 'D8.25']
time_names = time_names[time_names != 'D8.75']

# df = pd.read_csv(r'..\wot\data\full_df.csv')
df = pd.read_csv(r'..\data\proc_data\ot_df.csv')
df = df[df['day'] != 8.25]
df = df[df['day'] != 8.75]
time_labels = [float(i[1:]) for i in time_names]
sample_size = pd.DataFrame()
sample_size['day'] = time_labels
sample_size['sample size'] = df.groupby('day').size().to_numpy()
sample_size.to_csv(r'..\results\sample_size.csv')

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
# costm = costm / np.sum(costm)
reg_opt = solver.optimal_lambda_ts(probs, costm, reg, 0, 10)['opt_lambda']
# reg1 = reg2 = reg_opt
tmap = solver.ot_unbalanced_all(probs[:-1, ].T, probs[1:, ].T, costm, reg=reg, reg1=reg_opt, reg2=50)
cost = solver.loss_unbalanced_all_local(probs, costm, reg, reg1, reg2, sink=sink, win_size=win_size, weight=weight, partial=True)

### normalization
# cost = cost / np.sum(cost)
# cost = np.append(cost, np.nan)

# cp = test_util.get_cp_from_cost(cost, win_size=win_size)

if do_cost:
    local_cost_disc = solver.compute_cost_disc(probs, costm, reg, reg1, reg2, sink=sink, partial=True, max_win_size=4)
    np.save(r'..\results\cost_local_disc_wot.npy', local_cost_disc)

if do_load:
    local_cost_disc = np.load(r'..\results\cost_local_disc_wot.npy')
    
cost_disc_m = [solver.get_weighted_cost(local_cost_disc, weight=weight, win_size=i) for i in [1, 2, 3, 4]]
cp = []
cp_days = []

### normalizing the cost
for i in range(4):
    cost_disc_m[i] = cost_disc_m[i] / np.sum(cost_disc_m[i])
    cp.append(test_util.get_cp_from_cost(cost_disc_m[i], win_size=i + 1))
    cp_days.append(time_names[cp[i]])

x = list(range(T - 1))
# z = np.polyfit(x, [float(i) for i in cost], 7)
# f = np.poly1d(z)

fig, ax = plt.subplots(figsize=(10,8))

# ax.plot(x, f(x))
# ax.plot(x, cost, label='full')
# ax.plot(x, cost_partial, label='partial')
# ax.legend()
for i in range(4):
    ax.plot(x, cost_disc_m[i], label='clustered with win-size=' + str(i + 1))
ax.set_xticks(x)
ax.set_xticklabels(time_names[:-1], rotation=70, size=10)
ax.set_xlabel('Time', size=20)
ax.set_ylabel('Loss', size=20)
ax.set_title(r'Time vs Loss with PHATE Embedding', size = 20)
plt.legend()
plt.savefig(r'..\image\wot_res')

graph_tmap = False

if graph_tmap:
    for i in range(T - 1):
        fig, ax = plt.subplots(num=i + 2)
        img = plt.imshow(tmap[i], 'magma')
        plt.title('Transport map from ' + time_names[i] + ' to ' + time_names[i+1])
        cbar = plt.colorbar()
        cbar.set_label('probability')
        ax.set_xticklabels(np.insert(type_list, 0, ''), rotation=20)
        ax.set_yticklabels(np.insert(type_list, 0, ''))
        reform_time_name = time_names[i].replace('.', '_') + 'to' + time_names[i+1].replace('.', '_')
        plt.savefig(r'..\image\wot_tmap\transport_map_'+reform_time_name)

print("--- %s seconds ---" % (time.time() - start_time))