import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import sys
sys.path.insert(0,'..')
import test_util
import solver
os.chdir('..\preproc')

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
# costm = costm / np.median(costm)
res_tune = solver.optimal_lambda_ts(probs, costm, reg, 0, 0.1, grid_size=50)
reg_opt = res_tune['opt_lambda']
# reg1 = reg2 = reg_opt
tmap = solver.ot_unbalanced_all(probs[:-1, ].T, probs[1:, ].T, costm, reg=reg, reg1=reg1, reg2=50)
cost = solver.loss_unbalanced_all_local(probs, costm, reg, reg1, reg2=50, sink=sink, win_size=win_size, weight=weight, partial=True)

### normalization
# cost = cost / np.sum(cost)
# cost = np.append(cost, np.nan)

# cp = test_util.get_cp_from_cost(cost, win_size=win_size)

if do_cost:
    local_cost_disc = solver.compute_cost_disc(probs, costm, reg, reg1, reg2, sink=sink, partial=True, max_win_size=4)
    if do_save:
        np.save(r'..\results\cost_local_disc_wot.npy', local_cost_disc)

if do_load:
    local_cost_disc = np.load(r'..\results\cost_local_disc_wot.npy')

cost_disc_m = [solver.get_weighted_cost(local_cost_disc, weight=weight, win_size=i) for i in [1, 2, 3, 4]]
cp = []
cp_days = []

### normalizing the cost
for i in range(4):
    # cost_disc_m[i] = cost_disc_m[i] / np.sum(cost_disc_m[i])
    cp.append(test_util.get_cp_from_cost(cost_disc_m[i], win_size=i + 1))
    cp_days.append(time_names[cp[i]])

x = list(range(T - 1))
# z = np.polyfit(x, [float(i) for i in cost], 7)
# f = np.poly1d(z)

graph_cost0 = False

if graph_cost0:

    fig, ax = plt.subplots(figsize=(10,8))

    ax.plot(x, cost_disc_m[win_size - 1])
    ax.set_xticks(x)
    ax.set_xticklabels(time_names[:-1], rotation=70, size=10)
    ax.set_xlabel('Time', size=20)
    ax.set_ylabel('Loss', size=20)
    ax.set_title(r'Time vs Loss with PHATE Embedding', size = 20)
    plt.legend()
    # plt.savefig(r'..\image\wot_res')

graph_cost = False

if graph_cost:

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
    # plt.savefig(r'..\image\wot_res_all')

graph_tmap = False

if graph_tmap:
    for i in range(T - 1):
        fig, ax = plt.subplots(num=i + 2)
        img = plt.imshow(tmap[i], 'viridis')
        plt.title('Transport map from ' + time_names[i] + ' to ' + time_names[i+1])
        cbar = plt.colorbar()
        cbar.set_label('probability')
        ax.set_xticklabels(np.insert(type_list, 0, ''), rotation=20)
        ax.set_yticklabels(np.insert(type_list, 0, ''))
        reform_time_name = time_names[i].replace('.', '_') + 'to' + time_names[i+1].replace('.', '_')
        # plt.savefig(r'..\image\wot_tmap\transport_map_'+reform_time_name)

graph_tmap_hist = True

from matplotlib.transforms import Bbox

if graph_tmap_hist:
    for i in range(T - 1):
        fig, axs = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(8, 8),
                                gridspec_kw=dict(height_ratios=[1, 2.5], width_ratios=[2.5, 1],
                                                 hspace=0.01, wspace=0.02))
        axs[0, 1].set_visible(False)
        axs[0, 0].set_box_aspect(1/2.5)
        axs[1, 1].set_box_aspect(2.5/1)

        # Plot the heatmap
        im = axs[1, 0].imshow(tmap[i], 'viridis')

        # Create colorbar with label at the bottom
        cbar = fig.colorbar(im, ax=axs[1, 0], location="bottom")
        cbar.ax.set_xlabel('Probability', rotation=0, ha="center")

        # Adjust colorbar position and size
        cbar.ax.set_position([0.08, 0.15, 0.65, 0.02])

        # Show all ticks and label them with the respective list entries.
        axs[1, 0].set_xticks(np.arange(k), labels=type_list)
        axs[1, 0].set_yticks(np.arange(k), labels=type_list)

        # Rotate the tick labels and set their alignment.
        plt.setp(axs[1, 0].get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        # Turn spines off and remove gridlines for the heatmap
        axs[1, 0].spines[:].set_visible(False)
        axs[1, 0].xaxis.grid(False)
        axs[1, 0].yaxis.grid(False)

        axs[1, 0].set_xticks(np.arange(k + 1) -.5, minor=True)
        axs[1, 0].set_yticks(np.arange(k + 1) -.5, minor=True)
        axs[1, 0].grid(which="minor", color="w", linestyle='-', linewidth=3)
        axs[1, 0].tick_params(which="minor", bottom=False, left=False)

        plt.setp(axs[1,0].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Remove gridlines for histograms
        axs[1, 1].xaxis.grid(False)
        axs[1, 1].yaxis.grid(False)
        axs[0, 0].xaxis.grid(False)
        axs[0, 0].yaxis.grid(False)

        # Rotate the tick labels and set their alignment for histograms
        plt.setp(axs[1, 1].get_xticklabels(), rotation=0, ha="center")
        plt.setp(axs[0, 0].get_xticklabels(), rotation=0, ha="center")

        # Rotate the tick labels and set their alignment for row histogram
        axs[1, 1].set_yticklabels(type_list, rotation=0, ha="center")
        axs[1, 1].set_xticks(axs[1, 1].get_xticks(), minor=False)

        # Rotate the tick labels and set their alignment for column histogram
        axs[0, 0].set_xticklabels(type_list, rotation=45, ha="right", minor=False)

        # Plot the histograms
        axs[1, 1].barh(y=type_list, width=tmap[i].sum(axis=1))
        axs[0, 0].bar(x=type_list, height=tmap[i].sum(axis=0))

        # Resize
        (x0m, y0m), (x1m, y1m) = axs[1, 0].get_position().get_points()  # main heatmap
        (x0h, y0h), (x1h, y1h) = axs[0, 0].get_position().get_points()  # horizontal histogram
        axs[0, 0].set_position(Bbox([[x0m, y0h], [x1m, y1h]]))
        (x0v, y0v), (x1v, y1v) = axs[1, 1].get_position().get_points()  # vertical histogram
        axs[1, 1].set_position(Bbox([[x0v, y0m], [x1v, y1m]]))

        # Set separate titles for the histograms
        axs[1, 1].set_title('Cell type distribution at ' + time_names[i])
        axs[0, 0].set_title('Cell type distribution at ' + time_names[i+1])

        fig.suptitle('Transport map from ' + time_names[i] + ' to ' + time_names[i+1])

        reform_time_name = time_names[i].replace('.', '_') + 'to' + time_names[i+1].replace('.', '_')
        # plt.savefig(r'..\image\wot_tmap\transport_map_hist_'+reform_time_name)

graph_all = False

if graph_all:
    fig, axs = plt.subplots(9, 4, figsize=(36, 85))
    for i in range(T - 1):
        ax = axs[i // 4, i % 4]
        img = ax.imshow(tmap[i], 'viridis')
        ax.set_title('Transport map from ' + time_names[i] + ' to ' + time_names[i+1])
        ax.set_xticks(np.arange(len(type_list)))
        ax.set_yticks(np.arange(len(type_list)))
        ax.set_xticklabels(type_list, rotation=90)
        ax.set_yticklabels(type_list)
        ax.label_outer()
        reform_time_name = time_names[i].replace('.', '_') + 'to' + time_names[i+1].replace('.', '_')
    cbar = fig.colorbar(cm.ScalarMappable(cmap='viridis'), ax=axs, orientation='horizontal', anchor=(0.2, -0.7), aspect=40)
    cbar.set_label('probability')
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    # plt.savefig(r'../image/tmap_all.png')

print(cp_days)

# EDA

do_EDA = False
if do_EDA:

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)

    # Plot each category as a line
    for category in range(k):
        ax.plot(range(T), probs[:, category], label=type_list[category])

    # Add labels and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.set_title('Evolution of Cell Type Distribution over Time')

    # Set the x-axis ticks and labels
    ax.set_xticks(range(T))
    ax.set_xticklabels(time_names, rotation=70, size=10)

    ax.legend()

    # Adjust the layout to prevent label cutoff
    plt.tight_layout()

    # histogram

    # Calculate the number of rows and columns for the subplot grid
    rows = int(np.ceil(np.sqrt(T)))
    cols = int(np.ceil(T / rows))

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(rows, cols, figsize=(12, 8))

    # Flatten the axs array for easy indexing
    axs = axs.flatten()

    # Create histograms for each row of probs
    for i in range(T):
        ax = axs[i]
        ax.hist(probs[i], bins=20, range=(0, 1), alpha=0.5)
        ax.set_title(time_names[i])
        ax.set_xlabel('Cell Type')
        ax.set_ylabel('Probability')

    # Hide any remaining empty subplots
    for i in range(T, rows * cols):
        axs[i].axis('off')
    plt.tight_layout()

print("--- %s seconds ---" % (time.time() - start_time))