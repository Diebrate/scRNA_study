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

reg = 0.0001 # 0.05
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


# Define Morandi-style colors (You can adjust these hex colors)
morandi_colors = ['#abbdbe', '#c0b2a5', '#d4a5a5', '#929a88', '#7e7f9a', '#6e7582']

# Create a custom colormap
morandi_cmap = LinearSegmentedColormap.from_list("morandi", morandi_colors, N=256)

# Histogram bar color
hist_bar_color = '#929a88'  # A color from the Morandi palette


for i in range(T - 1):
    fig, axs = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(8, 8),
                            gridspec_kw=dict(height_ratios=[1, 2.5], width_ratios=[2.5, 1],
                                             hspace=0.01, wspace=0.02))
    axs[0, 1].set_visible(False)
    axs[0, 0].set_box_aspect(1/2.5)
    axs[1, 1].set_box_aspect(2.5/1)

    # Plot the heatmap
    im = axs[1, 0].imshow(tmaps[i], morandi_cmap)

    # Create colorbar with label at the bottom
    cbar = fig.colorbar(im, ax=axs[1, 0], location="bottom")
    cbar.ax.set_xlabel('Probability', rotation=0, ha="center")

    # Adjust colorbar position and size
    cbar.ax.set_position([0.08, 0.15, 0.65, 0.02])

    # Show all ticks and label them with the respective list entries.
    axs[1, 0].set_xticks(np.arange(k))
    axs[1, 0].set_xticklabels(type_list)
    axs[1, 0].set_yticks(np.arange(k))
    axs[1, 0].set_yticklabels(type_list)

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
    axs[1, 1].barh(y=type_list, width=tmaps[i].sum(axis=1), color=hist_bar_color)
    axs[0, 0].bar(x=type_list, height=tmaps[i].sum(axis=0), color=hist_bar_color)

    # Set y-axis limits
    axs[1, 1].set_xlim(0, 1)
    axs[0, 0].set_ylim(0, 1)

    for side in ['top', 'right', 'left', 'bottom']:
        axs[1, 1].spines[side].set_visible(False)
        axs[0, 0].spines[side].set_visible(False)

    # Resize
    (x0m, y0m), (x1m, y1m) = axs[1, 0].get_position().get_points()  # main heatmap
    (x0h, y0h), (x1h, y1h) = axs[0, 0].get_position().get_points()  # horizontal histogram
    axs[0, 0].set_position(Bbox([[x0m, y0h], [x1m, y1h]]))
    (x0v, y0v), (x1v, y1v) = axs[1, 1].get_position().get_points()  # vertical histogram
    axs[1, 1].set_position(Bbox([[x0v, y0m], [x1v, y1m]]))

    # Set title for the vertical histogram
    # axs[1, 1].set_title('Cell type distribution at ' + time_names[i])
    axs[0, 0].set_title('Cell type distribution at ' + time_names[i+1])

    # Rotate the title and move it to the right of the horizontal histogram
    axs[1, 1].set_title('Cell type distribution at ' + time_names[i],
                        rotation=-90,
                        # fontsize=10,
                        x=1.25,
                        y=0.15,
                        ha='center')

    fig.suptitle('Transport map from ' + time_names[i] + ' to ' + time_names[i+1])

    reform_time_name = time_names[i].replace('.', '_') + 'to' + time_names[i+1].replace('.', '_')
    plt.savefig(r'../image/wot_tmap/transport_map_cont_hist_'+reform_time_name, bbox_inches='tight')