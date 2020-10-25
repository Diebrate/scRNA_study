# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# from ipywidgets import interact, interactive, fixed, interact_manual
# import ipywidgets as widgets
# from matplotlib import cm


import util
import data_input

n = len(data_input.time_names)

# def density_graph(i):
#     data_temp = PCA(2).fit_transform(data_input.data[i])
#     prob_temp = util.get_kernel_weight_graph(data_temp)
#     fig = plt.figure(num=i, figsize=(10, 8))
#     ax = fig.gca(projection='3d')
#     ax.plot_trisurf(data_temp[:,0], data_temp[:,1], prob_temp, cmap='Reds',linewidth=0, antialiased=True, zorder = 0.5)
#     ax.set_zlabel('Density')
#     ax.set_title(data_input.time_names[i])
    
# interact(density_graph, i=widgets.IntSlider(min=0, max=n-1, step=1, value=10))

for i in range(n):
    data_temp = PCA(2).fit_transform(data_input.data[i])
    prob_temp = util.get_kernel_weight_graph(data_temp)
    fig = plt.figure(num=i, figsize=(10, 8))
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(data_temp[:,0], data_temp[:,1], prob_temp, cmap='Reds',linewidth=0, antialiased=True, zorder = 0.5)
    ax.set_zlabel('Density')
    ax.set_title(data_input.time_names[i])
    reform_time_name = data_input.time_names[i].replace('.', '_')
    fig.savefig(r'image\density\density_' + reform_time_name)

    

