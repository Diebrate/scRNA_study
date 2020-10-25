import os
import gc
# import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import util

# change wdir for correct data path
os.chdir('..')
import data_input
os.chdir('clustering')

dim = 2
index = range(0, len(data_input.time_names), 1)

# knn clustering
# for i in index:
#     plt.figure(i)
#     data = PCA(dim).fit_transform(data_input.data[i])
#     data_clustered = util.cluster_with_best_k(data)
#     scatter = plt.scatter(data[:,0], data[:,1], c=data_clustered.labels_, s=0.75)
#     plt.legend(*scatter.legend_elements(), title="Clusters")
#     plt.title('knn ' + data_input.time_names[i])
#     reform_time_name = data_input.time_names[i].replace('.', '_')
#     plt.savefig(r'..\image\knn_cluster\cluster_graph\knn_cluster_' + reform_time_name)
#     gc.collect()

# knn clustering with density
for i in index:
    plt.figure(i)
    data = PCA(dim).fit_transform(data_input.data[i])
    prob = util.get_kernel_weight_graph(data)
    data_clustered = util.cluster_with_best_k(data)
    
    fig = plt.figure(num=i, figsize=(20, 12))
    ax = fig.gca(projection='3d')
    scatter = ax.scatter(data[:,0], data[:,1], prob, c=data_clustered.labels_, s=0.6)
    ax.set_zlabel('Density')
    ax.set_title(data_input.time_names[i])
    
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title('knn ' + data_input.time_names[i])
    reform_time_name = data_input.time_names[i].replace('.', '_')
    plt.savefig(r'..\image\knn_cluster\cluster_with_density\knn_cluster_with_density_' + reform_time_name)
    gc.collect()
    
# hierarchical clustering
# for i in index:
#     plt.figure(i)
#     data = PCA(dim).fit_transform(data_input.data[i])
#     data_clustered = util.cluster_with_best_k_hier(data)
#     scatter = plt.scatter(data[:,0], data[:,1], c=data_clustered.labels_, s=0.75)
#     plt.legend(*scatter.legend_elements(), title="Clusters")
#     plt.title('hierarchy ' + data_input.time_names[i])
#     reform_time_name = data_input.time_names[i].replace('.', '_')
#     plt.savefig(r'..\image\hier_cluster\cluster_graph\h_cluster_' + reform_time_name)
#     gc.collect()

# hierarchical clustering with density
# for i in index:
#     plt.figure(i)
#     data = PCA(dim).fit_transform(data_input.data[i])
#     prob = util.get_kernel_weight_graph(data)
#     data_clustered = util.cluster_with_best_k_hier(data)
    
#     fig = plt.figure(num=i, figsize=(20, 12))
#     ax = fig.gca(projection='3d')
#     scatter = ax.scatter(data[:,0], data[:,1], prob, c=data_clustered.labels_, s=0.6)
#     ax.set_zlabel('Density')
#     ax.set_title(data_input.time_names[i])
    
#     plt.legend(*scatter.legend_elements(), title="Clusters")
#     plt.title('hierarchy ' + data_input.time_names[i])
#     reform_time_name = data_input.time_names[i].replace('.', '_')
#     plt.savefig(r'..\image\hier_cluster\cluster_with_density\h_cluster_with_density_' + reform_time_name)
#     gc.collect()

plt.show()