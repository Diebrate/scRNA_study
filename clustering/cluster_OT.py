import numpy as np
import util
import os
import matplotlib.pyplot as plt
from sklearn.neighbors.nearest_centroid import NearestCentroid

# change wdir for correct data path
os.chdir('..')
import data_input
os.chdir('clustering')

dim = 2

data_raw = data_input.data
data_transformed = []
for d in data_raw:
    data_temp = util.get_weight_and_value(d, dim)
    data_transformed.append(data_temp[1])

data_transformed = np.array(data_transformed)

data_clustered = []
labels = []
probs = []
centroids = []


# knn clustering
# for data in data_transformed:
#     cluster_temp = util.cluster_with_best_k(data)
#     data_clustered.append(cluster_temp)
#     labels.append(cluster_temp.labels_)
#     probs.append(util.get_prob_cluster(cluster_temp.labels_))
#     centroids.append(cluster_temp.cluster_centers_)
    

# hierarchical clustering
for data in data_transformed:
    cluster_temp = util.cluster_with_best_k_hier(data)
    data_clustered.append(cluster_temp)
    labels.append(cluster_temp.labels_)
    probs.append(util.get_prob_cluster(cluster_temp.labels_))
    clf = NearestCentroid().fit(data, cluster_temp.labels_)
    centroids.append(clf.centroids_)


labels = np.array(labels)
probs = np.array(probs)
centroids = np.array(centroids)

cluster_ot = util.compute_all_ot_cluster(centroids, labels, dim, reg=20)


# knn ot map
# for i in range(len(cluster_ot)):
#     # cluster_ot[i] = cluster_ot[i] / cluster_ot[i].sum()
#     k, l = cluster_ot[i].shape
#     cluster_ot[i] = np.reshape(util.renormalize_matrix(cluster_ot[i], 0), (k, l))
#     plt.figure(i)
#     plt.imshow(cluster_ot[i], 'magma')
#     plt.title('knn ' + data_input.time_names[i] + ' to ' + data_input.time_names[i+1])
#     plt.colorbar()
#     reform_time_name = data_input.time_names[i].replace('.', '_') + 'to' + data_input.time_names[i+1].replace('.', '_')
#     plt.savefig(r'..\image\knn_cluster\ot_map\knn_cluster_map_'+reform_time_name)


# hierarchy ot map
for i in range(len(cluster_ot)):
    # cluster_ot[i] = cluster_ot[i] / cluster_ot[i].sum()
    k, l = cluster_ot[i].shape
    cluster_ot[i] = np.reshape(util.renormalize_matrix(cluster_ot[i], 0), (k, l))
    plt.figure(i)
    plt.imshow(cluster_ot[i], 'magma')
    plt.title('hierarchy ' + data_input.time_names[i] + ' to ' + data_input.time_names[i+1])
    plt.colorbar()
    reform_time_name = data_input.time_names[i].replace('.', '_') + 'to' + data_input.time_names[i+1].replace('.', '_')
    plt.savefig(r'..\image\hier_cluster\ot_map\h_cluster_map_'+reform_time_name)
    