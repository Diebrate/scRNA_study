import numpy as np
import pandas as pd
import util
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# change wdir for correct data path
os.chdir('..')
import data_input
os.chdir('clustering')

dim = 2
k = 10
labels = list(range(k))

data_raw = data_input.data
data = pd.DataFrame(columns=['PC1', 'PC2', 'time', 'cluster'])
for d, time in zip(data_raw, data_input.time_names):
    data_temp = pd.DataFrame(PCA(dim).fit_transform(d), columns=['PC1', 'PC2'])
    data_temp['time'] = time
    data = data.append(data_temp)

cluster = KMeans(k).fit((data[['PC1','PC2']]))

data['cluster'] = cluster.labels_
centroids = cluster.cluster_centers_

prob_data = []

for name in data_input.time_names:
    df_temp = data.loc[data['time'] == name]
    prob_temp = []
    for i in labels:
        prob_temp.append(sum(df_temp['cluster'] == i))
    prob_data.append(prob_temp / np.sum(prob_temp))
    
prob_data = np.array(prob_data)
    
cluster_ot = util.compute_all_ot_cluster2(centroids, prob_data, dim, reg=20)

for i in range(len(cluster_ot)):
    # cluster_ot[i] = cluster_ot[i] / cluster_ot[i].sum()
    cluster_ot[i] = np.reshape(util.renormalize_matrix(cluster_ot[i], 0), (k, k))
    plt.figure(i)
    plt.imshow(cluster_ot[i], 'magma')
    plt.title('(total) knn ' + data_input.time_names[i] + ' to ' + data_input.time_names[i+1])
    plt.colorbar()
    reform_time_name = data_input.time_names[i].replace('.', '_') + 'to' + data_input.time_names[i+1].replace('.', '_')
    plt.savefig(r'..\image\total_cluster\knn\total_knn_cluster_map_' + reform_time_name)

plt.figure(100)    
scatter = plt.scatter(data=data, x='PC1', y='PC2', s=0.5, c='cluster')
plt.legend(*scatter.legend_elements(), title='cluster')
plt.savefig(r'..\image\total_cluster\knn\total_knn_scatter')












