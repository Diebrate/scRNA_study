import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

graph = False
balanced = False

df_rank = pd.DataFrame()
method_list = ['raw', 'pc', 'tsne', 'umap', 'phate']

for m in method_list:
    if balanced:
        df_rank[m] = np.load(r'..\data\proc_data\pp_rank_sum_balanced_' + m + r'_1.npy')
    else:
        df_rank[m] = np.load(r'..\data\proc_data\pp_rank_sum_unbalanced_' + m + r'_1_1_50.npy')

df = pd.read_csv(r'..\data\proc_data\proc_df_no_batch.csv')
time_names = pd.read_csv(r'..\data\time_names.csv', index_col=0)
time_names = time_names['time'].to_numpy()

T = len(time_names)
k = df['cluster'].max() + 1
x = list(range(T - 1))

fig, ax = plt.subplots(figsize=(10,8))
for m in method_list:
    z = np.polyfit(x, [float(i) for i in df_rank[m]], 7)
    f = np.poly1d(z)
    ax.plot(x, f(x), label=m)
ax.set_xticks(x)
ax.set_xticklabels(time_names[:-1], rotation=70, size=10)
ax.set_xlabel('Time', size=20)
ax.set_ylabel('Rank', size=20)
ax.legend()
if balanced:
    ax.set_title(r'Time vs Rank for Balanced Transport', size = 20)
else:
    ax.set_title(r'Time vs Rank for Unbalanced Transport', size = 20)

