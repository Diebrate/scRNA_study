import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib import cm
import numpy as np

# graph p_values
n = [50, 100, 500, 1000, 2000, 4000]

# fig, ax = plt.subplots(2, 3, constrained_layout=True, figsize=(16, 8))

# for i in range(len(n)):
#     df = pd.read_excel('results\p_value_rank.xlsx', sheet_name='n'+str(n[i]),
#                        names=['time','p_value'], header=None, skiprows=1)
#     df.loc[:, 'time'] = [float(i[1:]) for i in df.loc[:, 'time']]
#     df = df.sort_values(by='time')
#     time, p_value = df.loc[:, 'time'], 1 - df.loc[:, 'p_value']
#     p_value = p_value / np.sum(p_value)
#     # ax[i//3, i%3].plot(time, np.log(p_value/(1-p_value)))
#     ax[i//3, i%3].plot(time, p_value)
#     ax[i//3, i%3].set_title('n'+str(n[i]))
#     # fig.colorbar(ax[i//3, i%3])
    

fig_r, ax_r = plt.subplots(7, 1, constrained_layout=True, figsize=(10, 6))

for i in range(6):
    df = pd.read_excel('results\p_value_rank.xlsx', sheet_name='n'+str(n[i]),
                       names=['time','p_value'], header=None, skiprows=1)
    df.loc[:, 'time'] = [float(i[1:]) for i in df.loc[:, 'time']]
    df['rank'] = list(range(1, len(df.index) + 1))
    df = df.sort_values(by='time')
    time, rank = df.loc[:, 'time'], df.loc[:, 'rank']
    ax_r[i].imshow(np.vstack((rank, rank)))
    ax_r[i].text(s='n'+str(n[i]), x=-3, y=0.5, fontsize=12)
    ax_r[i].get_xaxis().set_visible(False)
    ax_r[i].get_yaxis().set_visible(False)


rank = list(range(1, len(df.index) + 1))
ax_r[6].imshow(np.vstack((rank, rank)))
ax_r[6].text(s='ref', x=-3, y=0.5, fontsize=12)
ax_r[6].get_yaxis().set_visible(False)
ax_r[6].get_yaxis().set_visible(False)



plt.show()
