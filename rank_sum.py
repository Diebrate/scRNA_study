import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('results\p_value_rank_all_seeds.xlsx', sheet_name='rank_sum',
                   names=['time','rank'], header=None, skiprows=1)

df_time = pd.read_excel('results\p_value_rank_all_seeds.xlsx', sheet_name='seed1',
                        names=['time','p_value', 'rank'], header=None, skiprows=1)

time_names = df_time['time']

x = list(range(df.shape[0]))

# smooth the rank sum curve
z = np.polyfit(list(range(df.shape[0])), [float(i) for i in df['rank']], 9)
f = np.poly1d(z)

fig, ax = plt.subplots(figsize=(10,8))

ax.plot(x, f(x))
ax.set_xticks(x)
ax.set_xticklabels(time_names, rotation=70, size=10)
ax.set_xlabel('Time', size=20)
ax.set_ylabel('Votes', size=20)
ax.set_title(r'Time vs Votes/Rank Sum', size = 30)

# fig.savefig('image/rank_sum')
