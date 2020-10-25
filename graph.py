import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import data_input

# import for prediction graph
import numpy as np
import util


# seaborn option
# import seaborn as sns

# t = 30
# lag = 4
# x_all = []
# day_names = []

# for i in range(t, t+lag):
#     x_all.append(PCA(2).fit_transform(data_input.data[i]))
#     day_names.append(data_input.time_names[i])

# # colours = ['red', 'blue', 'cyan', 'black', 'green', 'magenta']
# # markers = ['.', '<', '>', '^', ',', 'o']


# for i in range(lag):
#     x_temp = x_all[i].transpose()
#     plt.scatter(x_temp[0], x_temp[1], s=0.5, label=day_names[i])

# D13.5 to D14.5, compare D13 index 29 and 31
s = 10
x = data_input.data[s]
y = data_input.data[s+2]
x = PCA(2).fit_transform(x)
y = PCA(2).fit_transform(y)

frac = (data_input.time_values[s+1] - data_input.time_values[s]) / (data_input.time_values[s+2] - data_input.time_values[s])

# make sample smaller
size = 1000
# x = util.get_sample(x, size)
# y = util.get_sample(y, size)

px = np.repeat(1/len(x), len(x))
py = np.repeat(1/len(y), len(y))
M = util.ot_iter(px, py, x, y, dim=2, n_iter=1)

pred = util.interpolate(x, y,px, py, M, size=size, frac=frac)
# pred = util.interpolate_with_noise(x, y, M, size=1000)
# pred = util.interpolate_with_kernel1(x, y, M, size)
pred_kernel = util.interpolate_with_kernel(x, y, M, size=size, frac=frac)
obs = data_input.data[s+1]
obs = PCA(2).fit_transform(obs)

# make sample smaller
# obs = util.get_sample(obs, size)

plt.scatter(pred[:, 0], pred[:, 1], s=0.5, label='pred')
plt.scatter(pred_kernel[:, 0], pred_kernel[:, 1], s=0.5, label='pred_kernel')
plt.scatter(obs[:, 0], obs[:, 1], s=0.5, label='obs')


# t, lag = s-1, 2
# x_all, day_names = [], []
# for i in [t, t+2]:
#     x_all.append(PCA(2).fit_transform(data_input.data[i]))
#     day_names.append(data_input.time_names[i])

# for i in range(lag):
#     x_temp = x_all[i].transpose()
#     plt.scatter(x_temp[0], x_temp[1], s=0.5, label=day_names[i])


plt.legend()
plt.show()

# sns.kdeplot(x1[:,0], x1[:,1], shade=True)


