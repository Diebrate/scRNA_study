import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
import pandas as pd
import data_input
import util

import time
start_time = time.time()

pd.set_option('display.float_format', lambda x: '%.10f' % x)

dim = 2

data_raw = data_input.data
data_transformed = []
for d in data_raw:
    data_temp = util.get_weight_and_value(d, dim)
    data_transformed.append(data_temp[1])

data_transformed = np.array(data_transformed)

n = [50, 100, 500, 1000, 2000, 4000]

p_values_all = []
summary_all = []
time_p_values_all = []

# for i in range(len(n)):
#     p_values_all.append(util.test_triplet(data_transformed, util.rank_test, util.kernel_div_test, dim=dim, tail='right', metric='l1', size=n[i]))    
#     summary_all.append(pd.DataFrame(np.vstack((data_input.time_names[1:-1], p_values_all[i])).transpose()))
#     summary_all[i].columns = ['time', 'p_value']
#     summary_all[i].iloc[:,1]=[round(float(j), 10) for j in summary_all[i].iloc[:,1]]

for i in range(30):
    np.random.seed(i)
    p_values_all.append(util.test_triplet(data_transformed, util.rank_test, util.kernel_div_test, dim=dim, tail='right', metric='l1', size=1250))    
    summary_all.append(pd.DataFrame(np.vstack((data_input.time_names[1:-1], p_values_all[i])).transpose()))
    summary_all[i].columns = ['time', 'p_value']
    summary_all[i].iloc[:,1]=[round(float(j), 10) for j in summary_all[i].iloc[:,1]]
    
summary = [df.sort_values(by='p_value') for df in summary_all]


# xbar = []
# for d in data_transformed:
#     xbar.append(d[1].mean(axis=0))
# xbar = np.array(xbar)
# pc1, pc2, pc3 = xbar[:, 0], xbar[:, 1], xbar[:, 2]
#
# fig = plt.figure(figsize=(30, 5))
#
# ax1 = fig.add_subplot(131)
# ax2 = fig.add_subplot(132)
# ax3 = fig.add_subplot(133, projection='3d')
#
# ax1.plot(pc1)
# ax2.plot(pc1, pc2)
# ax3.plot3D(pc1, pc2, pc3)
#
# plt.show()


# script to save p values for different sizes n
# path = 'results\p_value_rank.xlsx'
# with pd.ExcelWriter(path) as writer:
#     for i in range(len(summary)):
#         summary[i].to_excel(writer, sheet_name='n'+str(n[i]))

print("--- %s seconds ---" % (time.time() - start_time))

