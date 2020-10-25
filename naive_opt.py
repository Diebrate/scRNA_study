# import pandas as pd
import numpy as np
import ot
from sklearn.linear_model import LinearRegression as LR
from matplotlib import pyplot as plt
import util
import data_input

mat_1 = data_input.read_data('data\GSE122662_RAW\GSM3195668_D5_Dox_C1_gene_bc_mat.h5', 'h5')
mat_2 = data_input.read_data('data\GSE122662_RAW\GSM3195672_D6_Dox_C1_gene_bc_mat.h5', 'h5')
mat_3 = data_input.read_data('data\GSE122662_RAW\GSM3195670_D5.5_Dox_C1_gene_bc_mat.h5', 'h5')

data1 = util.get_weight_and_value(mat_1, 1, use_kernel=True)
data2 = util.get_weight_and_value(mat_2, 1, use_kernel=True)
data3 = util.get_weight_and_value(mat_3, 1, use_kernel=True)

w1, x1 = data1[0], data1[1]
w2, x2 = data2[0], data2[1]
w3, x3 = data3[0], data3[1]

l1, l2, = len(x1), len(x2)

dist_mat = util.get_cost_matrix(x1, x2, dim=1)

ot_matrix = ot.unbalanced.sinkhorn_stabilized_unbalanced(w1, w2, dist_mat, 1, 1)
ot_matrix2 = util.ot_iter(w1, w2, x1, x2, t1=0, t2=2, dim=1, reg=1, reg_m=5, n_iter=10)

half_pred = util.interpolate(x1, x2, w1, w2, ot_matrix, l2, 0.5)
w_half_pred = half_pred[0]

half_pred2 = util.interpolate(x1, x2, w1, w2, ot_matrix2, l2, 0.5)
w_half_pred2 = half_pred2[0]

anc = x1[[i//l2 for i in half_pred[2]]]
des = half_pred[1]

anc2 = x1[[i//l2 for i in half_pred2[2]]]
des2 = half_pred2[1]

mod = LR().fit(np.reshape(anc, (-1, 1)), np.reshape(des, (-1, 1)))

mod2 = LR().fit(np.reshape(anc2, (-1, 1)), np.reshape(des2, (-1, 1)))

plt.hist([x3, des2], bins=50, label=['obs', 'pred'], density=True)

plt.legend(prop={'size': 10})



