import numpy as np
import test_util

import time
start_time = time.time()

n = 1
n_sub = 10

# centroids = np.array([[0,1],[1,0],[1,1],[0,0]])
# centroids = np.array([[0,1],[1,0]])
# centroids = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
centroids = np.array([[1, 1], [2, -1], [-1, 0.5], [-1, -3]])

M = test_util.get_cost_matrix(centroids, centroids, 2)
M_offdiag = test_util.get_offdiag(M)

k = M.shape[0]
# p_start = np.ones(k)/k
# p_start = np.random.rand(k)
# p_start = p_start / np.sum(p_start)
p_start = np.array([0.5, 0.3, 0.1, 0.1])
# p_trans = np.diag(np.repeat(1, k))
p_trans = np.array([[0.5, 0.5,   0,   0],
                    [  0, 0.5, 0.5,   0],
                    [  0,   0, 0.5, 0.5],
                    [0.5,   0,   0, 0.5]])
var = np.diag([M_offdiag.min() / 4, M_offdiag.min() / 4])

# x, y = test_util.simulate(k=k, p_start=p_start, p_trans=p_trans)

# test = test_util.perm_test(test_util.ot_map_test, x, y, tail='right', n_times=10000, M=M, k=k)

test_total = []

# for i in range(n):
#     x_temp, y_temp = test_util.simulate(k=k, p_start=p_start, p_trans=p_trans, size=50)
#     test_temp = test_util.perm_test(test_util.ot_map_test3, x_temp, y_temp, tail='right', n_times=5000, timer=False, M=M, k=k)
#     test_total.append(test_temp)
#     if (i + 1) % n_sub == 0:
#         print('Currently at test number ' + str(i + 1))

for i in range(n):
    lx_temp, ly_temp = test_util.simulate2d(p_start=p_start, p_trans=p_trans, mean=centroids, var=var, size=200)
    test_temp = test_util.perm_test(test_util.ot_map_test1, lx_temp, ly_temp, tail='right', n_times=2000, timer=False, M=M, k=k, reg=1)
    # test_temp = test_util.perm_test(test_util.ot_map_test, lx_temp, ly_temp, tail='right', n_times=2000, timer=False, cluster=True, M=M, k=k, reg=100)
    test_total.append(test_temp)
    if (i + 1) % n_sub == 0:
        print('Currently at test number ' + str(i + 1))

p_value = np.array([t['p_value'] for t in test_total])

prop_reject = (p_value<=0.05).sum() / len(p_value)
        
print("--- %s seconds ---" % (time.time() - start_time))