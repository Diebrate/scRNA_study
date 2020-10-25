import sim_pop
import util
import seaborn as sns
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt

start_mean, start_var = 0, 3
pop_size, n_step = 5000, 10
drift, sigma = 0, 1
prob, imp_size, imp_var = 0.5, 3, 0.5
sample_size = 1000

population = sim_pop.generate_pop_time_series(start_mean=start_mean,
                                              start_var=start_var,
                                              size=pop_size,
                                              n_step=n_step,
                                              drift=drift,
                                              sigma=sigma)

# population = sim_pop.random_impulse(prob=prob, imp_size=imp_size, imp_var=imp_var, pop_time_series=population)
population = sim_pop.fixed_impulse(imp_size=imp_size, imp_var=imp_var, pop_time_series=population, add_time=5)

sample = sim_pop.sample_from_pop(population, size=sample_size)

population_t = population.transpose()

sample_t = sample.transpose()

# for i in range(size):
#     plt.plot(np.arange(n_step)+1, population_t[i])

# for i in range(sample_size):
#     plt.scatter(np.arange(n_step)+1, sample_t[i])

x2 = sample[2]
x3 = sample[3]
x4 = sample[4]
x5 = sample[5]
x6 = sample[6]

n = x4.size
x_weight = np.repeat(1/n, n)

cost = util.get_cost_matrix(x4, x6)

ot_matrix = util.ot_iter(x_weight, x_weight, x4, x6, t1=0, t2=2, reg=1, reg_m=1)

x5_pred = util.interpolate(x4, x6, x_weight, x_weight, cost, size=2000, frac=0.5)[1]

# test_statistics = scipy.stats.ranksums(x5_pred, x5)

test = util.rank_test(util.kernel_div_test, x5_pred, x5, metric='l1', dim=1, n_points=100, tail='right')

# plt.figure(1)
# sns.distplot(x4, kde=True, rug=False, hist=False, label='x4')
# sns.distplot(x6, kde=True, rug=False, hist=False, label='x6')
# sns.distplot(x5, kde=True, rug=False, hist=False, label='x5: p_value=0
# sns.distplot(x5_pred, kde=True, rug=False, hist=False, label='x5_pred')

ot_matrix_2 = util.ot_iter(x_weight, x_weight, x2, x4, dim=1, t1=0, t2=2, reg=1, reg_m=1)

x3_pred = util.interpolate(x2, x4, x_weight, x_weight, cost, size=2000, frac=0.5)[1]

test2 = util.rank_test(util.kernel_div_test, x3_pred, x3, metric='l1', dim=1, n_points=100, tail='right')

# plt.figure(2)
# sns.distplot(x2, kde=True, rug=False, hist=False, label='x2')
# sns.distplot(x4, kde=True, rug=False, hist=False, label='x4')
# sns.distplot(x3, kde=True, rug=False, hist=False, label='x3: p_value=0.153')
# sns.distplot(x3_pred, kde=True, rug=False, hist=False, label='x3_pred')

all_test = util.test_triplet(sample, util.rank_test, util.kernel_div_test, dim=1, metric='l1', tail='right', size=1000)
