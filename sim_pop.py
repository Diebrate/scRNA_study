import numpy as np

def start_population(mean=0, var=1, size=1000, unif_weight=None):
    population = np.random.normal(mean, var, size)
    if unif_weight is None:
        return population
    elif type(unif_weight) == bool:
        if unif_weight:
            weight = np.repeat(1/size, size)
        else:
            weight = np.random.uniform(size=size)
            weight = weight/weight.sum()
        return population, weight
    else:
        raise TypeError('Invalid type for unif_weight.')

def generate_pop_time_series(start_mean, start_var, size, n_step, drift, sigma, weight=None):
    pop_time_series = np.empty((n_step, size))
    pop_time_series[0] = start_population(start_mean, start_var, size, unif_weight=weight)
    for i in range(1, n_step):
        pop_time_series[i] = pop_time_series[i-1] + drift + np.random.normal(loc=0, scale=sigma, size=size)
    return pop_time_series

def random_impulse(prob, imp_size, imp_var, pop_time_series):
    n_step = pop_time_series.shape[0]
    pop_size = pop_time_series.shape[1]
    for i in range(n_step):
        if np.random.binomial(n=1, p=prob) > 0:
            all_imp = np.repeat(imp_size, pop_size) + np.random.normal(loc=0, scale=imp_var, size=pop_size)
            pop_time_series[i] += all_imp
    return pop_time_series

def fixed_impulse(imp_size, imp_var, pop_time_series, add_time):
    pop_size = pop_time_series.shape[1]
    all_imp = np.repeat(imp_size, pop_size) + np.random.normal(loc=0, scale=imp_var, size=pop_size)
    pop_time_series[add_time] += all_imp
    return pop_time_series

def sample_from_pop(pop_time_series, size):
    T = pop_time_series.shape[0]
    sample = np.empty((T, size))
    for t in range(T):
        sample[t] = np.random.choice(pop_time_series[t], size=size, replace=True)
    return sample
