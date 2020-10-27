import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import ot
from scipy import spatial


def decimal_trunc(x, n_dec):
    x_temp = x * (10 ** n_dec)
    x_temp = np.trunc(x_temp)
    return x_temp * (10 ** -n_dec)


def get_weight(x, k):
    px = np.zeros(k)
    for i in x:
        px[int(i)] += 1
    for l in range(k):
        if px[l] == 0:
            px[l] += 0.0001
    return px / np.sum(px)


def get_entropy(p, k):
    e = 0
    for i in range(k):
        e += p[i] * np.log(p[i])
    return e


def get_cost_matrix(x, y, dim, method='euclidean'):
    if dim == 1:
        x_reshape, y_reshape = np.reshape(x, (-1, dim)), np.reshape(y, (-1, dim))
        c_matrix = spatial.distance.cdist(x_reshape, y_reshape, metric=method)
    else:
        c_matrix = spatial.distance.cdist(x, y, metric=method)
    return c_matrix


def get_offdiag(M):
    M = np.array(M)
    k = np.array(M).shape[0]
    mask = np.diag(np.ones(k, dtype=bool)) == False
    return M[mask]  


def ot_map_test(x, y, M, k, reg=1):
    px, py = get_weight(x, k), get_weight(y, k)
    ot_map = ot.sinkhorn(px, py, M, reg=reg)
    return (ot_map**2).sum() - (ot_map[range(k),range(k)]**2).sum()


def ot_map_test1(x, y, M, k, reg=1):
    px, py = get_weight(x, k), get_weight(y, k)
    ot_map = ot.sinkhorn(px, py, M, reg=reg)
    ot_map = np.diag(1 / px) @ ot_map
    return (ot_map**2).sum() - (ot_map[range(k),range(k)]**2).sum()


def ot_map_test2(x, y, M, k, reg=1):
    px, py = get_weight(x, k), get_weight(y, k)
    ot_map = ot.sinkhorn(px, py, M, reg=reg)
    mask = np.diag(np.ones(k, dtype=bool)) == False
    off_diag = ot_map[mask]
    return off_diag.max() * off_diag.min()


def ot_map_test3(x, y, M, k, reg=1):
    px, py = get_weight(x, k), get_weight(y, k)
    ot_map = ot.sinkhorn(px, py, M, reg=reg)
    mask = np.diag(np.ones(k, dtype=bool)) == False
    off_diag = ot_map[mask]
    return off_diag.max() * (off_diag.sum() + off_diag.min())


def ot_map_test4(x, y, M, k, reg=1):
    px, py = get_weight(x, k), get_weight(y, k)
    ot_map = ot.sinkhorn(px, py, M, reg=reg)
    mask = np.diag(np.ones(k, dtype=bool)) == False
    off_diag = ot_map[mask]
    return (off_diag.max() + off_diag.min()) ** 2


def perm_test(test_func, x, y, tail, n_times=2000, timer=True, cluster=False, **kwargs):
    nx = len(x)
    data = np.append(x, y)
    k = data.max() + 1
    test_statistics = test_func(x, y, **kwargs)
    reference = np.empty(n_times)
    if cluster:
        test_statistics /= get_entropy(get_weight(data, k), k)
        # weight = get_weight(data, k)
        # label = np.random.choice(np.arange(k), k, p=weight)
        label = np.random.permutation(np.arange(k))
        for i in range(n_times):
            data_temp = np.random.permutation(np.array([label[d] for d in data]))
            # data_temp = np.random.permutation(data)
            x_temp = data_temp[:nx]
            y_temp = data_temp[nx:]
            reference[i] = test_func(x_temp, y_temp, **kwargs)
            reference[i] /= get_entropy(get_weight(data_temp, k), k)
            if timer and (i + 1) % 100 == 0:
                print('Currently at step ' + str(i + 1))
    else:
        for i in range(n_times):
            data_temp = np.random.permutation(data)
            x_temp = data_temp[:nx]
            y_temp = data_temp[nx:]
            reference[i] = test_func(x_temp, y_temp, **kwargs)
            if timer and (i + 1) % 100 == 0:
                print('Currently at step ' + str(i + 1))
    percentile = scipy.stats.percentileofscore(reference, test_statistics)
    if percentile == 100 or percentile == 0:
        p_kde = sm.nonparametric.KDEUnivariate(reference)
        p_kde.fit()
        cdf = np.copy(p_kde.cdf)
        cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
        percentile = decimal_trunc(cdf[0 if percentile == 0 else -1], 10) * 100
    if tail == 'left':
        p_value = percentile / 100
    elif tail == 'right':
        p_value = (100 - percentile) / 100
    elif tail == 'both':
        p_value = min(percentile, 100 - percentile) / 100
    else:
        raise ValueError('Invalid tail option. Try \'left\', \'right\' or \'both\'')
    return {'test_statistics': test_statistics,
            'p_value': p_value,
            'reference': reference}


def simulate(k, p_start, p_trans, size=1000):
    x = np.random.choice(range(k), size=size, p=p_start)
    y = np.empty(size)
    for i in range(size):
        y[i] = np.random.choice(range(k), p=p_trans[x[i]])
    return x, y


def simulate2d(p_start, p_trans, mean, var, size=1000):
    '''return simulated 2D data with noise
    
    
    centroids = [[1, 1], [1, -1], [-1, 1], [-1, -1]]
    
    
    Parameters:
        p_start: starting probability
        p_trans: transitional probability matrix
        var: noise level (covariance matrix)
        
        
    '''
    def get_label(x, centroids):
        dist = spatial.distance.cdist(x, centroids)
        return np.argmin(dist, axis=1)
    k = 4
    lx = np.random.choice(range(k), size=size, p=p_start)
    x = np.empty((0, 2))
    for l in lx:
        x = np.vstack((x, np.random.multivariate_normal(mean[l], var)))
    lx_noise = get_label(x, centroids=mean)
    ly = np.empty(size, dtype=int)
    for i in range(size):
        ly[i] = np.random.choice(range(k), p=p_trans[lx_noise[i]])
    y = np.empty((0, 2))
    for l in ly:
        y = np.vstack((y, np.random.multivariate_normal(mean[l], var)))
    ly_noise = get_label(y, centroids=mean)
    return lx_noise, ly_noise
    


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
