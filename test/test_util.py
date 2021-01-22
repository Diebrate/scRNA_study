import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import ot
from scipy import spatial
import anndata
import solver


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


def get_weight_no_ignore(x, k):
    px = np.zeros(k)
    for i in x:
        px[int(i)] += 1
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


def diag_test(x1, x2, costm, reg, n_boot=1000):
    N, M = len(x1), len(x2)
    L = np.concatenate((x1, x2)).max() + 1
    rho = np.sqrt(N * M / (N + M))
    p1 = get_weight(x1, L)
    p2 = get_weight(x2, L)
    zs = []
    wass_base = solver.wasserstein_dual_sink(p1, p2, costm, reg)
    u_base, v_base = solver.ot_sinkdiv(p1, p2, costm, reg)['uv_dual']
    for i in range(n_boot):
        p1_temp = get_weight(np.random.choice(range(L), size=N, p=p1), L)
        p2_temp = get_weight(np.random.choice(range(L), size=M, p=p2), L)
        zs.append((rho ** 2) * (solver.wasserstein_dual_sink(p1_temp, p2_temp, costm, reg) -
                  wass_base - np.inner(u_base, p1_temp - p1) - np.inner(v_base, p2_temp - p2)))
    z = (rho ** 2) * wass_base
    return np.mean(np.abs(zs) >= z)


def perm_test1(x1, x2, costm, reg, k=None, n_sim=1000, fullreturn=False):
    N = len(x1)
    if k is None:
        L = np.concatenate((x1, x2)).max() + 1
    else:
        L = k
    p1 = get_weight(x1, L)
    p2 = get_weight(x2, L)
    z = solver.sink_loss(p1, p2, costm, reg)
    zs = []
    data = np.concatenate((x1, x2))
    for i in range(n_sim):
        data_temp = np.random.permutation(data)
        x1_temp = data_temp[:N]
        x2_temp = data_temp[N:]
        p1_temp = get_weight(x1_temp, L)
        p2_temp = get_weight(x2_temp, L)
        zs.append(solver.sink_loss(p1_temp, p2_temp, costm, reg))
    pval = np.mean(zs >= z)
    if pval == 0:
        p_kde = sm.nonparametric.KDEUnivariate(zs)
        p_kde.fit()
        cdf = np.copy(p_kde.cdf)
        cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
        pval = decimal_trunc(1 - cdf[-1], 10)
    if not fullreturn:
        return pval
    else:
        return {'zs': zs, 'pval': pval}
    
    
def perm_test1_unbalanced(x1, x2, costm, reg, reg1, reg2, k=None, n_sim=1000, fullreturn=False):
    N = len(x1)
    if k is None:
        L = np.concatenate((x1, x2)).max() + 1
    else:
        L = k
    p1 = get_weight(x1, L)
    p2 = get_weight(x2, L)
    z = solver.sink_loss(p1, p2, costm, reg, reg1, reg2)
    zs = []
    data = np.concatenate((x1, x2))
    for i in range(n_sim):
        data_temp = np.random.permutation(data)
        x1_temp = data_temp[:N]
        x2_temp = data_temp[N:]
        p1_temp = get_weight(x1_temp, L)
        p2_temp = get_weight(x2_temp, L)
        zs.append(solver.sink_loss_unbalanced(p1_temp, p2_temp, costm, reg, reg1, reg2))
    pval = np.mean(zs >= z)
    if pval == 0:
        p_kde = sm.nonparametric.KDEUnivariate(zs)
        p_kde.fit()
        cdf = np.copy(p_kde.cdf)
        cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
        pval = decimal_trunc(1 - cdf[-1], 10)
    if not fullreturn:
        return pval
    else:
        return {'zs': zs, 'pval': pval}
    
    
def perm_test_balanced_fast(x1, x2, costm, reg, k=None, sink=True, n_sim=1000, fullreturn=False):
    N = len(x1)
    if k is None:
        L = np.concatenate((x1, x2)).max() + 1
    else:
        L = k
    L = int(L)
    p1 = get_weight(x1, L)
    p2 = get_weight(x2, L)
    if sink:
        z = solver.sink_loss_balanced(p1, p2, costm, reg)
    else:
        z = solver.wass_loss_balanced(p1, p2, costm, reg)
    data = np.concatenate((x1, x2))
    d1 = np.zeros((L, n_sim))
    d2 = np.zeros((L, n_sim))
    for i in range(n_sim):
        data_temp = np.random.permutation(data)
        x1_temp = data_temp[:N]
        x2_temp = data_temp[N:]
        d1[:, i] = get_weight(x1_temp, L)
        d2[:, i] = get_weight(x2_temp, L)
    if sink:
        zs = solver.sink_loss_balanced_all(d1, d2, costm, reg)
    else:
        zs = solver.wass_loss_balanced_all(d1, d2, costm, reg)
    pval = np.mean(zs >= z)
    if pval == 0:
        p_kde = sm.nonparametric.KDEUnivariate(zs)
        p_kde.fit()
        cdf = np.copy(p_kde.cdf)
        cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
        pval = decimal_trunc(1 - cdf[-1], 10)
    # if not sink:
    #     pval = np.mean(zs >= z)
    #     if pval == 0:
    #         p_kde = sm.nonparametric.KDEUnivariate(zs)
    #         p_kde.fit()
    #         cdf = np.copy(p_kde.cdf)
    #         cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
    #         pval = decimal_trunc(1 - cdf[-1], 10)
    # else:
    #     pval1 = np.mean(zs <= z)
    #     pval2 = np.mean(zs >= z)
    #     pval = np.min([pval1, pval2])
    #     if pval == 0 or pval == 1:
    #         p_kde = sm.nonparametric.KDEUnivariate(zs)
    #         p_kde.fit()
    #         cdf = np.copy(p_kde.cdf)
    #         cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
    #         pval = 2 * decimal_trunc(cdf[0 if pval1 < pval2 else -1], 10)
    if not fullreturn:
        return pval
    else:
        return {'zs': zs, 'pval': pval}
    
    
def perm_test_unbalanced_fast(x1, x2, costm, reg, reg1, reg2, k=None, sink=True, n_sim=1000, fullreturn=False):
    N = len(x1)
    if k is None:
        L = np.concatenate((x1, x2)).max() + 1
    else:
        L = k
    L = int(L)
    p1 = get_weight(x1, L)
    p2 = get_weight(x2, L)
    if sink:
        z = solver.sink_loss_unbalanced(p1, p2, costm, reg, reg1, reg2)
    else:
        z = solver.wass_loss_unbalanced(p1, p2, costm, reg, reg1, reg2)
    data = np.concatenate((x1, x2))
    d1 = np.zeros((L, n_sim))
    d2 = np.zeros((L, n_sim))
    for i in range(n_sim):
        data_temp = np.random.permutation(data)
        x1_temp = data_temp[:N]
        x2_temp = data_temp[N:]
        d1[:, i] = get_weight(x1_temp, L)
        d2[:, i] = get_weight(x2_temp, L)
    if sink:
        zs = solver.sink_loss_unbalanced_all(d1, d2, costm, reg, reg1, reg2)
    else:
        zs = solver.wass_loss_unbalanced_all(d1, d2, costm, reg, reg1, reg2)
    pval = np.mean(zs >= z)
    if pval == 0:
        p_kde = sm.nonparametric.KDEUnivariate(zs)
        p_kde.fit()
        cdf = np.copy(p_kde.cdf)
        cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
    #     pval = decimal_trunc(1 - cdf[-1], 10)
    # if not sink:
    #     pval = np.mean(zs >= z)
    #     if pval == 0:
    #         p_kde = sm.nonparametric.KDEUnivariate(zs)
    #         p_kde.fit()
    #         cdf = np.copy(p_kde.cdf)
    #         cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
    #         pval = decimal_trunc(1 - cdf[-1], 10)
    # else:
    #     pval1 = np.mean(zs <= z)
    #     pval2 = np.mean(zs >= z)
    #     pval = np.min([pval1, pval2])
    #     if pval == 0 or pval == 1:
    #         p_kde = sm.nonparametric.KDEUnivariate(zs)
    #         p_kde.fit()
    #         cdf = np.copy(p_kde.cdf)
    #         cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
    #         pval = 2 * decimal_trunc(cdf[0 if pval1 < pval2 else -1], 10)
    if not fullreturn:
        return pval
    else:
        return {'zs': zs, 'pval': pval}
    
    
def boot_test_balanced_fast(x1, x2, costm, reg, k=None, n_sim=1000, fullreturn=False):
    N = len(x1)
    M = len(x2)
    if k is None:
        L = np.concatenate((x1, x2)).max() + 1
    else:
        L = k
    L = int(L)
    p1 = get_weight(x1, L)
    p2 = get_weight(x2, L)
    z = solver.wass_loss_balanced(p1, p2, costm, reg)
    d1 = np.zeros((L, n_sim))
    d2 = np.zeros((L, n_sim))
    for i in range(n_sim):
        x1_temp = np.random.choice(x1, size=N)
        x2_temp = np.random.choice(x2, size=M)
        d1[:, i] = get_weight(x1_temp, L)
        d2[:, i] = get_weight(x2_temp, L)
    zs = solver.wass_loss_balanced_all(d1, d2, costm, reg)
    pval = np.min([np.mean(zs >= z), np.mean(zs <= z)])
    if pval == 0 or pval == 1:
        p_kde = sm.nonparametric.KDEUnivariate(zs)
        p_kde.fit()
        cdf = np.copy(p_kde.cdf)
        cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
        pval = decimal_trunc(cdf[0 if pval == 0 else -1], 10)
    if not fullreturn:
        return pval
    else:
        return {'zs': zs, 'pval': pval}
    
    
def boot_test_unbalanced_fast(x1, x2, costm, reg, reg1, reg2, k=None, n_sim=1000, fullreturn=False):
    N = len(x1)
    M = len(x2)
    if k is None:
        L = np.concatenate((x1, x2)).max() + 1
    else:
        L = k
    L = int(L)
    p1 = get_weight(x1, L)
    p2 = get_weight(x2, L)
    z = solver.wass_loss_unbalanced(p1, p2, costm, reg, reg1, reg2)
    d1 = np.zeros((L, n_sim))
    d2 = np.zeros((L, n_sim))
    for i in range(n_sim):
        x1_temp = np.random.choice(x1, size=N)
        x2_temp = np.random.choice(x2, size=M)
        d1[:, i] = get_weight(x1_temp, L)
        d2[:, i] = get_weight(x2_temp, L)
    zs = solver.wass_loss_unbalanced_all(d1, d2, costm, reg, reg1, reg2)
    pval = np.min([np.mean(zs >= z), np.mean(zs <= z)])
    if pval == 0 or pval == 1:
        p_kde = sm.nonparametric.KDEUnivariate(zs)
        p_kde.fit()
        cdf = np.copy(p_kde.cdf)
        cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
        pval = decimal_trunc(cdf[0 if pval == 0 else -1], 10)
    if not fullreturn:
        return pval
    else:
        return {'zs': zs, 'pval': pval}
    
    
def growth_CI1(x1, x2, costm, reg, reg1, reg2, k=None, ignore_empty=True, n_sim=1000, conv=False):
    N = len(x1)
    M = len(x2)
    if k is None:
        L = np.concatenate((x1, x2)).max() + 1
    else:
        L = k
    L = int(L)
    if ignore_empty:
        p1 = get_weight(x1, L)
        p2 = get_weight(x2, L)
    else:
        p1 = get_weight_no_ignore(x1, L)
        p2 = get_weight_no_ignore(x2, L)
        label1 = np.arange(L)[p1 > 0]
        label2 = np.arange(L)[p2 > 0]
        costm = costm[p1 > 0, :]
        costm = costm[:, p2 > 0]
        p1 = p1[p1 > 0]
        p2 = p2[p2 > 0]
    z = solver.estimate_growth1(p1, p2, costm, reg, reg1, reg2, single=True, conv=conv)
    d1 = np.zeros((L, n_sim))
    d2 = np.zeros((L, n_sim))
    for i in range(n_sim):
        x1_temp = np.random.choice(x1, size=N)
        x2_temp = np.random.choice(x2, size=M)
        d1[:, i] = get_weight_no_ignore(x1_temp, L)
        d2[:, i] = get_weight_no_ignore(x2_temp, L)
    d1[d1 == 0] = 0.0001
    d2[d2 == 0] = 0.0001
    if not ignore_empty:
        d1 = d1[label1, :]
        d2 = d2[label2, :]
    zs = solver.estimate_growth1(d1, d2, costm, reg, reg1, reg2, single=False, conv=conv)
    zs = np.array(zs)
    if ignore_empty:
        return {'est': z, 
                '2.5%': np.percentile(zs, 2.5, axis=0), 
                '97.5%': np.percentile(zs, 97.5, axis=0),
                'sim': zs}
    else:
        return {'est': z, 
                '2.5%': np.percentile(zs, 2.5, axis=0), 
                '97.5%': np.percentile(zs, 97.5, axis=0),
                'sim': zs,
                'valid_group1': label1,
                'valid_group2': label2}


def growth_CI2(x1, x2, costm, reg, reg1, reg2, k=None, ignore_empty=True, n_sim=1000, conv=False):
    N = len(x1)
    M = len(x2)
    if k is None:
        L = np.concatenate((x1, x2)).max() + 1
    else:
        L = k
    L = int(L)
    if ignore_empty:
        p1 = get_weight(x1, L)
        p2 = get_weight(x2, L)
    else:
        p1 = get_weight_no_ignore(x1, L)
        p2 = get_weight_no_ignore(x2, L)
        label1 = np.arange(L)[p1 > 0]
        label2 = np.arange(L)[p2 > 0]
        costm = costm[p1 > 0, :]
        costm = costm[:, p2 > 0]
        p1 = p1[p1 > 0]
        p2 = p2[p2 > 0]
    z = solver.estimate_growth2(p1, p2, costm, reg, reg1, reg2, single=True, conv=conv)
    d1 = np.zeros((L, n_sim))
    d2 = np.zeros((L, n_sim))
    for i in range(n_sim):
        x1_temp = np.random.choice(x1, size=N)
        x2_temp = np.random.choice(x2, size=M)
        d1[:, i] = get_weight_no_ignore(x1_temp, L)
        d2[:, i] = get_weight_no_ignore(x2_temp, L)
    d1[d1 == 0] = 0.0001
    d2[d2 == 0] = 0.0001
    if not ignore_empty:
        d1 = d1[label1, :]
        d2 = d2[label2, :]
    zs = solver.estimate_growth2(d1, d2, costm, reg, reg1, reg2, single=False, conv=conv)
    zs = np.array(zs)
    if ignore_empty:
        return {'est': z, 
                '2.5%': np.percentile(zs, 2.5, axis=0), 
                '97.5%': np.percentile(zs, 97.5, axis=0),
                'sim': zs}
    else:
        return {'est': z, 
                '2.5%': np.percentile(zs, 2.5, axis=0), 
                '97.5%': np.percentile(zs, 97.5, axis=0),
                'sim': zs,
                'valid_group1': label1,
                'valid_group2': label2}



def boot_test(x1, x2, costm, reg, n_boot=1000):
    N, M = len(x1), len(x2)
    L = np.concatenate((x1, x2)).max() + 1
    p1 = get_weight(x1, L)
    p2 = get_weight(x2, L)
    z = solver.sink_loss(p1, p2, costm, reg)
    zs = []
    for i in range(n_boot):
        p1_temp = get_weight(np.random.choice(x1, size=N), L)
        p2_temp = get_weight(np.random.choice(x2, size=M), L)
        zs.append(solver.sink_loss(p1_temp, p2_temp, costm, reg))
    return np.mean(zs >= z)


def boot_test_multi(x1, x2, costm, reg, n_boot=1000, n_seed=100):
    p = []
    for i in range(n_seed):
        p.append(boot_test(x1, x2, costm, reg))
    return p


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


# test data
##################################################
# import seaborn as sns

# n = 100
# m = 100
# n_null = 100
# n_sim = 1000

# k = 10
# r = np.repeat(1, k)
# c = r.copy()
# r, c = r / sum(r), c / sum(c)
# a = np.zeros((n_null, n))
# b = np.zeros((n_null, m))

# px = np.repeat(1, k)
# py = np.arange(1, k + 1)
# px, py = px / sum(px), py / sum(py)
# x = np.zeros((n_null, n))
# y = np.zeros((n_null, m))

# for i in range(n_null):
#     a[i] = np.random.choice(range(k), size=n, p=r)
#     b[i] = np.random.choice(range(k), size=m, p=c)
#     x[i] = np.random.choice(range(k), size=n, p=px)
#     y[i] = np.random.choice(range(k), size=m, p=py)

# a = np.array(a, dtype=int)
# b = np.array(b, dtype=int)
# x = np.array(x, dtype=int)
# y = np.array(y, dtype=int)

# costm = np.random.rand(k, k) 
# costm = costm @ costm.transpose()
# np.fill_diagonal(costm, 0)
# reg = 0.5
# reg1 = 1
# reg2 = 50

# index = 88

# balanced = False
# sink = False
# fast = True

# d1 = np.zeros((k, n_null))
# d2 = np.zeros((k, n_null))
# for i in range(n_null):
#     d1[:, i] = get_weight(x[i], k)
#     d2[:, i] = get_weight(y[i], k)

# zs_null = solver.estimate_growth1(d1, d2, costm, reg, reg1, reg2, single=False)
# zs = growth_CI1(x[index], y[index], costm, reg, reg1, reg2, n_sim=10000)['sim']

# k_index = 7

# df = pd.DataFrame()
# df['value'] = np.concatenate((zs_null[:, k_index], zs[:, k_index]))
# df['source'] = np.concatenate((np.repeat('null', zs_null.shape[0]),
#                                np.repeat('boot', zs.shape[0])))
# sns.kdeplot(data=df, x='value', hue='source', common_norm=False)
##################################################

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
