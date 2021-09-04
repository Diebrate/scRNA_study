import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import ot
from scipy import spatial
from sklearn.cluster import KMeans
import anndata
import solver
import matplotlib.pyplot as plt
import seaborn as sns

import peak_detection

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


def get_count(x, k):
    cx = np.zeros(k)
    for i in x:
        cx[int(i)] += 1
    return cx


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
    if not fullreturn:
        return pval
    else:
        return {'zs': zs, 'pval': pval}
    
    
def perm_test_unbalanced_fast(x1, x2, costm, reg, reg1, reg2, k=None, sink=True, n_sim=1000, fullreturn=False, n_conv=None):
    N = len(x1)
    if k is None:
        L = np.concatenate((x1, x2)).max() + 1
    else:
        L = k
    L = int(L)
    p1 = get_weight(x1, L)
    p2 = get_weight(x2, L)
    if n_conv is None:
        # if sink:
        #     z = solver.sink_loss_unbalanced(p1, p2, costm, reg, reg1, reg2)
        # else:
        #     z = solver.wass_loss_unbalanced(p1, p2, costm, reg, reg1, reg2)
        z = solver.loss_unbalanced(p1, p2, costm, reg, reg1, reg2, sink=sink, single=True)
        data = np.concatenate((x1, x2))
        d1 = np.zeros((L, n_sim))
        d2 = np.zeros((L, n_sim))
        for i in range(n_sim):
            data_temp = np.random.permutation(data)
            x1_temp = data_temp[:N]
            x2_temp = data_temp[N:]
            d1[:, i] = get_weight(x1_temp, L)
            d2[:, i] = get_weight(x2_temp, L)
    else:
        p1_temp = p1.copy()
        for j in range(n_conv):
            tmap = solver.ot_unbalanced(p1_temp, p2, costm, reg, reg1, reg2)
            p1_temp = tmap.sum(axis=1)
        if sink:
            z = solver.sink_loss_unbalanced(p1_temp, p2, costm, reg, reg1, reg2)
        else:
            z = solver.wass_loss_unbalanced(p1_temp, p2, costm, reg, reg1, reg2)
        data = np.concatenate((x1, x2))
        d1 = np.zeros((L, n_sim))
        d2 = np.zeros((L, n_sim))
        for i in range(n_sim):
            data = np.random.permutation(data)
            x1_temp = data_temp[:N]
            x2_temp = data_temp[N:]
            d1[:, i] = get_weight(x1_temp, L)
            d2[:, i] = get_weight(x2_temp, L)
        d1_temp = d1.copy()
        for j in range(n_conv):
            tmap_all = solver.ot_unbalanced_all(d1_temp, d2, costm, reg, reg1, reg2)
            for it in range(n_sim):
                d1_temp[:, it] = tmap_all[it].sum(axis=1)
        d1 = d1_temp.copy()
    # if sink:
    #     zs = solver.sink_loss_unbalanced_all(d1, d2, costm, reg, reg1, reg2)
    # else:
    #     zs = solver.wass_loss_unbalanced_all(d1, d2, costm, reg, reg1, reg2)
    zs = solver.loss_unbalanced(d1, d2, costm, reg, reg1, reg2, sink=sink, single=False)
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
    
    
def boot_test_balanced_fast(x1, x2, costm, reg, k=None, sink=True, n_sim=1000, fullreturn=False):
    N = len(x1)
    M = len(x2)
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
    d1 = np.zeros((L, n_sim))
    d2 = np.zeros((L, n_sim))
    for i in range(n_sim):
        x1_temp = np.random.choice(x1, size=N)
        x2_temp = np.random.choice(x2, size=M)
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
    # pval = np.min([np.mean(zs >= z), np.mean(zs <= z)])
    # if pval == 0 or pval == 1:
    #     p_kde = sm.nonparametric.KDEUnivariate(zs)
    #     p_kde.fit()
    #     cdf = np.copy(p_kde.cdf)
    #     cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
    #     pval = decimal_trunc(cdf[0 if pval == 0 else -1], 10)
    if not fullreturn:
        return pval
    else:
        return {'zs': zs, 'pval': pval}
    
    
def boot_test_unbalanced_fast(x1, x2, costm, reg, reg1, reg2, k=None, sink=True, n_sim=1000, fullreturn=False, n_conv=None, scale=5):
    N = len(x1)
    M = len(x2)
    if k is None:
        L = np.concatenate((x1, x2)).max() + 1
    else:
        L = k
    L = int(L)
    p1 = get_weight(x1, L)
    p2 = get_weight(x2, L)
    d1 = np.zeros((L, n_sim))
    d2 = np.zeros((L, n_sim))
    if n_conv is None:
        z = solver.loss_unbalanced(p1, p2, costm, reg, reg1, reg2, sink=sink, single=True)
        for i in range(n_sim):
            x1_temp = np.random.choice(x1, size=N)
            x2_temp = np.random.choice(x2, size=M)
            d1[:, i] = get_weight(x1_temp, L)
            d2[:, i] = get_weight(x2_temp, L)
        zs = solver.loss_unbalanced(d1, d2, costm, reg, reg1, reg2, sink=sink, single=False)
    else:
        p1_temp = p1.copy()
        for j in range(n_conv):
            tmap = solver.ot_unbalanced(p1_temp, p2, costm, reg, reg1, 50)
            p1_temp = tmap.sum(axis=1)
        z = solver.loss_unbalanced(p1, p2, costm, reg, reg1, reg2, sink=sink, single=True)
        # z += solver.loss_balanced(p1_temp, p2, costm, reg, sink=sink, single=True)
        # data = np.array([])
        # for q in range(L):
        #     data = np.concatenate((data, np.repeat(q, int(p1_temp[q] * N))))
        # n_sub = len(data)
        # data = np.concatenate((data, x2))
        for i in range(n_sim):
            # data_temp = np.random.permutation(data)
            # x1_temp = data_temp[:n_sub]
            # x2_temp = data_temp[n_sub:]
            x1_temp = np.random.choice(np.arange(L), size=N, p=p1_temp)
            x2_temp = np.random.choice(np.arange(L), size=M, p=p2)
            d1[:, i] = get_weight(x1_temp, L)
            d2[:, i] = get_weight(x2_temp, L)
        # d1_temp = d1.copy()
        # for j in range(n_conv):
        #     tmap_all = solver.ot_unbalanced_all(d1_temp, d2, costm, reg, reg1, reg2)
        #     for it in range(n_sim):
        #         d1_temp[:, it] = tmap_all[it].sum(axis=1)
        zs = solver.loss_unbalanced(d1, d2, costm, reg, reg1, reg2, sink=sink, single=False)
    pval = np.mean(zs >= z)
    if pval == 0:
        p_kde = sm.nonparametric.KDEUnivariate(zs)
        p_kde.fit()
        cdf = np.copy(p_kde.cdf)
        cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
        pval = decimal_trunc(1 - cdf[-1], 10)  
    # pval = np.min([np.mean(zs >= z), np.mean(zs <= z)])
    # if pval == 0 or pval == 1:
    #     p_kde = sm.nonparametric.KDEUnivariate(zs)
    #     p_kde.fit()
    #     cdf = np.copy(p_kde.cdf)
    #     cdf = cdf[np.all(np.vstack((cdf!=0, cdf!=1)), axis=0)]
    #     pval = decimal_trunc(cdf[0 if pval == 0 else -1], 10)
    if not fullreturn:
        return pval
    else:
        return {'zs': zs, 'pval': pval}
    
    
def power_test_unbalanced_fast(p, g, B, costm, reg, reg1, reg2, k, sink=True, n_sim=1000, n_sample=1000):
    p1 = p
    p2 = p1 * g / np.sum(p1 * g)
    p2_alt = B.T @ p2
    d1 = np.zeros((k, n_sim))
    d2 = np.zeros((k, n_sim))
    d2_alt = np.zeros((k, n_sim))
    for i in range(n_sim):
        x1_temp = np.random.choice(np.arange(k), size=n_sample, p=p2)
        x2_temp = np.random.choice(np.arange(k), size=n_sample, p=p2)
        x2_alt_temp = np.random.choice(np.arange(k), size=n_sample, p=p2_alt)
        d1[:, i] = get_weight(x1_temp, k)
        d2[:, i] = get_weight(x2_temp, k)
        d2_alt[:, i] = get_weight(x2_alt_temp, k)
    null = solver.loss_unbalanced(d1, d2, costm, reg, reg1, reg2, sink=sink, single=False)
    alt = solver.loss_unbalanced(d1, d2_alt, costm, reg, reg1, reg2, sink=sink, single=False)
    res = np.array([np.mean(null >= j) for j in alt])
    return {'null': null, 'alt': alt, 'power': np.mean(res <= 0.05)}


def est_cost(x1, x2, costm, reg, reg1, reg2, k, n_conv, sink=True, n_sim=1000):
    N = len(x1)
    M = len(x2)
    p1 = get_weight(x1, k)
    p2 = get_weight(x2, k)
    p1_g = p1.copy()
    for j in range(n_conv):
        tmap = solver.ot_unbalanced(p1_g, p2, costm, reg, reg1, 50)
        p1_g = tmap.sum(axis=1)
    res = {'cost_g': None, 'cost_c': None, 'cost_all': None}
    d1 = np.zeros((k, n_sim))
    d1_g = np.zeros((k, n_sim))
    d2 = np.zeros((k, n_sim))
    for i in range(n_sim):
        x1_temp = np.random.choice(np.arange(k), size=N, p=p1)
        x1_g_temp = np.random.choice(np.arange(k), size=N, p=p1_g)
        x2_temp = np.random.choice(np.arange(k), size=M, p=p2)
        d1[:, i] = get_weight(x1_temp, k)
        d1_g[:, i] = get_weight(x1_g_temp, k)
        d2[:, i] = get_weight(x2_temp, k)
    res['cost_g'] = solver.loss_unbalanced(d1, d1_g, costm, reg, reg1, reg2, sink=sink, single=False)
    res['cost_c'] = solver.loss_unbalanced(d1_g, d2, costm, reg, reg2, reg2, sink=sink, single=False)
    res['cost_all'] = solver.loss_unbalanced(d1, d2, costm, reg, reg1, reg2, sink=sink, single=False)
    res['cost_add'] = res['cost_g'] + res['cost_c']
    return res


def real_cost(p1, g, B, costm, reg, reg1, reg2, k, n1, n2, sink=True, n_sim=1000):
    p1_g = p1 * g
    p1_g = p1_g / np.sum(p1_g)
    p2 = B.T @ p1_g
    res = {'cost_g': None, 'cost_c': None, 'cost_all': None}
    d1 = np.zeros((k, n_sim))
    d1_g = np.zeros((k, n_sim))
    d2 = np.zeros((k, n_sim))
    for i in range(n_sim):
        x1_temp = np.random.choice(np.arange(k), size=n1, p=p1)
        x1_g_temp = np.random.choice(np.arange(k), size=n1, p=p1_g)
        x2_temp = np.random.choice(np.arange(k), size=n2, p=p2)
        d1[:, i] = get_weight(x1_temp, k)
        d1_g[:, i] = get_weight(x1_g_temp, k)
        d2[:, i] = get_weight(x2_temp, k)
    res['cost_g'] = solver.loss_unbalanced(d1, d1_g, costm, reg, reg1, reg2, sink=sink, single=False)
    res['cost_c'] = solver.loss_unbalanced(d1_g, d2, costm, reg, reg2, reg2, sink=sink, single=False)
    res['cost_all'] = solver.loss_unbalanced(d1, d2, costm, reg, reg1, reg2, sink=sink, single=False)
    res['cost_add'] = res['cost_g'] + res['cost_c']
    return res


def est_check_unbalanced_fast(x1, x2, p1, g, B, costm, reg, reg1, reg2, k, n_conv, sink=True, n_sim=1000, n_sample=1000):
    N = len(x1)
    M = len(x2)
    est_res = est_cost(x1, x2, costm, reg, reg1, reg2, k, n_conv, sink=sink)
    real_res = real_cost(p1, g, B, costm, reg, reg1, reg2, k, n1=N, n2=M, sink=sink)
    names = list(est_res.keys())
    for name in names:
        df = {'value': np.concatenate((est_res[name], real_res[name])),
              'source': np.concatenate((np.repeat('est', n_sim), np.repeat('real', n_sample)))} 
        plt.figure()
        sns.kdeplot(data=df, x='value', hue='source', common_norm=False, alpha=.5, fill=True).set_title(name)
    def dict_concat(d):
        a = np.array([])
        for key, item in d.items():
            a = np.concatenate((a, item))
        return a
    df = {'value': dict_concat(est_res),
          'source': np.repeat(names, n_sim)}
    plt.figure()
    sns.kdeplot(data=df, x='value', hue='source', common_norm=False, alpha=.5, fill=True).set_title('sim')
    df = {'value': dict_concat(real_res),
          'source': np.repeat(names, n_sim)}
    plt.figure()
    sns.kdeplot(data=df, x='value', hue='source', common_norm=False, alpha=.5, fill=True).set_title('real')
    return est_res, real_res


def offset_growth(prob_all, costm, reg, reg1, conv=True):
    prob_bef = prob_all[:-1, ].T
    prob_aft = prob_all[1:, ].T
    gs = 1 + solver.estimate_growth1(prob_bef, prob_aft, costm, reg, reg1, 50, single=False, conv=conv)    
    def backward(g, p):
        p0 = np.array(p / g)
        return p0 / np.sum(p0)
    prob_offset = np.zeros(prob_all.shape)
    prob_offset[0, ] = prob_all[0, ]
    T = prob_all.shape[0]
    for t in range(1, T):
        t_temp = t
        p_temp = prob_all[t, ]
        while t_temp > 0:
            p_temp = backward(gs[t - 1, ], p_temp)
            t_temp -= 1
        prob_offset[t, ] = p_temp
    return prob_offset


def offset_growth_mc(prob_all_mc, costm, reg, reg1, conv=True):
    M = len(prob_all_mc)
    res = []
    for m in range(M):
        res.append(offset_growth(prob_all_mc[m], costm, reg, reg1, conv=conv))
    return np.array(res)
            
    
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


def cp_detection(data, k, costm, reg=1, reg1=1, reg2=1, balanced=True, sink=True, order=None, perm=True, n_conv=None):
    T = data.shape[0]
    ps = []
    for t in range(T - 1):
        if order is None:
            data1 = data[t, :]
            data2 = data[t + 1, :]
        else:
            data1 = data[max(0, t - order):min(T, t + order + 1), :].flatten()
            data2 = data[max(0, t + 1 - order):min(T, t + order + 2), :].flatten()
        if balanced:
            if perm:
                ps.append(perm_test_balanced_fast(data1, data2, costm, reg, sink=sink, k=k))
            else:
                ps.append(boot_test_balanced_fast(data1, data2, costm, reg, sink=sink, k=k))
        else:
            if perm:
                ps.append(perm_test_unbalanced_fast(data1, data2, costm, reg, reg1, reg2, sink=sink, k=k))
            else:
                ps.append(boot_test_unbalanced_fast(data1, data2, costm, reg, reg1, reg2, sink=sink, k=k, n_conv=n_conv))
    ps = np.array(ps)
    return {'ps': ps, 'res': np.arange(len(ps))[ps <= 0.05]}


def run_cusum_ot(data, k, costm, reg=1, reg1=1, reg2=1, balanced=True, sink=True, n_conv=None, full_return=False):
    T = data.shape[0]
    cost = []
    for t in range(T - 1):
        data1 = data[t, :]
        data2 = data[t + 1, :]
        p1 = get_weight(data1, k)
        p2 = get_weight(data2, k)
        if balanced:
            cost.append(solver.loss_balanced(p1, p2, costm, reg, sink=sink))
        else:
            if n_conv is not None:
                p1_g = p1.copy()
                for i in range(n_conv):
                    tmap = solver.ot_unbalanced(p1_g, p2, costm, reg, reg1, 10)
                    p1_g = np.sum(tmap, axis=1)
                    cost.append(solver.loss_unbalanced(p1, p1_g, costm, reg, reg1, reg2, sink=sink))
            else:
                cost.append(solver.loss_unbalanced(p1, p2, costm, reg, reg1, reg2, sink=sink))
            # tmap = solver.ot_unbalanced(p1_g, p2, costm, reg, reg1, reg2)
            # cost.append(np.sum(tmap * costm))
    res = np.arange(T - 1)[cost > np.mean(cost)]
    res = np.arange(T - 1)[cost > np.mean(cost) + 2 * np.std(cost)]
    res = res[res != 0]
    if full_return:
        return res, cost
    else:
        return res
    
    
def run_cusum(data, k, return_ll=False):
    def get_count(x, k):
        count = np.zeros(k)
        for i in x:
            count[int(i)] += 1
        return count
    T = data.shape[0]
    prob = np.zeros((T, k))
    count = np.zeros((T, k))
    n = np.zeros(T)
    st = np.zeros(T - 1)
    for t in range(T):
        prob[t, :] = get_weight(data[t, :], k)
        count[t, :] = get_count(data[t, :], k)
        n[t] = len(data[t, :])
    for t in range(T - 1):
        l1 = np.sum(count[t + 1, :] * np.log(prob[t + 1, :]))
        l0 = np.sum(count[t + 1, :] * np.log(prob[t, :]))
        st[t] = l1 - l0
    st = np.cumsum(st)
    gt = np.array([st[i] - np.min(st[:i + 1]) for i in range(len(st))])
    diff = np.concatenate(([0], np.diff(gt)))
    cp = []
    diff = np.abs(np.diff(gt))
    ref = diff.mean()
    for i in range(len(diff) - 1):
        if diff[i] > ref and diff[i + 1] > ref:
            cp.append(i + 1)
    cp = np.array(cp)
    cp = np.arange(1, T - 1)[diff > np.mean(diff) + 2 * np.std(diff)]
    if return_ll:
        return cp, st, gt, diff
    else:
        return cp    
    
    
def cp_detection_ot(data, k, costm, reg=1, reg1=1, reg2=1, return_cost=True, **kwargs):
    cost = run_cusum_ot(data, k, costm, reg, reg1, reg2, full_return=True, **kwargs)[1]
    # sort_ind = np.argsort(cost)[::-1]
    # cp = []
    # i = 0
    # while i < len(sort_ind):
    #     x1 = data[sort_ind[i]]
    #     x2 = data[sort_ind[i] + 1]
    #     loglik_null, loglik_alt, loglik_sim = ot_loglik_unbalanced(x1, x2, k, costm, reg, reg1, reg2)
    #     l_up = np.percentile(loglik_sim, 97.5)
    #     l_low = np.percentile(loglik_sim, 2.5)
    #     if loglik_alt <= l_low or loglik_alt >= l_up:
    #         cp.append(sort_ind[i])
    #         i += 1
    #     else:
    #         i = len(sort_ind)
    cp = []
    cost_diff = np.abs(np.diff(cost))
    ref = cost_diff.mean()
    for i in range(len(cost_diff) - 1):
        if cost_diff[i] > ref and cost_diff[i + 1] > ref:
            cp.append(i + 1)
    cp = np.array(cp)
    cp = np.arange(data.shape[0] - 1)[cost > np.mean(cost)]
    cp = np.arange(data.shape[0] - 1)[cost > np.mean(cost) + np.std(cost)]
    cp = cp[cp != 0]
    if return_cost:
        return cp, cost
    else:
        return cp
    
    
def ot_loglik_unbalanced(x1, x2, k, costm, reg, reg1, reg2, n_conv=7):
    p1 = get_weight(x1, k)
    p2 = get_weight(x2, k)
    p1_g = p1.copy()
    for i in range(n_conv):
        tmap = solver.ot_unbalanced(p1, p2, costm, reg, reg1, reg2)
        p1_g = np.sum(tmap, axis=1)
    loglik_null = 0
    loglik_alt = 0
    loglik_sim = np.repeat(0, 1000)
    data = get_count(x2, k)
    p2_sim = np.zeros((k, 1000))
    for j in range(1000):
        x1_temp = np.random.choice(np.arange(k), size=len(x2), p=p1_g)
        p2_sim[:, j] = get_weight_no_ignore(x1_temp, k)
    for i in range(k):
        loglik_null += data[i] * np.log(p1_g[i])
        loglik_alt += data[i] * np.log(p2[i])
        loglik_sim = loglik_sim + data[i] * np.log(p2_sim[i])
    return loglik_null, loglik_alt, loglik_sim
        

def cp_detection_mc(data, k, costm, reg=1, reg1=1, reg2=1, balanced=True, sink=True, n_sim=1000, track_iter=True, order=None):
    # change point detection designed for monte carlo data
    M, T, n = data.shape
    ps = np.zeros((M, T - 1))
    for m in range(M):
        p1 = np.zeros((k, n_sim * (T - 1)))
        p2 = np.zeros((k, n_sim * (T - 1)))
        p1_real = np.zeros((k, T - 1))
        p2_real = np.zeros((k, T - 1))
        if order is None:
            d1 = data[m, :-1, :]
            d2 = data[m, 1:, :]
            for t in range(T - 1):
                data_all = np.concatenate((d1[t, :], d2[t, :]))
                p1_real[:, t] = get_weight(d1[t, :], k)
                p2_real[:, t] = get_weight(d2[t, :], k)
                for j in range(n_sim):
                    data_temp = np.random.permutation(data_all)
                    d1_temp = data_temp[:n]
                    d2_temp = data_temp[n:]
                    p1[:, t * n_sim + j] = get_weight(d1_temp, k)
                    p2[:, t * n_sim + j] = get_weight(d2_temp, k)
        else:
            for t in range(T - 1):
                data_m = data[m]
                d1_order = data_m[max(0, t - order):max(T, t + order + 1), :].flatten()
                d2_order = data_m[max(0, t + 1 - order):max(T, t+ order + 2), :].flatten()
                data_all = np.concatenate((d1_order[t, :], d2_order[t, :]))
                p1_real[:, t] = get_weight(d1_order, k)
                p2_real[:, t] = get_weight(d2_order, k)
                for j in range(n_sim):
                    data_temp = np.random.permutation(data_all)
                    d1_temp = data_temp[:n]
                    d2_temp = data_temp[n:]
                    p1[:, t * n_sim + j] = get_weight(d1_temp, k)
                    p2[:, t * n_sim + j] = get_weight(d2_temp, k)
        if sink:
            if balanced:
                zs = solver.sink_loss_balanced_all(p1, p2, costm, reg)
                z = solver.sink_loss_balanced_all(p1_real, p2_real, costm, reg)
            else:
                zs = solver.sink_loss_unbalanced_all(p1, p2, costm, reg, reg1, reg2)
                z = solver.sink_loss_unbalanced_all(p1_real, p2_real, costm, reg, reg1, reg2)
        else:
            if balanced:
                zs = solver.wass_loss_balanced_all(p1, p2, costm, reg)
                z = solver.wass_loss_balanced_all(p1_real, p2_real, costm, reg)
            else:
                zs = solver.wass_loss_unbalanced_all(p1, p2, costm, reg, reg1, reg2)
                z = solver.wass_loss_unbalanced_all(p1_real, p2_real, costm, reg, reg1, reg2)
        for t in range(T - 1):
            ps[m, t] = np.mean(zs[t * n_sim : (t + 1) * n_sim] >= z[t])
        if track_iter:
            print('finished iteration ' + str(m))
    return ps


def get_cp_from_cost(cost, win_size=None):
    l = len(cost)
    if win_size is not None:
        trimmed_cost = cost[(win_size-1):(l - win_size)]
        res = peak_detection.peaks_detection(trimmed_cost, np.arange(1, l + 1), min_snr=0.001)[0]
        # res = peak_detection.peaks_detection(cost, np.arange(1, l + 1), min_snr=0.001)[0]
        res = np.array(res)
        # res = res[res >= win_size - 1]
        # return res[res < l - win_size]
        return res + win_size - 1 
    else:
        est = KMeans(n_clusters=2).fit_predict(cost.reshape((-1, 1)))
        ind = est[np.argmax(cost)]
        return np.arange(l)[est == ind]


def multisetting_cp_ot_cost(cost_all, T, win_size=None):
    n_ns, n_nus, n_etas = cost_all.shape
    M = cost_all[0, 0, 0].shape[0]
    res = np.empty(cost_all.shape, dtype=object)
    for i in range(n_ns):
        for j in range(n_nus):
            for k in range(n_etas):
                res[i, j, k] = []
                cost = cost_all[i, j, k]
                for m in range(M):
                    res[i, j, k].append(get_cp_from_cost(cost[m], win_size=win_size))
                res[i, j, k] = np.array(res[i, j, k], dtype=object)
    return res


def multisetting_cp_ot_cost_ng(cost_all, T, win_size=None):
    n_ns, n_pwrs = cost_all.shape
    M = cost_all[0, 0].shape[0]
    res = np.empty(cost_all.shape, dtype=object)
    for i in range(n_ns):
        for j in range(n_pwrs):
            res[i, j] = []
            cost = cost_all[i, j]
            for m in range(M):
                res[i, j].append(get_cp_from_cost(cost[m], win_size=win_size))
            res[i, j] = np.array(res[i, j], dtype=object)
    return res


def multisetting_cp_ot(data_all, d, costm, reg, reg1, reg2, sink=True, balanced=False, n_conv=None):
    n_ns, n_nus, n_etas = data_all.shape
    M = data_all[0, 0, 0].shape[0]
    res = np.empty(data_all.shape, dtype=object)
    for i in range(n_ns):
        for j in range(n_nus):
            for k in range(n_etas):
                res[i, j, k] = []
                data = data_all[i, j, k]
                for m in range(M):
                    res[i, j, k].append(cp_detection_ot(data[m], d, costm, reg, reg1, reg2, balanced=balanced, sink=sink, n_conv=n_conv, return_cost=False))
                res[i, j, k] = np.array(res[i, j, k], dtype=object)
    return res


def multisetting_cp_cs(data_all, d):
    n_ns, n_nus, n_etas = data_all.shape
    M = data_all[0, 0, 0].shape[0]
    res = np.empty(data_all.shape, dtype=object)
    for i in range(n_ns):
        for j in range(n_nus):
            for k in range(n_etas):
                res[i, j, k] = []
                data = data_all[i, j, k]
                for m in range(M):
                    res[i, j, k].append(run_cusum(data[m], d))
                res[i, j, k] = np.array(res[i, j, k], dtype=object)
    return res


def get_oe(real, test, T=None):
    if len(test) == 0:
        test = np.array([0])
    p = len(real)
    arr = np.zeros(p)
    for i in range(p):
        arr[i] = np.min(np.abs(test - real[i]))
    return np.max(arr)


def get_ue(real, test, T=None):
    if len(test) == 0:
        test = np.array([0])
    q = len(test)
    arr = np.zeros(q)
    for i in range(q):
        arr[i] = np.min(np.abs(test[i] - real))
    return np.max(arr)


def get_e(real, test):
    return np.abs(len(real) - len(test))


def sum_res(data, real_cp):
    num_cp = []
    num_cp_ref = len(real_cp)
    for cp in data:
        num_cp.append(len(cp))
    num_cp = np.array(num_cp)
    avg_num_cp = np.mean(num_cp)
    prob_less = np.mean(num_cp < num_cp_ref)
    prob_more = np.mean(num_cp > num_cp_ref)
    prob_equal = np.mean(num_cp == num_cp_ref)
    prob_detect = []
    for x in real_cp:
        temp = []
        for r in data:
            temp.append(x in r)
        prob_detect.append(np.mean(temp))
    return np.concatenate(([avg_num_cp, prob_equal, prob_less, prob_more], prob_detect))


def prf_res(data, real_cp, win_size=None):
    p = []
    r = []
    def hit_cp(cp_raw, real_cp, win_size=None):
        if win_size is None:
            return np.intersect1d(cp_raw, real_cp)
        else:
            cp_new = []
            l = len(cp_raw)
            for cp_temp in real_cp:
                find = False
                i = 0
                while (not find) and i < l:
                    if cp_raw[i] > cp_temp - win_size and cp_raw[i] < cp_temp + win_size:
                        cp_new.append(cp_temp)
                        find = True
                    i += 1
            return cp_new
    for cp in data:
        int_cp = hit_cp(cp_raw=cp, real_cp=real_cp, win_size=win_size)
        n_int = len(int_cp)
        p.append(n_int / len(cp) if len(cp) > 0  else 0)
        r.append(n_int / len(real_cp))
    p = np.array(p)
    r = np.array(r)
    f = []
    for i, j in zip(p, r):
        if i + j > 0:
            f.append(2 * i * j / (i + j))
        else:
            f.append(0)
    f = np.array(f)
    txt = '{}({})'
    return np.array([txt.format(np.round(p.mean(), 3), np.round(p.std(), 3)),
                     txt.format(np.round(r.mean(), 3), np.round(r.std(), 3)),
                     txt.format(np.round(f.mean(), 3), np.round(f.std(), 3))])


def sum_table(*data_all, real_cp, index):
    raw = [sum_res(data, real_cp) for data in data_all]
    names = ['prob_detect t=' + str(i) for i in real_cp]
    names = np.concatenate((['avg_num_cp', 'prob_equal', 'prob_less', 'prob_more'], names))
    res = pd.DataFrame(index=index, columns=names)
    for j in range(len(raw)):
        res.iloc[j, :] = raw[j]
    return res


def prf_table(*data_all, real_cp, index, win_size=None):
    raw = [prf_res(data, real_cp, win_size=win_size) for data in data_all]
    names = np.array(['precision', 'recall', 'f-score'])
    res = pd.DataFrame(index=index, columns=names)
    for j in range(len(raw)):
        res.iloc[j, :] = raw[j]
    return res


def sum_table_all(*res_all, ns, nus, etas, real_cp, index):
    n_ns, n_nus, n_etas = len(ns), len(nus), len(etas)
    table_all = []
    for i in range(n_ns):
        table_temp_n = []
        for j in range(n_nus):
            table_temp_nu = []
            for k in range(n_etas):
                table_temp_nu.append(sum_table(*[res[i, j, k] for res in res_all], real_cp=real_cp, index=index))
            table_temp_n.append(pd.concat(table_temp_nu, axis='columns', keys=['eta=' + str(etas[s]) for s in range(n_etas)]))
        table_all.append(pd.concat(table_temp_n, axis='rows', keys=['nu=' + str(nus[s]) for s in range(n_nus)]))
    table_all = pd.concat(table_all, axis='rows', keys=['n=' + str(ns[s]) for s in range(n_ns)])
    return table_all


def prf_table_all(*res_all, ns, nus, etas, real_cp, index, win_size=None):
    n_ns, n_nus, n_etas = len(ns), len(nus), len(etas)
    table_all = []
    for i in range(n_ns):
        table_temp_n = []
        for j in range(n_nus):
            table_temp_nu = []
            for k in range(n_etas):
                table_temp_nu.append(prf_table(*[res[i, j, k] for res in res_all], real_cp=real_cp, index=index, win_size=win_size))
            table_temp_n.append(pd.concat(table_temp_nu, axis='columns', keys=['eta=' + str(etas[s]) for s in range(n_etas)]))
        table_all.append(pd.concat(table_temp_n, axis='rows', keys=['nu=' + str(nus[s]) for s in range(n_nus)]))
    table_all = pd.concat(table_all, axis='rows', keys=['n=' + str(ns[s]) for s in range(n_ns)])
    return table_all


def prf_table_all_ng(*res_all, ns, etas, real_cp, switch, index, win_size=None):
    txt = 'pwr=' if switch else 'eta='
    n_ns, n_etas = len(ns), len(etas)
    table_all = []
    for i in range(n_ns):
        table_temp_n = []
        for j in range(n_etas):
            table_temp_n.append(prf_table(*[res[i, j] for res in res_all], real_cp=real_cp, index=index, win_size=win_size))
        table_all.append(pd.concat(table_temp_n, axis='columns', keys=[txt + str(etas[s]) for s in range(n_etas)]))
    table_all = pd.concat(table_all, axis='rows', keys=['n=' + str(ns[s]) for s in range(n_ns)])
    return table_all


def get_prob_all(T, d, g, cp, eta, nu):
    p = np.zeros((T + 1, d))
    p[0, ] = np.repeat(1 / d, d)
    state = 0
    for t in range(1, T + 1):
        g_temp = g[t - 1, ] ** nu
        p[t, ] = p[t - 1, ] * g_temp
        p[t, ] = p[t, ] / np.sum(p[t, ])
        if t - 1 in cp:
            if state % 2 == 0:
                for d_temp in range(d):
                    if d_temp % 2 == 0:
                        p[t, d_temp] *= np.exp(eta)
                    else:
                        p[t, d_temp] *= np.exp(-eta)
            else:
                for d_temp in range(d):
                    if d_temp % 2 == 0:
                        p[t, d_temp] *= np.exp(-eta)
                    else:
                        p[t, d_temp] *= np.exp(eta)
            p[t, ] = p[t, ] / np.sum(p[t, ])
            state += 1
    return p


def get_prob_ng_all(T, d, cp, eta):
    p = np.zeros((T + 1, d))
    p[0, ] = np.ones(d)
    p[0, ] = p[0, ] / np.sum(p[0, ])
    state = 0
    for t in range(1, T + 1):
        p[t, ] = p[t - 1, ]
        if t - 1 in cp:
            if state % 2 == 0:
                for d_temp in range(d):
                    if d_temp % 2 == 0:
                        p[t, d_temp] *= np.exp(eta)
                    else:
                        p[t, d_temp] *= np.exp(-eta)
            else:
                for d_temp in range(d):
                    if d_temp % 2 == 0:
                        p[t, d_temp] *= np.exp(-eta)
                    else:
                        p[t, d_temp] *= np.exp(eta)
            p[t, ] = p[t, ] / np.sum(p[t, ])
            state += 1
    return p


def check_unbalanced_tmap_conv(p1, p2, d, costm, reg, reg1, reg2, n_size, n_sim):
    p1_sim = np.zeros((n_sim, d))
    p2_sim = np.zeros((n_sim, d))
    for i in range(n_sim):
        p1_sim[i, ] = np.random.multinomial(n_size, p1) / n_size
        p2_sim[i, ] = np.random.multinomial(n_size, p2) / n_size
    tmap_sim = solver.ot_unbalanced_all(p1_sim.T, p2_sim.T, costm, reg, reg1, reg2)
    tmap_real = solver.ot_unbalanced(p1, p2, costm, reg, reg1, reg2)
    diff_sim = tmap_compare_mc(tmap_sim, tmap_real)
    return {'p1_sim': p1_sim,
            'p2_sim': p2_sim,
            'tmap_sim': tmap_sim,
            'tmap_real': tmap_real,
            'diff': diff_sim}


def plot_diff(*res, ns, title=None):
    plt.figure()
    for i in range(len(ns)):
        sns.kdeplot(res[i]['diff'], label='n = ' + str(ns[i]))
    plt.legend()
    plt.xlabel('diff')
    if title is not None:
        plt.title(title)


def dgp_from_prob(prob, n, T, d, M):
    # generate data given probability vectors over all times
    data_mc = []
    for m in range(M):
        data_temp = np.zeros((T + 1, n), dtype=int)
        for t in range(T + 1):
            data_temp[t, ] = np.random.choice(np.arange(d), size=n, p=prob[t, ])
        data_mc.append(data_temp)
    return np.array(data_mc)
    

def dgp_with_prob(prob, n, matformat=True):
    # rescaling of probability vectors
    M, T, d = prob.shape
    if matformat:
        res = np.zeros((T, d, M))
        for m in range(M):
            res[:, :, m] = np.array([n * prob[m, t, ] for t in range(T)])
    else:
        res = np.zeros((M, T, d))
        for m in range(M):
            res[m, :, :] = np.array([n * prob[m, t, ] for t in range(T)])
    return np.array(res)


def dgp(nu, eta, cp, g, d, T, n, M):
    data_mc = []
    sep = int(d / 2)
    B = np.ones((d, d))
    # B1[:, sep:] = eta
    # B2[:, :sep] = eta
    for i in range(d):
        for j in range(d):
            if i < d / 2:
                if j >= d / 2:
                    B[i, j] = eta / (d - sep)
                else:
                    B[i, j] = 0
            else:
                if j < d / 2:
                    B[i, j] = eta / sep
                else:
                    B[i, j] = 0
        B[i, i] = 1 - eta
    # B = B1 = B2 = np.random.rand(d, d)
    # B = np.ones((d, d))
    B = np.diag(1 / B.sum(axis=1)) @ B
    p = np.zeros((T + 1, d))
    p[0, ] = np.repeat(1 / d, d)
    for t in range(1, T + 1):
        g_temp = g[t - 1, ] ** nu
        p[t, ] = p[t - 1, ] * g_temp
        p[t, ] = p[t, ] / np.sum(p[t, ])
        if t - 1 in cp:
            # p[t, ] = np.repeat(1 / d, d)
            p[t, ] = B.T @ p[t, ]
    data_mc = []        
    for m in range(M):
        data_temp = np.zeros((T + 1, n), dtype=int)
        for t in range(T + 1):
            data_temp[t, ] = np.random.choice(np.arange(d), size=n, p=p[t, ])
        data_mc.append(data_temp)
    return np.array(data_mc)


def dgp_no_trans(nu, eta, cp, g, d, T, n, M):
    data_mc = []
    sep = int(d / 2)
    p = np.zeros((T + 1, d))
    p[0, ] = np.repeat(1 / d, d)
    state = 0
    for t in range(1, T + 1):
        g_temp = g[t - 1, ] ** nu
        p[t, ] = p[t - 1, ] * g_temp
        p[t, ] = p[t, ] / np.sum(p[t, ])
        if t - 1 in cp:
            if state % 2 == 0:
                for d_temp in range(d):
                    if d_temp < sep:
                        p[t, d_temp] *= np.exp(eta)
                    else:
                        p[t, d_temp] *= np.exp(-eta)
            else:
                for d_temp in range(d):
                    if d_temp < sep:
                        p[t, d_temp] *= np.exp(-eta)
                    else:
                        p[t, d_temp] *= np.exp(eta)
            p[t, ] = p[t, ] / np.sum(p[t, ])
            state += 1
    data_mc = []        
    for m in range(M):
        data_temp = np.zeros((T + 1, n), dtype=int)
        for t in range(T + 1):
            data_temp[t, ] = np.random.choice(np.arange(d), size=n, p=p[t, ])
        data_mc.append(data_temp)
    return np.array(data_mc)


def dgp_ng(eta, cp, d, T, n, M):
    data_mc = []
    sep = int(d / 2)
    B1 = np.zeros((d, d))
    B2 = np.zeros((d, d))
    # B1[:, sep:] = eta
    # B2[:, :sep] = eta
    for i in range(d):
        for j in range(d):
            if i < d / 2:
                if j >= d / 2:
                    B1[i, j] = eta / (d - sep)
                elif j == i:
                    B1[i, j] = 1 - eta
                    B2[i, j] = 1
            else:
                if j < d / 2:
                    B2[i, j] = eta / sep
                elif j == i:
                    B1[i, j] = 1
                    B2[i, j] = 1 - eta
    # B = B1 = B2 = np.random.rand(d, d)
    # B = np.ones((d, d))
    B1 = np.diag(1 / B1.sum(axis=1)) @ B1
    B2 = np.diag(1 / B2.sum(axis=1)) @ B2
    p = np.zeros((T + 1, d))
    p[0, ] = np.ones(d)
    p[0, ] = p[0, ] / np.sum(p[0, ])
    state = 0
    for t in range(1, T + 1):
        p[t, ] = p[t - 1, ]
        if t - 1 in cp:
            B = B1 if state % 2 == 0 else B2
            # p[t, ] = np.repeat(1 / d, d)
            p[t, ] = B.T @ p[t, ]
            state += 1
    data_mc = []        
    for m in range(M):
        data_temp = np.zeros((T + 1, n), dtype=int)
        for t in range(T + 1):
            data_temp[t, ] = np.random.choice(np.arange(d), size=n, p=p[t, ])
        data_mc.append(data_temp)
    return np.array(data_mc)


def dgp_ng_no_trans(eta, cp, d, T, n, M):
    data_mc = []
    sep = int(d / 2)
    p = np.zeros((T + 1, d))
    p[0, ] = np.ones(d)
    p[0, ] = p[0, ] / np.sum(p[0, ])
    state = 0
    for t in range(1, T + 1):
        p[t, ] = p[t - 1, ]
        if t - 1 in cp:
            if state % 2 == 0:
                for d_temp in range(d):
                    if d_temp < sep:
                        p[t, d_temp] *= np.exp(eta)
                    else:
                        p[t, d_temp] *= np.exp(-eta)
            else:
                for d_temp in range(d):
                    if d_temp < sep:
                        p[t, d_temp] *= np.exp(-eta)
                    else:
                        p[t, d_temp] *= np.exp(eta)
            p[t, ] = p[t, ] / np.sum(p[t, ])
            state += 1
    data_mc = []        
    for m in range(M):
        data_temp = np.zeros((T + 1, n), dtype=int)
        for t in range(T + 1):
            data_temp[t, ] = np.random.choice(np.arange(d), size=n, p=p[t, ])
        data_mc.append(data_temp)
    return np.array(data_mc)


def dgp_ng_switch(pwr, cp, d, T, n, M):
    data_mc = []  
    theta = 0
    for m in range(M):
        data_temp = np.zeros((T + 1, n), dtype=int)
        for t in range(T + 1):
            p_temp = np.ones(d)
            if theta % 2 == 0:
                p_temp[:d // 2] = pwr
            else:
                p_temp[d // 2:] = pwr
            if t in cp:
                theta += 1
            p_temp = p_temp / np.sum(p_temp)
            data_temp[t, ] = np.random.choice(np.arange(d), size=n, p=p_temp)
        data_mc.append(data_temp)
    return np.array(data_mc)


def get_res_from_others(res_raw, n_ns, n_nus, n_etas, M, ftype=None):
    if ftype == 'mat':
        res = np.empty((n_ns, n_nus, n_etas), dtype=object)
        for i in range(n_ns):
            for j in range(n_nus):
                for k in range(n_etas):
                    res[i, j, k] = []
                    for m in range(M):
                        res[i, j, k].append(res_raw[i, j, k, m].flatten().astype(dtype=int))
                    res[i, j, k] = np.array(res[i, j, k], dtype=object)
    elif ftype == 'r':
        res = np.empty((n_ns, n_nus, n_etas), dtype=object)
        for i in range(n_ns):
            for j in range(n_nus):
                for k in range(n_etas):
                    res[i, j, k] = []
        i = j = k = m = 0
        l_temp = len(res_raw)
        temp = []
        for ind in range(l_temp):
            if not res_raw[ind] == 'done':
                if not res_raw[ind] == 'null':
                    temp.append(int(res_raw[ind]))
            else:
                res[i, j, k].append(np.array(temp))
                temp = []
                m += 1
                if m == M:
                    k += 1
                    m = 0
                if k == n_etas:
                    j += 1
                    k = 0
                if j == n_nus:
                    i += 1
                    j = 0
        for i in range(n_ns):
            for j in range(n_nus):
                for k in range(n_etas):
                    res[i, j, k] = np.array(res[i, j, k], dtype=object)
    return res


def get_res_from_others_ng(res_raw, n_ns, n_temp, M, ftype=None):
    if ftype == 'mat':
        res = np.empty((n_ns, n_temp), dtype=object)
        for i in range(n_ns):
            for j in range(n_temp):
                res[i, j] = []
                for m in range(M):
                    res[i, j].append(res_raw[i, j, m].flatten().astype(dtype=int))
                res[i, j] = np.array(res[i, j], dtype=object)
    elif ftype == 'r':
        res = np.empty((n_ns, n_temp), dtype=object)
        for i in range(n_ns):
            for j in range(n_temp):
                res[i, j] = []
        i = j = m = 0
        l_temp = len(res_raw)
        temp = []
        for ind in range(l_temp):
            if not res_raw[ind] == 'done':
                if not res_raw[ind] == 'null':
                    temp.append(int(res_raw[ind]))
            else:
                res[i, j].append(np.array(temp))
                temp = []
                m += 1
                if m == M:
                    j += 1
                    m = 0
                if j == n_temp:
                    i += 1
                    j = 0
        for i in range(n_ns):
            for j in range(n_temp):
                res[i, j] = np.array(res[i, j], dtype=object)
    return res


def get_weight_no_ignore_mc(data_mc, T, k, matformat=True, count=True):
    M = len(data_mc)
    get_stat = get_count if count else get_weight
    if matformat:
        res = np.zeros((T + 1, k, M))
        for m in range(M):
            res[:, :, m] = np.array([get_stat(data_mc[m][t, :], k) for t in range(T + 1)])
    else:
        res = np.zeros((M, T + 1, k))
        for m in range(M):
            res[m, :, :] = np.array([get_stat(data_mc[m][t, :], k) for t in range(T + 1)])
    return np.array(res)


def get_ot_unbalanced_cost_mc(data_mc, costm, reg, reg1, reg2, sink):
    M = len(data_mc)
    T = data_mc.shape[1] - 1
    res = np.zeros((M, T))
    for m in range(M):
        for t in range(T):
            res[m, t] = solver.loss_unbalanced(data_mc[m, t], data_mc[m, t + 1], costm, reg, reg1, reg2, sink=sink, single=True)
    return res


def get_ot_unbalanced_cost_local_mc(data_mc, costm, reg, reg1, reg2, sink, win_size=None, weight=None):
    if win_size is None:
        return get_ot_unbalanced_cost_mc(data_mc, costm, reg, reg1, reg2, sink=sink)
    else:
        M = len(data_mc)
        res = []
        for m in range(M):
            res.append(solver.loss_unbalanced_all_local(data_mc[m], costm, reg, reg1, reg2, sink=sink, win_size=win_size, weight=weight))
        return np.array(res)


def get_ot_unbalanced_cost_mm_mc(data_mc, costm, reg, reg_phi, coeff, win_size, n_iter=10, exp_threshold=10):
    M = len(data_mc)
    res = []
    for m in range(M):
        res.append(solver.multimarg_unbalanced_ot_all(data_mc[m], costm, reg, reg_phi, win_size, coeff=coeff, n_iter=n_iter, exp_threshold=exp_threshold))
    return np.array(res)


def tmap_compare_mc(tmap_all, real_tmap):
    M = tmap_all.shape[0]
    res = np.zeros(M)
    for m in range(M):
        res[m] = np.sum(np.abs(tmap_all[m] - real_tmap))
    return res


# def multisetting_cp_detection(nus, etas, ns, d, T, M, seed, costm, reg, reg1, reg2, balanced, sink, n_conv, *args, **method):
#     np.random.seed(seed)
#     def cp_all(data_all, cp_method, *param):
#         cp = []
#         for data in data_all:
#             cp.append(cp_method(data, *param))
#         return np.array(cp, dtype=object)
#     names = [*method.keys()]
#     funcs = [*method.values()]
#     n_func = len(names)
#     res_total = {}
#     for n in ns:
#         res_n0 = {}
#         for nu in nus:
#             res_nu0 = []
#             for eta in etas:
#                 data_mc, real_cp = dgp(nu, eta, 0, d, T, n, M)
#                 ot_temp = np.array([cp_detection_ot(data_mc[m], d, costm, reg, reg1, reg2, sink=sink, balanced=balanced, n_conv=n_conv) for m in range(M)][0], dtype=object)
#                 cs_temp = np.array([run_cusum(data_mc[m], d) for m in range(M)], dtype=object)
#                 res_nu0.append(sum_table(ot_temp, cs_temp, real_cp=real_cp, index=['OT', 'CS']))
#             res_n0['nu=' + str(nu)] = pd.concat(res_nu0, axis='columns', keys=['eta=' + str(e) for e in etas])
#         res_total['n=' + str(n)] = pd.concat(res_n0)
#     res_total = pd.concat(res_total)    
#     return res_total


def sim_costm(d, low_close, high_close, low_far, high_far):
    costm = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            if i < d / 2:
                if j >= d / 2:
                    costm[i, j] = np.random.uniform(low=low_far, high=high_far)
                    costm[j, i] = costm[i, j]
                else:
                    costm[i, j] = np.random.uniform(low=low_close, high=high_close)
                    costm[j, i] = costm[i, j]
            else:
                if j < d / 2:
                    costm[i, j] = np.random.uniform(low=low_far, high=high_far)
                    costm[j, i] = costm[i, j]
                else:
                    costm[i, j] = np.random.uniform(low=low_close, high=high_close)
                    costm[j, i] = costm[i, j]
        np.fill_diagonal(costm, 0)
    return costm
                

def constant_costm(d, low, high):
    costm = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            if i < d / 2:
                if j >= d / 2:
                    costm[i, j] = high
                    costm[j, i] = costm[i, j]
                else:
                    costm[i, j] = low
                    costm[j, i] = costm[i, j]
            else:
                if j < d / 2:
                    costm[i, j] = high
                    costm[j, i] = costm[i, j]
                else:
                    costm[i, j] = low
                    costm[j, i] = costm[i, j]
    np.fill_diagonal(costm, 0)
    return costm


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
# reg = 1
# reg1 = 1
# reg2 = 50

# index = 88

# balanced = False
# sink = True

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
#                                 np.repeat('boot', zs.shape[0])))
# sns.kdeplot(data=df, x='value', hue='source', common_norm=False)
##################################################

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
