import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.api as sm
import ot
from scipy import spatial
import anndata
import solver


# test data
n = 200
m = 200

# r = np.random.rand(3)
# c = np.random.rand(3)
# r, c = r / sum(r), c / sum(c)
r = [0.3, 0.4, 0.3]
c = [0.3, 0.4, 0.3]
a = np.random.choice(range(3), p=r, size=n)
b = np.random.choice(range(3), p=c, size=m)

px = [0.6, 0.2, 0.2]
py = [0.2, 0.2, 0.6]
x = np.random.choice(range(3), p=px, size=n)
y = np.random.choice(range(3), p=py, size=m)

costm = np.random.rand(3, 3)
reg = 10


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


def diag_test(x, y, costm, reg=10, n_sim=1000):
    k = np.concatenate((x, y)).max() + 1
    px = np.zeros(k)
    py = np.zeros(k)
    for x_temp in x:
        px[x_temp] += 1
    for p in px:
        if p == 0:
            p += 0.00001
    for y_temp  in y:
        py[y_temp] += 1
    for p in py:
        if p == 0:
            p += 0.00001
    r = px / sum(px)
    c = py / sum(py)
    res = solver.ot_entropy_uv(r, c, costm, reg)
    tmap = res['tmap']
    u, v = res['uv']
    K = res['K']
    L = len(r)
    D = np.block([[np.diag(1 / (K @ v)), np.zeros((L, L))],
                  [np.zeros((L, L)), np.diag(1 / (K.transpose() @ u))]])
    sigma_uu = -np.outer(r, r)
    np.fill_diagonal(sigma_uu, r * (1 - r))
    sigma_uu = sigma_uu / len(x)
    sigma_vv = -np.outer(c, c)
    np.fill_diagonal(sigma_vv, c * (1 - c))
    sigma_vv = sigma_vv / len(y)
    # sigma_uv = tmap - np.outer(r, c) / max(len(x), len(y))
    sigma_uv = np.zeros((L, L))
    sigma = np.block([[sigma_uu, sigma_uv],
                      [sigma_uv.transpose(), sigma_vv]])
    # p = tmap.flatten()
    # data = np.zeros((n_sim, 2 * L))
    # for i in range(n_sim):
    #     choice = np.random.choice(L ** 2, p=p)
    #     r_temp = np.zeros(L)
    #     c_temp = np.zeros(L)
    #     r_temp[choice // L] = 1
    #     c_temp[choice % L] = 1
    #     p_temp = np.concatenate((r_temp, c_temp))
    #     data[i] = p_temp
    #     # sigma += np.outer(p_temp, p_temp)
    # data_mean = D @ data.mean(axis=0)
    # sigma = np.cov(data, rowvar=False)
    # sigma /= n_sim
    # sigma -= np.outer(data_mean, data_mean)
    zs = []
    data_mean = D @ np.hstack((r, c))
    cov = D @ sigma @ D.transpose()
    # cov = D @ sigma @ D.transpose() / n_sim
    for i in range(n_sim):
        uv_temp = np.random.multivariate_normal(mean = data_mean, cov = cov)
        u_temp = uv_temp[:L]
        u_temp[u_temp < 0] = 0
        v_temp = uv_temp[L:]
        v_temp[v_temp < 0] = 0
        zs.append(np.reshape(u_temp, (1, -1)) @ np.diag(np.diag(K)) @ v_temp)
        # zs.append(np.linalg.norm(np.diag(np.diag(K)) @ v_temp, ord=2))
        # zs.append(np.sum(np.diag(np.diag(K)) @ v_temp))
        # zs.append(np.prod(np.reshape(u_temp, (1, -1)) @ np.diag(np.diag(K)) @ v_temp))
        # zs.append(np.prod(np.diag(np.diag(K)) @ v_temp))
        # zs.append((np.reshape(u_temp, (1, -1)) @ np.diag(np.diag(K)) @ v_temp).max())
    z = np.reshape(u, (1, -1)) @ np.diag(np.diag(K)) @ v
    # z = np.linalg.norm(np.diag(np.diag(K)) @ v, ord=2)
    # z = np.sum(np.diag(np.diag(K)) @ v)
    # z = np.prod(np.reshape(u, (1, -1)) @ np.diag(np.diag(K)) @ v)
    # z = np.prod(np.diag(np.diag(K)) @ v)
    # z = (np.reshape(u, (1, -1)) @ np.diag(np.diag(K)) @ v).max()
    percentile = scipy.stats.percentileofscore(zs, z)
    return percentile / 100


def diag_test2(x, y, costm, reg=10, n_sim=500, n_boot=1000):
    n, m = len(x), len(y)
    L = np.concatenate((x, y)).max() + 1
    r = get_weight(x, L)
    c = get_weight(y, L)
    res = solver.ot_entropy_uv(r, c, costm, reg)
    K = res['K']
    u, v = res['uv']
    D = np.block([[np.diag(1 / (K @ v)), np.zeros((L, L))],
                  [np.zeros((L, L)), np.diag(1 / (K.transpose() @ u))]])
    # mimic sampling distribution of (r, c)
    data = np.zeros((1000, 2 * L))
    for i in range(n_boot):
        x_temp = np.random.choice(x, size=n)
        y_temp = np.random.choice(y, size=m)
        data[i] = np.concatenate((get_weight(x_temp, L), get_weight(y_temp, L)))
    mean = D @ np.mean(data, axis=0)
    cov = D @ np.cov(data, rowvar=False) @ D.transpose()
    zs = []
    for i in range(n_sim):
        uv_temp = np.random.multivariate_normal(mean=mean, cov=cov)
        u_temp = uv_temp[:L]
        u_temp[u_temp < 0] = 0
        v_temp = uv_temp[L:]
        v_temp[v_temp < 0] = 0
        zs.append(np.reshape(u_temp, (1, -1)) @ np.diag(np.diag(K)) @ v_temp)
    z = np.reshape(u, (1, -1)) @ np.diag(np.diag(K)) @ v
    percentile = scipy.stats.percentileofscore(zs, z)
    return percentile / 100


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
    

class OTwork:
    
    method = 'phate'
    method_list = ['raw', 'pca', 'tsne', 'umap', 'phate']
    
    time_names = None
    
    reg = 50
    centroids = None
    
    ot_cluster = {}
    
    def __init__(self, data_path, time_names=None):
        self.data = anndata.read_h5ad(data_path)
        if time_names is None:
            self.time_names = self.data.obs['time'].unique().to_list()
        else:
            self.time_names = time_names
        self.T = len(time_names)
    
    def set_reg(self, x):
        self.reg = x
        
    def get_ot_cluster(self, method=method, inplace=False):
        if method == 'raw':
            df = pd.DataFrame(self.data.X)
            df['cluster'] = self.data.obs['louvain']
            self.centroids = df.groupby('cluster').mean().to_numpy()
        else:
            df = pd.DataFrame()
            df[method + '1'] = self.data.obsm['X_' + method][:, 0]
            df[method + '2'] = self.data.obsm['X_' + method][:, 1]
            df['cluster'] = self.data.obs['louvain']
            self.centroids = df.groupby('cluster').mean().to_numpy()
        costm = get_cost_matrix(self.centroids, self.centroids, dim=self.centroids.shape[1], method='euclidean')
        batch = self.data.obs['batch'].astype('int32')
        tmap = []
        for t in range(self.T - 1):
            p1_temp = self.data.obs['louvain'][batch == t]
            p2_temp = self.data.obs['louvain'][batch == t + 1]
            tmap.append(ot.sinkhorn(p1_temp, p2_temp, M=costm, reg=self.reg))
        if inplace:
            self.ot_cluster.update({method: tmap})
        else:
            return method, self.reg, tmap
    
    def test_cp(tset_func, single=True, index=None):
        pass
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
