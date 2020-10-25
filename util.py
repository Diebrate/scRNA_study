import numpy as np
# import pandas as pd
import ot
import scipy.stats
import scipy.special
import statsmodels.api as sm
# import scipy.linalg
from scipy import spatial
# from statsmodels.nonparametric.kernel_density import KDEMultivariate
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as hierarchy
from sklearn.metrics import calinski_harabasz_score


def decimal_trunc(x, n_dec):
    x_temp = x * (10 ** n_dec)
    x_temp = np.trunc(x_temp)
    return x_temp * (10 ** -n_dec)


def get_weight_and_value(data, pca_dim, use_kernel = False, h = None):
    pca = PCA(pca_dim)
    data_cleaned = pca.fit_transform(data)
    l = len(data_cleaned)
    p = np.ones(l)
    return p/p.sum(), data_cleaned


def get_cost_matrix(x, y, dim, method='euclidean'):
    if dim == 1:
        x_reshape, y_reshape = np.reshape(x, (-1, dim)), np.reshape(y, (-1, dim))
        c_matrix = spatial.distance.cdist(x_reshape, y_reshape, metric=method)
    else:
        c_matrix = spatial.distance.cdist(x, y, metric=method)
    return c_matrix


def renormalize_weight(M,p1,t1,t2):
    g = np.power(M.sum(axis=1), t2-t1)
    p = np.multiply(p1, g)
    return p/p.sum()


def renormalize_matrix(M, frac):
    '''return normalized transport matrix (sum up to 1)
    
    
    Parameters:
        M:      un-normalized transport matrix
        frac:   (1 - frac) * x + frac * y
        
        
    '''
    p = np.array([i / np.power(i.sum(), 1-frac) if not i.sum() == 0 else np.zeros(len(i)) for i in M])
    p = p/p.sum()
    return p.flatten(order='C')


# original setting reg=1, reg_m=5
def ot_iter(p1, p2, x1, x2, dim, t1=0, t2=1, reg=1, reg_m=5, n_iter=2):
    d_mat = get_cost_matrix(x1, x2, dim)
    p1_temp = p1
    n = 0
    while n < n_iter:
        map_temp = ot.unbalanced.sinkhorn_stabilized_unbalanced(p1_temp, p2, d_mat, reg=reg, reg_m=reg_m)
        p1_temp = renormalize_weight(map_temp, p1_temp, t1, t2) + 0.001
        p1_temp = p1_temp/p1_temp.sum()
        n += 1
    return map_temp


def ot_iter_cluster(p1, p2, x1, x2, dim, reg=10):
    d_mat = get_cost_matrix(x1, x2, dim)
    map_temp = ot.sinkhorn(p1, p2, d_mat, reg=reg)
    return map_temp


def compute_all_ot(data, dim, **kwargs):
    T = data.shape[0]
    all_ot_matrix = []
    for t in range(T-1):
        x_temp, y_temp = data[t], data[t+1]
        px_temp = np.repeat(1/len(x_temp), len(x_temp))
        py_temp = np.repeat(1/len(y_temp), len(y_temp))
        all_ot_matrix.append(ot_iter(px_temp, py_temp, x_temp, y_temp, dim=dim, t1=0, t2=1, **kwargs))
    return np.array(all_ot_matrix)


def compute_all_ot_cluster(data, labels, dim, **kwargs):
    # compute all OT matrices clustering at single time points
    T = data.shape[0]
    all_ot_matrix = []
    for t in range(T-1):
        x_temp, y_temp = data[t], data[t+1]
        px_temp = get_prob_cluster(labels[t])
        py_temp = get_prob_cluster(labels[t+1])
        all_ot_matrix.append(ot_iter_cluster(px_temp, py_temp, x_temp, y_temp, dim=dim, **kwargs))
    return np.array(all_ot_matrix)


def compute_all_ot_cluster2(centroids, probs, dim, **kwargs):
    # compute all OT matrices for total clustering
    T = probs.shape[0]
    k = centroids.shape[0]
    all_ot_matrix = []
    for t in range(T-1):
        px_temp, py_temp = probs[t], probs[t+1]
        for i in range(k):
            if px_temp[i] == 0:
                px_temp[i] += 0.00000001
            if py_temp[i] == 0:
                py_temp[i] += 0.00000001
            px_temp = px_temp / np.sum(px_temp)
            py_temp = py_temp / np.sum(py_temp)
        all_ot_matrix.append(ot_iter_cluster(px_temp, py_temp, centroids, centroids, dim=dim, **kwargs))
    return np.array(all_ot_matrix)


def interpolate(x1, x2, p1, p2, M, size, frac=0.5, value_only=True):
    I = len(p1)
    J = len(p2)
    p = renormalize_matrix(M, frac)
    choices = np.random.choice(I*J, p=p, size=size)
    values = np.array([x1[i // J] * (1 - frac) + x2[i % J] * frac for i in choices])
    if value_only:
        return values
    else:
        w = np.array([p1[i // J] * (1 - frac) + p2[i % J] * frac for i in choices])
        return w/w.sum(), values, choices
    

def interpolate_with_noise(x1, x2, M, size, frac=0.5):
    I = len(x1)
    J = len(x2)
    # dim = x1.shape[1]
    p = renormalize_matrix(M, frac)
    choices = np.random.choice(I*J, p=p, size=size)
    values = []
    # d_pool = np.vstack((x1, x2)).transpose()
    # cov = np.cov(d_pool) / (I + J)
    for i in choices:
        mean = x1[i // J] * (1 - frac) + x2[i % J] * frac
        # sd = np.sqrt(sum(np.power(x2[i % J] - x1[i // J], 2))) * 100
        # cov = np.diag(np.repeat(sd, dim))
        # values.append(np.random.multivariate_normal(mean=mean, cov=cov))
        values.append(mean)
    return np.array(values)


def interpolate_with_kernel(x1, x2, M, size, frac=0.5):
    I = len(x1)
    J = len(x2)
    p = renormalize_matrix(M, frac=frac)
    choices = np.random.choice(I*J, p=p, size=size)
    values = []
    for i in choices:
        k = np.random.choice([0, 1], p=[1 - frac, frac])
        if k == 0:
            values.append(np.random.multivariate_normal(mean=x1[i // J], cov=np.diag([1, 1])))
        else:
            values.append(np.random.multivariate_normal(mean=x2[i % J], cov=np.diag([1, 1])))
    return np.array(values)


# def interpolate_with_kernel_simple(x1, x2, M, size, frac=0.5):
#     I = len(p1)
#     J = len(p2)
#     p = renormalize_matrix(M, frac)
#     choices = np.random.choice(I*J, p=p, size=size)
#     values = np.array([x1[i // J] * (1 - frac) + x2[i % J] * frac for i in choices])
#     def kernel(x, y, z, frac):
#         k_xz = np.exp(-np.linalg.norm(x - z) ** 2 * 0.5)
#         k_yz = np.exp(-np.linalg.norm(y - z) ** 2 * 0.5)
#         return (1 - frac) * k_xz + frac * k_yz
#     l = values.shape[0]
#     for i in range(l):
        
        

# def interpolate_with_kernel1(x1, x2, M, size, frac=0.5):
#     def gauss_kernel(x, z):
#         return np.exp(-np.linalg.norm(x - z) ** 2 * 0.5)
#     def get_p_hat(x1, x2, M, z):
#         n, m = M.shape
#         phat = 0
#         for i in range(n):
#             for j in range(m):
#                 phi_temp = (1 - frac) * gauss_kernel(x1[i,:], z) + frac * gauss_kernel(x2[j,:], z)
#                 phat += phi_temp * M[i, j]
#         return phat
#     I, J = M.shape
#     M = np.reshape(renormalize_matrix(M, 0.5), (I, J))
#     points = np.transpose(get_grid(x1, x2))
#     l = points.shape[0]
#     density = []
#     for k in range(l):
#         density.append(get_p_hat(x1, x2, M, points[k, :]))
#     density = density/np.sum(density)
#     return points[np.random.choice(range(l), size=size, p=density),:]


def get_grid(x, y, grid_size=25, delta=0.1):
    z = np.vstack((x, y))
    span1 = z[0,:].min(), z[0,:].max()
    span2 = z[1,:].min(), z[1,:].max()
    l1, l2 = span1[1] - span1[0], span2[1] - span2[0]
    axis1, axis2 = np.meshgrid(np.linspace(span1[0]-delta*l1, span1[1]+delta*l1, grid_size),
                               np.linspace(span2[0]-delta*l2, span2[1]+delta*l2, grid_size))
    return np.vstack((axis1.flatten(), axis2.flatten()))
    

def get_kernel_weight_graph(data):
    # x_lim = np.linspace(data[:, 0].min(), data[:, 0].max(), n_points)
    # y_lim = np.linspace(data[:, 1].min(), data[:, 1].max(), n_points)
    kde = scipy.stats.gaussian_kde(data.transpose(), bw_method=None)
    # return kde.pdf(np.vstack((x_lim, y_lim)))
    return kde.pdf(data.transpose())


def kernel_div_test(x, y, metric, dim, n_points, bandwidth=None):

    def kl_div(px, py):
        if not len(px) == len(py):
            raise Exception(" x and y do not have the same length.")
        else:
            sum_div = 0
            l = len(px)
            for i in range(l):
                if not px[i] == py[i]:
                    sum_div += px[i] * np.log(px[i] / py[i])
            return sum_div

    def l1_div(px, py):
        if not len(px) == len(py):
            raise Exception(" x and y do not have the same length.")
        else:
            return sum([abs(i - j) for i, j in zip(px, py)])

    def get_kernel_weight(x, points, bw=None):
        x_kde = scipy.stats.gaussian_kde(x.transpose(), bw_method=bw)
        return x_kde.pdf(points)

    n_points = int(n_points**(1/dim))

    if dim == 1:
        points = np.linspace(min(x.min(), y.min()), max(x.max(), y.max()), n_points)
    elif dim == 2:
        d1 = np.append(x.transpose()[0], y.transpose()[0])
        d2 = np.append(x.transpose()[1], y.transpose()[1])
        axis1 = np.linspace(min(d1), max(d1), n_points)
        axis2 = np.linspace(min(d2), max(d2), n_points)
        coord1, coord2 = np.meshgrid(axis1, axis2)
        points = np.vstack((coord1.flatten(), coord2.flatten()))
    elif dim == 3:
        d1 = np.append(x.transpose()[0], y.transpose()[0])
        d2 = np.append(x.transpose()[1], y.transpose()[1])
        d3 = np.append(x.transpose()[2], y.transpose()[2])
        axis1 = np.linspace(min(d1), max(d1), n_points)
        axis2 = np.linspace(min(d2), max(d2), n_points)
        axis3 = np.linspace(min(d3), max(d3), n_points)
        coord1, coord2, coord3 = np.meshgrid(axis1, axis2, axis3)
        points = np.vstack((coord1.flatten(), coord2.flatten(), coord3.flatten()))
    else:
        raise ValueError('Invalid dimension.')

    if type(bandwidth) in (float, str):
        bwx = bandwidth
        bwy = bandwidth
    elif bandwidth is None:
        bwx = 1 * (x.max() - x.min()) / len(x)
        bwy = 1 * (y.max() - y.min()) / len(y)
    else:
        raise TypeError('Invalid bandwidth.')

    px = get_kernel_weight(x, points, bw=bwx)
    py = get_kernel_weight(y, points, bw=bwy)

    if metric == 'kl':
        return kl_div(px, py)
    elif metric == 'l1':
        return l1_div(px, py)
    else:
        raise ValueError('Unknown metric. Try \'kl\' or \'l1\'')


def rank_test(test_func, x, y, *, tail, metric, n_times=2000, **kwargs):
    if not x.shape[1] == y.shape[1]:
        raise ValueError('x and y have different dimensions.')
    nx = len(x)
    test_statistics = test_func(x, y, metric=metric, **kwargs)
    if x.shape[1] == 1:
        data = np.append(x, y)
    else:
        data = np.vstack((x, y))
    reference = np.empty(n_times)
    for i in range(n_times):
        data_temp = np.copy(data)
        np.random.shuffle(data_temp)
        x_temp = data_temp[:nx]
        y_temp = data_temp[nx:]
        reference[i] = test_func(x_temp, y_temp, metric=metric, **kwargs)
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


def test_triplet(data, stat_func, test_func, dim, *, tail, metric, **kwargs):
    T = data.shape[0]
    t_middle = range(1, T-1)
    div_score = []
    for t in t_middle:
        x_temp, y_temp = data[t-1], data[t+1]
        px_temp = np.repeat(1/len(x_temp), len(x_temp))
        py_temp = np.repeat(1/len(y_temp), len(y_temp))
        ot_temp = ot_iter(px_temp, py_temp, x_temp, y_temp, dim=dim)
        pred_temp = interpolate(x_temp, y_temp, px_temp, py_temp, ot_temp, **kwargs)
        div_score.append(stat_func(test_func, pred_temp, data[t], metric=metric, tail=tail, dim=dim, n_points=200)['p_value'])
    return div_score


def cluster_with_best_k(data, k_max=10):
    if k_max <= 2:
        raise Exception('invalid k')
    dist_opt = 0
    # k_opt = 2
    cluster_opt = None
    for k in range(2, k_max+1):
        data_clustered = KMeans(n_clusters=k).fit(data)
        dist = calinski_harabasz_score(data, data_clustered.labels_)
        if dist >= dist_opt:
            dist_opt = dist
            # k_opt = k
            cluster_opt = data_clustered
    return cluster_opt


def cluster_with_best_k_hier(data, k_max=10, linkage='ward'):
    if k_max <= 2:
        raise Exception('invalid k')
    dist_opt = 0
    # k_opt = 2
    cluster_opt = None
    for k in range(2, k_max+1):
        data_clustered = hierarchy(n_clusters=k, linkage=linkage).fit(data)
        dist = calinski_harabasz_score(data, data_clustered.labels_)
        if dist >= dist_opt:
            dist_opt = dist
            # k_opt = k
            cluster_opt = data_clustered
    return cluster_opt


def get_prob_cluster(label):
    # n = len(label)
    k_max = max(label)
    p = []
    for i in range(k_max+1):
        p.append(np.count_nonzero(label==i))
    p = np.array(p)
    return p / p.sum()
            
def get_sample(data, size):
    choices = np.random.choice(range(data.shape[0]), size=size)
    return data[choices, :]







































