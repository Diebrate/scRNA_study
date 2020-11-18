import numpy as np

def ot_entropy(a, b, costm, reg, n_iter=1000):
    tmap = np.exp(-costm / reg)
    for i in range(n_iter):
        tmap = np.diag(a) @ np.diag(1 / tmap.sum(axis=1)) @ tmap
        tmap = tmap @ np.diag(1 / tmap.sum(axis=0)) @ np.diag(b)
    return tmap / tmap.sum()

def ot_entropy_uv(a, b, costm, reg, n_iter=1000, fullreturn=True):
    K = np.exp(-costm / reg)
    u = np.repeat(1, len(a))
    v = np.repeat(1, len(b))
    for i in range(n_iter):
        v = b / (np.transpose(K) @ u)
        u = a / (K @ v)
    if fullreturn:
        return {'tmap': np.diag(u) @ K @ np.diag(v),
            'uv': (u, v),
            'K': K}
    else:
        return u, v