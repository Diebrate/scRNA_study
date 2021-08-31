import numpy as np

def ot_entropy(a, b, costm, reg, n_iter=1000):
    tmap = np.exp(-costm / reg)
    for i in range(n_iter):
        tmap = np.diag(a) @ np.diag(1 / tmap.sum(axis=1)) @ tmap
        tmap = tmap @ np.diag(1 / tmap.sum(axis=0)) @ np.diag(b)
    return tmap


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
    

def ot_sinkdiv(a, b, costm, reg, n_iter=1000):
    K = np.exp(-costm / reg - 1)
    u = np.repeat(1, len(a))
    v = np.repeat(1, len(b))
    for i in range(n_iter):
        u = a / (K @ v)
        v = b / (np.transpose(K) @ u)
    return {'tmap': np.multiply(np.diag(u) @ K @ np.diag(v), np.outer(a, b)),
            'uv_dual': (reg * (np.log(u) / a), reg * (np.log(v) / b)),
            'uv_primal': (u, v)}    


def wasserstein_dual(a, b, costm, reg, n_iter=1000):
    u, v = ot_sinkdiv(a, b, costm, reg, n_iter)['uv_dual']
    n, m = costm.shape
    term = 0
    for i in range(n):
        for j in range(m):
            term += a[i] * b[j] * np.exp(-(costm[i, j] - u[i] - v[j]) / reg - 1)
    return np.inner(u, a) + np.inner(v, b) - reg * term


def wasserstein_dual_sink(a, b, costm, reg, n_iter=1000):
    w_ab = wasserstein_dual(a, b, costm, reg)
    w_aa = wasserstein_dual(a, a, costm, reg)
    w_bb = wasserstein_dual(b, b, costm, reg)
    return w_ab - 0.5 * (w_aa + w_bb)


def sink_loss(a, b, costm, reg):
    t_ab = ot_entropy(a, b, costm, reg)
    t_aa = ot_entropy(a, a, costm, reg)
    t_bb = ot_entropy(b, b, costm, reg)
    c = np.sum((t_ab - 0.5 * t_aa - 0.5 * t_bb) * costm)
    return c + reg * np.sum(t_ab * np.log(t_ab) -
                            0.5 * t_aa * np.log(t_aa) - 
                            0.5 * t_bb * np.log(t_bb))


def get_entropy(tmap):
    return np.sum(tmap * np.log(tmap))


def kl_div(p, a):
    return np.sum(p * np.log(p / a))


def ot_balanced(a, b, costm, reg, n_iter=1000):
    tmap = np.exp(-costm / reg)
    for i in range(n_iter):
        tmap = np.diag(a) @ np.diag(1 / tmap.sum(axis=1)) @ tmap
        tmap = tmap @ np.diag(1 / tmap.sum(axis=0)) @ np.diag(b)
    return tmap
    

def ot_unbalanced(a, b, costm, reg, reg1, reg2, n_iter=1000):
    K = np.exp(-costm / reg)
    v = np.repeat(1, len(b))
    for i in range(n_iter):
        u = (a / (K @ v)) ** (reg1 / (reg + reg1))
        v = (b / (np.transpose(K) @ u)) ** (reg2 / (reg + reg2))
    tmap = np.diag(u) @ K @ np.diag(v)
    return tmap / np.sum(tmap)


def ot_unbalanced_uv(a, b, costm, reg, reg1, reg2, n_iter=1000):
    K = np.exp(-costm / reg)
    v = np.repeat(1, len(b))
    for i in range(n_iter):
        u = (a / (K @ v)) ** (reg1 / (reg + reg1))
        v = (b / (np.transpose(K) @ u)) ** (reg2 / (reg + reg2))
    tmap = np.diag(u) @ K @ np.diag(v)
    return u / np.sqrt(np.sum(tmap)), v / np.sqrt(np.sum(tmap)), tmap / np.sum(tmap) 


def ot_unbalanced_uv_all(a, b, costm, reg, reg1, reg2, n_iter=1000):
    K = np.exp(-costm / reg)
    if a.shape[1] == 1:
        a = np.tile(a, b.shape[1])
    if b.shape[1] == 1:
        b = np.tile(b, a.shape[1])
    n = a.shape[1]
    v = np.ones((b.shape[0], n)) / b.shape[0]
    for i in range(n_iter):
        u = (a / (K @ v)) ** (reg1 / (reg + reg1))
        v = (b / (np.transpose(K) @ u)) ** (reg2 / (reg + reg2))
    tmap = [np.diag(i) @ K @ np.diag(j) for i, j in zip(u.transpose(), v.transpose())]
    norm_sum = np.array([np.sum(t_temp) for t_temp in tmap])
    for i in range(n_iter):
        u[:, i] = u[:, i] / np.sqrt(norm_sum[i])
        v[:, i] = v[:, i] / np.sqrt(norm_sum[i])
    return u, v, np.array([t / np.sum(t) for t in tmap])


def sink_gradient_unbalanced(u, v, a, b, reg, reg1, reg2):
    da = -reg1 * np.exp(-u / reg - 1)
    db = -reg2 * np.exp(-v / reg - 1)
    return da, db


def sink_loss_boot_unbalanced(px, py, a, b, costm, reg, reg1, reg2, single=True):
    u, v, tmap = ot_unbalanced_uv(px, py, costm, reg, reg1, reg2)
    tmap_ab = ot_unbalanced_all(a, b, costm, reg, reg1, reg2)
    tmap_aa = ot_unbalanced_all(a, a, costm, reg, reg1, reg2)
    tmap_bb = ot_unbalanced_all(b, b, costm, reg, reg1, reg2)
    def loss(t, pa, pb, m, r, r1, r2):
        c = np.sum(t * (m + r * np.log(t)))
        c += r1 * kl_div(np.sum(t, axis=1), pa) + r2 * kl_div(np.sum(t, axis=0), pb)
        return c
    res_loss = np.array([loss(i, pa, pb, costm, reg, reg1, reg2) - 
                         0.5 * loss(j, pa, pa, costm, reg, reg1, reg2) - 
                         0.5 * loss(k, pb, pb, costm, reg, reg1, reg2) for i, j, k, pa, pb in zip(tmap_ab, tmap_aa, tmap_bb, a.transpose(), b.transpose())])
    n_res = len(res_loss)
    ref = sink_loss_unbalanced(px, py, costm, reg, reg1, reg2)
    da, db = sink_gradient_unbalanced(u, v, px, py, reg, reg1, reg2)
    for nt in range(n_res):
        res_loss[nt] = res_loss[nt] - ref# - np.inner(a[:, nt] - px, da) - np.inner(b[:, nt] - py, db)
    return res_loss


def sink_loss_balanced(a, b, costm, reg):
    t_ab = ot_entropy(a, b, costm, reg)
    t_aa = ot_entropy(a, a, costm, reg)
    t_bb = ot_entropy(b, b, costm, reg)
    c = np.sum((t_ab - 0.5 * t_aa - 0.5 * t_bb) * costm)
    return c + reg * np.sum(t_ab * np.log(t_ab) -
                            0.5 * t_aa * np.log(t_aa) - 
                            0.5 * t_bb * np.log(t_bb))


def wass_loss_balanced(a, b, costm, reg):
    tmap = ot_entropy(a, b, costm, reg)
    return np.sum(tmap * (costm + reg * np.log(tmap)))


def sink_loss_unbalanced(a, b, costm, reg, reg1, reg2):
    t_ab = ot_unbalanced(a, b, costm, reg, reg1, reg2)
    t_aa = ot_unbalanced(a, a, costm, reg, reg1, reg2)
    t_bb = ot_unbalanced(b, b, costm, reg, reg1, reg2)
    c = np.sum((t_ab - 0.5 * (t_aa + t_bb)) * costm)
    c += (get_entropy(t_ab) - 0.5 * (get_entropy(t_aa) + get_entropy(t_bb))) * reg
    c += (kl_div(np.sum(t_ab, axis=1), a) - 0.5 * (kl_div(np.sum(t_aa, axis=1), a) + kl_div(np.sum(t_bb, axis=1), b))) * reg1 
    c += (kl_div(np.sum(t_ab, axis=0), b) - 0.5 * (kl_div(np.sum(t_aa, axis=0), a) + kl_div(np.sum(t_bb, axis=0), b))) * reg2
    return c


def wass_loss_unbalanced(a, b, costm, reg, reg1, reg2):
    tmap = ot_unbalanced(a, b, costm, reg, reg1, reg2)
    c = np.sum(tmap * (costm + reg * np.log(tmap)))
    c += reg1 * kl_div(np.sum(tmap, axis=1), a) + reg2 * kl_div(np.sum(tmap, axis=0), b)
    return c


def ot_balanced_all(a, b, costm, reg, n_iter=1000):
    K = np.exp(-costm / reg)
    if a.shape[1] == 1:
        a = np.tile(a, b.shape[1])
    if b.shape[1] == 1:
        b = np.tile(b, a.shape[1])
    n = a.shape[1]
    v = np.ones((b.shape[0], n)) / b.shape[0]
    for i in range(n_iter):
        u = a / (K @ v)
        v = b / (np.transpose(K) @ u)
    tmap = [np.diag(i) @ K @ np.diag(j) for i, j in zip(u.transpose(), v.transpose())]
    return np.array([t / np.sum(t) for t in tmap])


def ot_unbalanced_all(a, b, costm, reg, reg1, reg2, n_iter=1000):
    K = np.exp(-costm / reg)
    if a.shape[1] == 1:
        a = np.tile(a, b.shape[1])
    if b.shape[1] == 1:
        b = np.tile(b, a.shape[1])
    n = a.shape[1]
    v = np.ones((b.shape[0], n)) / b.shape[0]
    for i in range(n_iter):
        u = (a / (K @ v)) ** (reg1 / (reg + reg1))
        v = (b / (np.transpose(K) @ u)) ** (reg2 / (reg + reg2))
    tmap = [np.diag(i) @ K @ np.diag(j) for i, j in zip(u.transpose(), v.transpose())]
    return np.array([t / np.sum(t) for t in tmap])


def ot_unbalanced_iter(a, b, costm, reg, reg1, reg2, n_iter=1000, n_conv=3):
    a_temp = a.copy()
    for i in range(n_conv):
        tmap = ot_unbalanced(a_temp, b, costm, reg, reg1, 50)
        a_temp = tmap.sum(axis=1)
    return tmap


def ot_unbalanced_all_iter(a, b, costm, reg, reg1, reg2, n_iter=1000, n_conv=3):
    a_temp = a
    l = a.shape[0]
    for i in range(n_conv):
        tmap = ot_unbalanced_all(a_temp, b, costm, reg, reg1, reg2)
        if i < n_conv - 1:
            for j in range(l):
                a_temp[:, j] = np.sum(tmap[j], axis=1)
    return tmap


def ot_unbalanced_all_mc(prob_mc, costm, reg, reg1, reg2):
    M, T, d = prob_mc.shape
    tmap_mc = []
    for m in range(M):
        tmap_mc.append(ot_unbalanced_all(prob_mc[m, :T - 1, :].T, prob_mc[m, 1:, :].T, costm, reg, reg1, reg2))
    return np.array(tmap_mc)


def norm_tmap(tmap):
    tm = np.array(tmap)
    return np.diag(1 / tm.sum(axis=1)) @ tm


def est_trans(ref, raw):
    d = ref.shape[0]
    p_trans = np.zeros((d, d))
    for i in range(d):
        ref_one = ref[i, i]
        ref_zero = (ref[i, ].sum() - ref_one) / (d - 1)
        for j in range(d):
            p_temp = (raw[i, j] - ref_zero) / (ref_one - ref_zero)
            if p_temp < 0:
                p_temp = 0
            elif p_temp > 1:
                p_temp = 1
            p_trans[i, j] = p_temp
    return norm_tmap(p_trans)


def est_trans_from_ts(ref_ts, raw, cp):
    T = len(ref_ts)
    ref = []
    for t in range(T):
        if t in cp:
            ref.append(norm_tmap(ref_ts[t]))
    ref = np.array(ref)
    return est_trans(ref.mean(axis=0), norm_tmap(raw))


def solve_growth(tmap, p, by_row=True):
    if by_row:
        marg = np.sum(tmap, axis=1)
    else:
        marg = np.sum(tmap, axis=0)
    A = np.tile(p, (len(p), 1))
    A = np.diag(marg) * A
    A = A - np.diag(p)
    b = p - marg
    return 1 + np.linalg.solve(A, b)


def estimate_growth1(a, b, costm, reg, reg1, reg2, n_iter=1000, single=True, conv=False):
    if single:
        if not conv:
            tmap = ot_unbalanced(a, b, costm, reg, reg1, reg2)
        else:
            tmap = ot_unbalanced_iter(a, b, costm, reg, reg1, reg2)
        return np.sum(tmap, axis=1) / np.array(a)
    else:
        if not conv:
            tmap = ot_unbalanced_all(a, b, costm, reg, reg1, reg2)
        else:
            tmap = ot_unbalanced_all_iter(a, b, costm, reg, reg1, reg2)
        zs = [solve_growth(t, a_temp, by_row=True) for t, a_temp in zip(tmap, a.transpose())]
        return np.array(zs)
    
    
def estimate_growth2(a, b, costm, reg, reg1, reg2, n_iter=1000, single=True, conv=False):
    if single:
        if not conv:
            tmap = ot_unbalanced(a, b, costm, reg, reg1, reg2)
        else:
            tmap = ot_unbalanced_iter(a, b, costm, reg, reg1, reg2)
        return np.sum(tmap, axis=1) / np.array(a)
    else:
        if not conv:
            tmap = ot_unbalanced_all(a, b, costm, reg, reg1, reg2)
        else:
            tmap = ot_unbalanced_all_iter(a, b, costm, reg, reg1, reg2)
        zs = [solve_growth(t, b_temp, by_row=False) for t, b_temp in zip(tmap, b.transpose())]
        return np.array(zs)


def wass_loss_balanced_all(a, b, costm, reg, n_iter=1000):
    tmap = ot_balanced_all(a, b, costm, reg)
    def loss(t, costm, reg):
        return np.sum(t * (costm + reg * np.log(t)))
    return np.array([loss(t, costm, reg) for t in tmap])


def wass_loss_unbalanced_all(a, b, costm, reg, reg1, reg2, n_iter=1000):
    tmap = ot_unbalanced_all(a, b, costm, reg, reg1, reg2)
    def loss(t, pa, pb, costm, reg, reg1, reg2):
        c = np.sum(t * (costm + reg * np.log(t)))
        c += reg1 * kl_div(np.sum(t, axis=1), pa) + reg2 * kl_div(np.sum(t, axis=0), pb)
        return c
    return np.array([loss(t, pa, pb, costm, reg, reg1, reg2) for t, pa, pb in zip(tmap, a.transpose(), b.transpose())])


def sink_loss_balanced_all(a, b, costm, reg, n_iter=1000):
    tmap_ab = ot_balanced_all(a, b, costm, reg)
    tmap_aa = ot_balanced_all(a, a, costm, reg)
    tmap_bb = ot_balanced_all(b, b, costm, reg)
    def loss(t, m, r):
        return np.sum(t * (m + r * np.log(t)))
    return np.array([loss(i, costm, reg) - 
                     0.5 * loss(j, costm, reg) - 
                     0.5 * loss(k, costm, reg) for i, j, k in zip(tmap_ab, tmap_aa, tmap_bb)])


def sink_loss_unbalanced_all(a, b, costm, reg, reg1, reg2, n_iter=1000):
    tmap_ab = ot_unbalanced_all(a, b, costm, reg, reg1, reg2)
    tmap_aa = ot_unbalanced_all(a, a, costm, reg, reg1, reg2)
    tmap_bb = ot_unbalanced_all(b, b, costm, reg, reg1, reg2)
    def loss(t, pa, pb, m, r, r1, r2):
        c = np.sum(t * (m + r * np.log(t)))
        c += r1 * kl_div(np.sum(t, axis=1), pa) + r2 * kl_div(np.sum(t, axis=0), pb)
        return c
    return np.array([loss(i, pa, pb, costm, reg, reg1, reg2) - 
                     0.5 * loss(j, pa, pa, costm, reg, reg1, reg2) - 
                     0.5 * loss(k, pb, pb, costm, reg, reg1, reg2) for i, j, k, pa, pb in zip(tmap_ab, tmap_aa, tmap_bb, a.transpose(), b.transpose())])


def loss_balanced(a, b, costm, reg, sink=True, single=True):
    def loss_single(t, m, r):
        c = np.sum(t * (m + r * np.log(t)))
        return c
    def loss_all(t_all, m, r):
        return np.array([loss_single(i, m, r) for i in t_all])
    comp = ot_balanced if single else ot_balanced_all
    loss = loss_single if single else loss_all
    if sink:
        tmap_ab = comp(a, b, costm, reg)
        tmap_aa = comp(a, a, costm, reg)
        tmap_bb = comp(b, b, costm, reg)
        c = loss(tmap_ab, costm, reg)
        c -= 0.5 * loss(tmap_aa, costm, reg)
        c -= 0.5 * loss(tmap_bb, costm, reg)
        return c
    else:
        tmap = comp(a, b, costm, reg)
        return loss(tmap, a, b, costm, reg)


def loss_unbalanced(a, b, costm, reg, reg1, reg2, sink=True, single=True):
    def loss_single(t, pa, pb, m, r, r1, r2):
        c = np.sum(t * (m + r * np.log(t)))
        c += r1 * kl_div(np.sum(t, axis=1), pa) + r2 * kl_div(np.sum(t, axis=0), pb)
        return c
    def loss_all(t_all, pa_all, pb_all, m, r, r1, r2):
        return np.array([loss_single(i, pa, pb, m, r, r1, r2) for i, pa, pb in zip(t_all, pa_all.transpose(), pb_all.transpose())])
    comp = ot_unbalanced if single else ot_unbalanced_all
    loss = loss_single if single else loss_all
    if sink:
        tmap_ab = comp(a, b, costm, reg, reg1, reg2)
        tmap_ba = comp(b, a, costm, reg, reg2, reg2)
        tmap_aa = comp(a, a, costm, reg, reg1, reg1)
        tmap_bb = comp(b, b, costm, reg, reg2, reg2)
        c = loss(tmap_ab, a, b, costm, reg, reg1, reg2) + loss(tmap_ba, b, a, costm, reg, reg2, reg1)
        c -= 1 * loss(tmap_aa, a, a, costm, reg, reg1, reg2)
        c -= 1 * loss(tmap_bb, b, b, costm, reg, reg1, reg2)
        return c
    else:
        tmap_ab = comp(a, b, costm, reg, reg1, reg2)
        tmap_ba = comp(b, a, costm, reg, reg2, reg2)
        return loss(tmap_ab, a, b, costm, reg, reg1, reg2) + loss(tmap_ba, b, a, costm, reg, reg2, reg1)


def loss_unbalanced_partial(a, b, costm, reg, reg1, reg2, sink=True):
    def loss(t, pa, pb, m, r, r1, r2):
        c = np.sum(t * (m + r * np.log(t)))
        c += r1 * kl_div(np.sum(t, axis=1), pa) + r2 * kl_div(np.sum(t, axis=0), pb)
        return c
    def get_sub(m, ia, ib):
        res = m[ia, :]
        return res[:, ib]
    d = costm.shape[0]
    ind_a = np.arange(d)[a > 0]
    ind_b = np.arange(d)[b > 0]
    a_sub = a[a > 0]
    b_sub = b[b > 0]
    if sink:
        tmap_ab = ot_unbalanced(a_sub, b_sub, get_sub(costm, ind_a, ind_b), reg, reg1, reg2)
        tmap_ba = ot_unbalanced(b_sub, a_sub, get_sub(costm, ind_b, ind_a), reg, reg2, reg2)
        tmap_aa = ot_unbalanced(a_sub, a_sub, get_sub(costm, ind_a, ind_a), reg, reg1, reg1)
        tmap_bb = ot_unbalanced(b_sub, b_sub, get_sub(costm, ind_b, ind_b), reg, reg2, reg2)
        c = loss(tmap_ab, a_sub, b_sub, get_sub(costm, ind_a, ind_b), reg, reg1, reg2) + loss(tmap_ba, b_sub, a_sub, get_sub(costm, ind_b, ind_a), reg, reg2, reg1)
        c -= 1 * loss(tmap_aa, a_sub, a_sub, get_sub(costm, ind_a, ind_a), reg, reg1, reg2)
        c -= 1 * loss(tmap_bb, b_sub, b_sub, get_sub(costm, ind_b, ind_b), reg, reg1, reg2)
        return c
    else:
        tmap_ab = ot_balanced(a_sub, b_sub, get_sub(costm, ind_a, ind_b), reg, reg1, reg2)
        tmap_ba = ot_balanced(b_sub, a_sub, get_sub(costm, ind_b, ind_a), reg, reg2, reg2)
        return loss(tmap_ab, a_sub, b_sub, get_sub(costm, ind_a, ind_b), reg, reg1, reg2) + loss(tmap_ba, b_sub, a_sub, get_sub(costm, ind_b, ind_a), reg, reg2, reg1)
    

def loss_unbalanced_local(probs, costm, reg, reg1, reg2, sink=True, win_size=None, weight=None, partial=False):
    T = probs.shape[0] - 1
    cost_win = np.zeros((T + 1, T + 1))
    for t in range(T + 1):
        lower = np.min([T, t + 1])
        upper = np.min([T, t + 2 * win_size - 1])
        if lower != upper:
            for j in range(lower, upper + 1):
                if partial:
                    cost_win[t, j] = loss_unbalanced_partial(probs[t, ], probs[j, ], costm, reg, reg1, reg2, sink=sink)
                else:    
                    cost_win[t, j] = loss_unbalanced(probs[t, ], probs[j, ], costm, reg, reg1, reg2, sink=sink, single=True)
    res = np.zeros(T)
    for t in range(T):
        lower = np.max([0, t - win_size + 1])
        upper = np.min([T, t + win_size])
        cost_temp = []
        weight_temp = []
        for i in range(lower, t + 1):
            for j in range(t + 1, upper + 1):
                cost_temp.append(cost_win[i, j])
                if weight is None:
                    weight_temp.append(1)
                elif weight == 'exp':
                    weight_temp.append(np.exp(-((i - t) ** 2) - ((j - t - 1) ** 2)))
                elif weight == 'exp1':
                    weight_temp.append(np.exp(-(np.abs(i - t)) - (np.abs(j - t - 1))))
                elif weight == 'frac':
                    weight_temp.append(1 / ((1 + np.abs(i - t)) * (1 + np.abs(j - t - 1))))
                elif weight == 'lin':
                    weight_temp.append((win_size - np.abs(i - t)) * (win_size - np.abs(j - t - 1)))
        cost_temp = np.array(cost_temp)
        weight_temp = np.array(weight_temp)
        res[t] = np.mean(cost_temp)
        res[t] = np.sum(cost_temp * weight_temp / weight_temp.sum())
    return res


def multimarg_unbalanced_ot(*margs, costm, reg, reg_phi, coeff, n_iter=10, exp_threshold=10, loss_only=False):
    def prox(reg, reg_kl, value):
        return value * (reg_kl / (reg_kl + reg))
    J = len(margs)
    d = costm.shape[0]
    u = np.ones((J, d))
    alpha_out = np.ones((J, d))
    alpha_in = np.ones((J, d))
    K = []
    for j in range(J):
        K.append(np.exp(-coeff[j] * costm / reg))
    for j in range(J)[::-1]:
        alpha_out[j, ] = K[j] @ u[j, ]
    for n in range(n_iter):
        for j in range(J):
            ind_temp = np.arange(J)[np.arange(J) != j]
            alpha_out_temp = alpha_out[ind_temp, ]
            alpha_in[j, ] = K[j] @ (np.ones(d) * alpha_out_temp.prod(axis=0))
            if np.max(np.abs(np.log(alpha_in[j, ]))) > exp_threshold:
                K[j] = K[j] @ np.diag(np.exp(prox(reg, reg_phi * coeff[j], -np.log(alpha_in[j, ]))))
                alpha_in[j, ] = np.ones(d)
            u[j, ] = np.exp(prox(reg, reg_phi * coeff[j], np.log(margs[j]) - np.log(alpha_in[j, ])))
        for j in range(J)[::-1]:
            alpha_out[j, ] = K[j] @ u[j, ]
            if np.max(np.abs(np.sum(np.log(alpha_out), axis=0))) > exp_threshold:
                for j in range(J):
                    ind_temp = np.arange(J)[np.arange(J) != j]
                    alpha_out_temp = alpha_out[ind_temp, ]
                    K[j] = K[j] @ np.diag(alpha_out_temp.prod(axis=0))
                alpha_out = np.ones((J, d))
    K_mat = np.zeros(tuple(np.repeat(d, J + 1)))
    u_mat = np.zeros(tuple(np.repeat(d, J + 1)))
    cost_mat = np.zeros(tuple(np.repeat(d, J + 1)))
    coords = list(np.ndindex(*np.repeat(d, J + 1)))
    for coord in coords:
        k_temp = 1
        u_temp = 1
        marg_coord = coord[:-1]
        res_coord = coord[-1]
        for j in range(J):
            k_temp *= K[j][marg_coord[j], res_coord]
            u_temp *= u[j, marg_coord[j]]
        K_mat[coord] = k_temp
        u_mat[coord] = u_temp
        cost_mat[coord] = np.sum(np.array(coeff) * costm[marg_coord, res_coord])
    tmap = K_mat * u_mat
    tmap = tmap / tmap.sum()
    res = tmap.sum(axis=tuple(np.arange(J)))
    loss = np.sum(cost_mat * tmap) + reg * np.sum(tmap * np.log(tmap))
    for j in range(J):
        loss += coeff[j] * reg_phi * kl_div(margs[j], res)
    if loss_only:
        return loss
    else:
        return {'tmap': tmap,
                'res': res,
                'loss': loss}
    
    
def multimarg_unbalanced_ot_all(probs, costm, reg, reg_phi, win_size, coeff=None, n_iter=10, exp_threshold=10):
    T = probs.shape[0] - 1
    res = np.zeros(T)
    for t in range(T):
        lower = np.max([0, t - win_size + 1])
        upper = np.min([T, t + win_size])
        margs = [probs[ind] for ind in range(lower, upper + 1)]
        if coeff is None:
            coeff_temp = np.ones(len(margs))
        elif coeff == 'exp':
            coeff_temp = np.array([np.exp(-np.min(np.abs(t_temp - np.array([t, t + 1]))) ** 2) for t_temp in range(lower, upper + 1)])
        elif coeff == 'exp1':
            coeff_temp = np.array([np.exp(-np.min(np.abs(t_temp - np.array([t, t + 1])))) for t_temp in range(lower, upper + 1)])
        elif coeff == 'frac':
            coeff_temp = np.array([1 / (1 + np.min(np.abs(t_temp - np.array([t, t + 1])))) for t_temp in range(lower, upper + 1)])
        elif coeff == 'lin':
            coeff_temp = np.array([win_size - np.min(np.abs(t_temp - np.array([t, t + 1]))) for t_temp in range(lower, upper + 1)])
        coeff_temp = coeff_temp / np.sum(coeff_temp)
        res[t] = multimarg_unbalanced_ot(*margs, costm=costm, reg=reg, reg_phi=reg_phi, coeff=coeff_temp, n_iter=n_iter, exp_threshold=exp_threshold, loss_only=True)
    return res

 
def optimal_lambda(a, b, costm, reg, reg2, reg1_min, reg1_max, step=20):
    def obj_func(t, m):
        return np.sum(t * m)
    reg1_arr = np.linspace(reg1_min, reg1_max, step)
    obj_val = []
    growth = []
    for reg1 in reg1_arr:
        tmap = ot_unbalanced_iter(a, b, costm, reg, reg1, reg2)
        obj_val.append(obj_func(tmap, costm))
        growth.append(estimate_growth1(a, b, costm, reg, reg1, reg2, conv=True))
    opt_ind = np.argmin(np.array(obj_val))
    return {'obj_func': np.array(obj_val),
            'growth_est': np.array(growth),
            'opt_lambda': reg1_arr[opt_ind],
            'opt_index': opt_ind}


def interpolate_weight(a, b, costm, reg, reg1, reg2, h, p0=None, n_conv=1000):
    def get_uv(a, b, costm, reg, reg1, reg2):
        K = np.exp(-costm / reg)
        v = np.repeat(1, len(b))
        for i in range(1000):
            u = (a / (K @ v)) ** (reg1 / (reg + reg1))
            v = (b / (np.transpose(K) @ u)) ** (reg2 / (reg + reg2))
        return u, v
    def loss(t_ab, t_aa, t_bb, pa, pb, costm, reg, reg1, reg2):
        c = np.sum((t_ab - 0.5 * (t_aa + t_bb)) * costm)
        c += get_entropy(t_ab) - 0.5 * (get_entropy(t_aa) + get_entropy(t_bb)) * reg
        c += (kl_div(np.sum(t_ab, axis=1), pa) - 0.5 * (kl_div(np.sum(t_aa, axis=1), pa) + kl_div(np.sum(t_bb, axis=1), pa)) * reg1) 
        c += (kl_div(np.sum(t_ab, axis=0), pb) - 0.5 * (kl_div(np.sum(t_aa, axis=0), pb) + kl_div(np.sum(t_bb, axis=0), pb)) * reg2)
        return c
    def norm(tmap):
        return tmap / np.sum(tmap)
    if p0 is None:
        p = np.repeat(1 / len(a), len(a))
    else:
        p = p0
    K = np.exp(-costm / reg)
    obj = []
    step = 0.03
    for i in range(n_conv):
        uv_ap = get_uv(a, p, costm, reg, reg1, reg2)
        uv_pb = get_uv(p, b, costm, reg, reg1, reg2)
        uv_pp = get_uv(p, p, costm, reg, reg1, reg2)
        uv_aa = get_uv(a, a, costm, reg, reg1, reg2)
        uv_bb = get_uv(b, b, costm, reg, reg1, reg2)
        gradient = h * (-np.exp(uv_ap[1]) + 0.5 * np.exp(-uv_pp[1]) + 0.5)
        gradient += (1 - h) * (-np.exp(uv_pb[0]) + 0.5 * np.exp(-uv_pp[0]) + 0.5)
        t_ap = norm(np.diag(uv_ap[0]) @ K @ np.diag(uv_ap[1]))
        t_aa = norm(np.diag(uv_aa[0]) @ K @ np.diag(uv_aa[1]))
        t_pp = norm(np.diag(uv_pp[0]) @ K @ np.diag(uv_pp[1]))
        t_pb = norm(np.diag(uv_pb[0]) @ K @ np.diag(uv_pb[1]))
        t_bb = norm(np.diag(uv_bb[0]) @ K @ np.diag(uv_bb[1]))
        obj.append(h * loss(t_ap, t_aa, t_pp, a, p, costm, reg, reg1, reg2) +
                   (1 - h) * loss(t_pb, t_pp, t_bb, p, b, costm, reg, reg1, reg2))
        p = p - step * gradient
        p = p / np.sum(p)
    return {'p': p / np.sum(p),
            'obj': obj}

      
# test data 1
##################################################
# pa=np.repeat(1 / 5, 5)
# pb=np.array([1, 2, 3, 4, 5]) / (1 + 2 + 3 + 4 + 5)
# a=np.zeros((5, 100))
# b=np.zeros((5, 100))
# x=np.zeros((5, 100))
# y=np.zeros((5, 100))
# for i in range(100):
#     a_temp = np.random.multinomial(100, pa)
#     b_temp = np.random.multinomial(100, pa)
#     x_temp = np.random.multinomial(100, pa)
#     y_temp = np.random.multinomial(100, pb)
#     a[:,i] = a_temp / np.sum(a_temp)
#     b[:,i] = b_temp / np.sum(b_temp)
#     x[:,i] = x_temp / np.sum(x_temp)
#     y[:,i] = y_temp / np.sum(y_temp)
# costm = np.random.rand(5, 5) * 10
# costm = costm @ costm.transpose()
# np.fill_diagonal(costm, 0)
# reg = 1
# reg1 = 1
# reg2 = 50
# res_bal = sink_loss_balanced_all(a, b, costm, reg)
# res_unbal = sink_loss_unbalanced_all(a, b, costm, reg, reg1, reg2)
# est = interpolate_weight(pa, pb, costm, reg, reg1, reg2, 0.5, p0=pb)
# res = optimal_lambda(pa, pb, costm, reg, reg2, 1, 50, step=100)
# import matplotlib.pyplot as plt
# plt.plot(np.linspace(1, 50, 100), res['obj_func'])
# plt.xlabel('lambda1')
# plt.ylabel('objection function')
##################################################


# test data 2
##################################################

##################################################    





































