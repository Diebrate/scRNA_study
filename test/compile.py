import numpy as np
import pandas as pd
import pyreadr
from scipy.io import loadmat

M = 100

nus = {'low': 0.1, 'high': 0.25}
etas = {'low': 0.5, 'high': 1}

cp = np.zeros(50)
cp[[9, 19, 29, 39]] = 1

summary = []


def prf(pred):

    precision = (pred @ cp) / np.sum(pred, axis=1)
    precision[np.sum(pred, axis=1) == 0] = 0

    recall =  (pred @ cp) / np.sum(cp)

    f_score = (2 * precision * recall) / (precision + recall)
    f_score[(precision == 0) & (recall == 0)] = 0

    return precision, recall, f_score


for eta in ['low', 'high']:

    summary_eta = []

    for nu in ['low', 'high']:

        file_path = '../results/simulation/' + nu + '_nu_' + eta + '_eta/'

        res_ot = []
        res_ecp = []
        res_mn = []

        for m in range(1, M + 1):
            res_ot.append(np.load(file_path + 'test_ot_id' + str(m) + '.npy', allow_pickle=True))
            res_ecp.append(np.array(pyreadr.read_r(file_path + 'test_ecp_id' + str(m) + '.RDS')[None]))
            res_mn.append(loadmat(file_path + 'test_mn_id' + str(m) + '.mat')['res'])

        res = {'ot': np.vstack(res_ot),
               'ecp': np.vstack(res_ecp),
               'mn': np.vstack(res_mn)}

        perf = {method: {'precision': 0, 'recall': 0, 'f-score': 0} for method in res.keys()}

        for method in res.keys():
            p, r, f = prf(res[method])
            perf[method]['precision'] = p
            perf[method]['recall'] = r
            perf[method]['f-score'] = f

        summary_temp = pd.DataFrame(index=res.keys(), columns=['precision', 'recall', 'f-score'])

        txt = '{}({})'

        for method in res.keys():
            for metric in ['precision', 'recall', 'f-score']:
                summary_temp.loc[method, metric] = txt.format(np.round(np.mean(perf[method][metric]), 3),
                                                              np.round(np.std(perf[method][metric]), 3))

        summary_eta.append(summary_temp)

    summary.append(pd.concat(summary_eta, axis='columns', keys=['nu = ' + str(k) for k in nus.values()]))

summary = pd.concat(summary, axis='rows', keys=['eta = ' + str(k) for k in etas.values()])
