import numpy as np
import h5py
import scipy.sparse
import scipy.io
import os
import re


def read_data(path, format):
    if format == 'h5':
        df = h5py.File(path, 'r')['mm10']
        return scipy.sparse.csc_matrix((df['data'], df['indices'], df['indptr']), shape=df['shape']).transpose().toarray()
    elif format == 'mtx':
        return scipy.io.mmread(path).transpose().toarray()
    else:
        raise Exception('Invalid file type.')

path_dir = 'data\GSE122662_RAW'

file_names = []

pattern = re.compile(r'^.*_D[0-9.]{1,4}_(?!2i).*_C1_.*\.h5$')

for name in os.listdir(path_dir):
    if pattern.match(name) is not None and not re.findall('exp|ctrl', name):
        file_names.append(name)

path_names = [path_dir + '\\' + n for n in file_names]

time_names = []

for name in file_names:
    time_names.append(re.search('D[0-9.]*', name).group())

time_values = [float(name[1:]) for name in time_names]

data = np.array([read_data(p_name, 'h5') for p_name in path_names])