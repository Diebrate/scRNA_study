import numpy as np
import pandas as pd
import test_util

use_sys = False
if use_sys:
    import sys
    m = int(sys.argv[1])
else:
    m = 0

data = pd.read_csv('../data/simulation_data/simulation_id' + str(m) + '.csv')
T = int(data.time.max())
d = int(data.type.max()) + 1

centroids = data[['phate1', 'phate2', 'type']].groupby('type').mean().to_numpy()
costm = test_util.get_cost_matrix(centroids, centroids, dim=2)
probs = np.zeros(T + 1, d)
for t in range(T + 1):
    data['type'][data['time'] == t]