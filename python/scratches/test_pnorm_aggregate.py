import numpy as np
from numpy.linalg import norm

y_hat = np.asarray([0.1, 0.2, 0.25, .8, 0.5, 0.7])
p = np.asarray([1, 1, 1, 2, 2, 2])

# import scipy.stats.mstats
from scipy.stats import hmean
# r = norm(y_hat[p == 1])

up = np.unique(p)
y_agg = np.zeros(up.shape)

for i in range(0, len(up)):
    # y_agg[i] = norm(y_hat[p == up[i]], 1)
    y_agg[i] = hmean(y_hat[p == up[i]])

print y_agg

