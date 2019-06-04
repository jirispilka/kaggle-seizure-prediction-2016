import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)

import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid

N = 100
X = np.random.randn(N, 50)
y = np.random.randint(0, 2, N)

print X.shape, y.shape

pipe = Pipeline([('fs', SelectFromModel(estimator=LogisticRegression(penalty='l1'))),
                 ('lr', LogisticRegression(penalty='l1'))])

parameters = {'fs__estimator__C': [1, 10], 'lr__C': [100]}
param_grid = ParameterGrid(parameters)

for param in param_grid:

    pipe.set_params(**param)
    d = pipe.get_params()
    print('SELECTED:')
    print('SelectFromModel - LogisticRegression: C', d['fs__estimator__C'])

    pipe.fit(X, y)
    print('ACTUAL:')
    print(pipe.named_steps['fs'].estimator_)

