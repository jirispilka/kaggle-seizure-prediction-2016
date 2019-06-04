import pandas as pd
from utils import get_from_all_10_min
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

nsubject = 1
idfeat = 6

# nsubject = 2
# idfeat = 42

# nsubject = 3
# idfeat = 12

# filename_tr = 'features/sp2016_feat_train_{0}_stat_20160912'.format(nsubject)
# filename_ts = 'features/sp2016_feat_test_{0}_stat_20160912'.format(nsubject)

filename_tr = 'features/archive/sp2016_feat_train_{0}_stat_20160909'.format(nsubject)
filename_ts = 'features/archive/sp2016_feat_test_{0}_stat_20160909'.format(nsubject)

XTRAIN, ytrain, aFeatNames, aFiles_tr, dummy2 = get_from_all_10_min(filename_tr)
XTEST, ytest, dummy, aFiles_ts, dummy3 = get_from_all_10_min(filename_ts)

mask = np.any(np.isnan(XTRAIN), axis=1)
XTRAIN = XTRAIN[~mask]
ytrain = ytrain[~mask].ravel()

X = XTRAIN[:, idfeat]
scaler = MinMaxScaler()
X = 1/X
X = scaler.fit_transform(X)


fpr, tpr, thresholds = metrics.roc_curve(ytrain, X, pos_label=1)
print metrics.auc(fpr, tpr)

# y_hat = np.random.uniform(-20, 20, size=(X_test.shape[0], 1)).ravel()

X = XTEST[:, idfeat]
X[np.isnan(X)] = 0.01
X = 1/X
X = scaler.fit_transform(X)

print X.shape
y_hat = X


submission = pd.DataFrame({
    "File": aFiles_ts,
    "Class": y_hat
})

print submission

# submission[["File", "Class"]].to_csv('submission_sp2016_01_20160909_152000_{0}.csv'.format(nsubject), index=False, header=True)