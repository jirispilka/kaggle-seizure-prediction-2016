from feast import JMI
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from matplotlib import pyplot as plt
import sys

from utils import *
from spp_00_load_data import load_features

nsubject = 1

# feat_select = ['stat']
# feat_select = ['spectral']
# feat_select = ['sp_entropy']
feat_select = ['mfj']
# feat_select = ['spectral', 'sp_entropy']

XTRAIN, ytrain, aFeatNames, aFiles_tr, plabels, data_q = load_features('train', nsubject, feat_select)
XTEST, ytest, aFeatNames_ts, dummy4, dummy5, dummy3 = load_features('test', nsubject, feat_select)

XTRAIN, ytrain, XTEST, aFeatNames, plabels, data_q = \
    preprocess_pipeline(XTRAIN, ytrain, XTEST, aFeatNames, plabels, data_q, verbose=True)

print 'Original dataset'
print 'TRAIN:', XTRAIN.shape
print 'ytrain', ytrain.shape

thr = 10
XTRAIN, ytrain, plabels = drop_data_quality_thr(XTRAIN, ytrain, plabels, data_q, thr)

print '\nRemoved data quality with treshold: ', thr
print 'TRAIN :', XTRAIN.shape
print 'ytrain:', ytrain.shape
print 'plabels', plabels.shape

# data = load_features('train', 2, feat_select)
# XTRAIN2 = data[0]
#
# d = XTRAIN - XTRAIN2
# d = d.ravel()
# print d

# plt.figure()
# plt.stem(d[0:100])
# plt.show()

# sys.exit(0)

# ind = range(480, 485)
# print ind
# XTRAIN = XTRAIN[:, ind]
# XTEST = XTEST[:, ind]
# aFeatNames = [s for i, s in enumerate(aFeatNames) if i in ind]
# print aFeatNames

# print sum(XTRAIN[:, 0])
# print aFeatNames[150]

# print aFeatNames[490]

# print np.sum(XTRAIN[:,0], axis=0)
# print X.sum(axis=0)

# print clf.steps[1][1].mean_
# print clf.steps[1][1].var_
# print clf.steps[1][1].n_samples_seen_
# print clf.steps[1][1].n_samples_seen_

n_select_feat = 10
selected_indicies = JMI(XTRAIN, ytrain, n_select_feat)

aFeatSeleced = list()
for i in selected_indicies:
    aFeatSeleced.append(aFeatNames[int(i)])

# aFeatSeleced = [s for i, s in enumerate(aFeatNames) if i in selected_indicies]
print selected_indicies
print aFeatSeleced
print 'order alphabetically:', [s for i, s in enumerate(aFeatNames) if i in selected_indicies]
print 'sort indicies + 1 !!!:', np.sort(selected_indicies)+1

# print sum(X[:,370])