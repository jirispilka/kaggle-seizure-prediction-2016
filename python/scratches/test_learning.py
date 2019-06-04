
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.grid_search import GridSearchCV, ParameterGrid
from sklearn.svm import SVC, l1_min_c
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_selection import RFE, RFECV, chi2, SelectKBest, SelectPercentile, SelectFromModel
from sklearn.cross_validation import StratifiedKFold, LeavePLabelOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.learning_curve import validation_curve
import xgboost as xgb
from sklearn import neighbors
from mlxtend.classifier import StackingClassifier

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import pandas as pd

from utils import get_from_all_10_min

nsubject = 1

filename_tr = 'features/sp2016_feat_train_{0}_stat_20160909'.format(nsubject)
filename_ts = 'features/sp2016_feat_test_{0}_stat_20160909'.format(nsubject)

XTRAIN, ytrain, aFeatNames, aFiles_tr = get_from_all_10_min(filename_tr)
# X_test, y_test, dummy, aFiles_ts = get_from_10_min(filename_ts)

clf = xgb.XGBClassifier()
parameters = {
    'max_depth': range(2, 31, 5),
    'gamma': [0, 2],
    # 'n_estimators': [10, 25, 50, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    # 'subsample': [0.5, 1]
    # 'learning_rate': [0.01, 0.05]
}

# clf = DecisionTreeClassifier(class_weight='balanced')
# parameters = {
#     # 'max_leaf_nodes': [int(x) for x in np.exp2(range(1, 8))]
#     # 'max_depth': [int(x) for x in np.exp2(range(1, 8))]
#     'max_depth': [1]
# }

print XTRAIN.shape
print ytrain.shape
# print list(aFeatNames)
# print Xt[:,0].shape

mask = np.any(np.isnan(XTRAIN), axis=1)
XTRAIN = XTRAIN[~mask]
ytrain = ytrain[~mask].ravel()

print XTRAIN.shape
print ytrain.shape

# print ind.shape
# print len(ind)
# print sum(ind)
# Xt = Xt[ind, :]
# y = y[ind]

# import sys
# sys.exit()

print "Parameters:"
for key, val in parameters.iteritems():
    print key, val

# K = [5, 10, 20, 50]
K = [10]
R = 11  # repeat cross-validation

nr = np.sum([R*k for k in K])
nr *= np.prod([len(d) for d in parameters.itervalues()])
index = range(0, nr - 1)

l = parameters.keys()
l.append('repetition')
l.append('folds')
l.append('result')
l.append('index')
df_results = pd.DataFrame(columns=l, index=index, dtype=float)

# df_results.info()
cnt = -1

for k in K:
    for r in range(0, R):
        # cnt_fold = -1
        # cv = cross_validation.StratifiedKFold(y, n_folds=k_feat)

        X, y = shuffle(XTRAIN, ytrain)
        # print Xt.shape
        # print y.shape
        print 'FOLDS:', k, 'REPEAT: ', r,

        # rfe = RFECV(estimator=clffs, cv=StratifiedKFold(y, k_feat), scoring='accuracy')
        # rfe.fit(Xt, y)
        # print 'Features: ', sum(rfe.support_)
        # Xt = rfe.transform(Xt)

        # grid_search = GridSearchCV(pipe, parameters, verbose=1, n_jobs=2, cv=k_feat)
        grid_search = GridSearchCV(clf, parameters, verbose=1, cv=k, scoring='roc_auc')
        grid_search.fit(X, y)

        print("Best score: %0.3f" % grid_search.best_score_)

        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        # print 'Grid scores'
        for score in grid_search.grid_scores_:
            # print score
            d = score[0]
            res = score[2]
            d['repetition'] = r
            d['folds'] = k

            for rr in res:
                cnt += 1
                d['result'] = rr
                d['index'] = cnt
                df_results.loc[cnt] = d

# print df_results.describe()
print df_results.info()
print df_results

# ''' Naive Bayes '''
# print df_results['binarize']
# df_results['binarize'] = df_results['binarize'].map({True: 1, False: 0}).astype(int)
print df_results.groupby(parameters.keys()).median()

from matplotlib import pyplot as plt
import seaborn as sns

# """ Random forest """
# plt.figure()
# sns.pointplot(x='max_leaf_nodes', hue='n_estimators', y='result', data=df_results, estimator=np.median)
#
# plt.figure()
# sns.pointplot(x='n_estimators', hue='max_leaf_nodes', y='result', data=df_results, estimator=np.median)
#
# # plt.figure()
# df_results.boxplot(column="result", by=['max_leaf_nodes', 'n_estimators'], notch=True)
# dummy, labels = plt.xticks()
# plt.setp(labels, rotation=90)

# """ Decision Tree """
# plt.figure()
# sns.pointplot(x='max_depth', y='result', data=df_results, estimator=np.median)

# plt.figure()
# sns.pointplot(x='rfe__n_features_to_select', y='result', data=df_results, estimator=np.median)
#
# plt.figure()
# sns.pointplot(x='clf__max_depth', y='result', data=df_results, estimator=np.median)

# import time
# stime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
# sname = "{0}_{1}".format(stime, "results_voting_classifier")
# df_results.to_pickle(sname)
