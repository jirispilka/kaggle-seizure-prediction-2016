from matplotlib.axis import XTick

import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV, ParameterGrid, _CVScoreTuple
from sklearn.svm import SVC, l1_min_c, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFromModel, SelectKBest, RFECV
from spp_ut_settings import Settings
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
# from sklearn.cross_validation import StratifiedKFold, LeavePLabelOut
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

# from matplotlib.pyplot import *
from utils import compute_roc_auc_score_label_safe, load_features_and_preprocess, rename_groups_random, \
    TimeSeriesSplitGroupSafe, insert_pathol_to_normal_random_keep_order

nsubject = 3
feat_select = ['stat']
# feat_select = ['spectral']
# feat_select = ['sp_entropy']
# feat_select = ['stat', 'spectral']
# feat_select = ['sp_entropy']
# feat_select = ['spectral', 'sp_entropy']
# feat_select = ['spectral', 'sp_entropy']
# feat_select = ['sp_entropy_log']
# feat_select = ['mfj']
# feat_select = ['stat', 'spectral', 'sp_entropy']

settings = Settings()
# print settings

# K = [settings.kfoldCV]
# R = settings.repeatCV

K = 7
R = 20

d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select, settings=settings)
XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                              d_tr[4], d_tr[5], d_tr[6]
XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                         d_ts[4], d_ts[5], d_ts[6]
# ''' FEATURE SELECTION '''

# n_select_feat = 10
# jmi = JMIFeatureSelector(k_feat=n_select_feat)
# jmi.fit(XTRAIN, ytrain)
# XTRAIN = jmi.fit_transform(XTRAIN, ytrain)
#
# aFeatSeleced = list()
# for i in jmi.selected_indicies_:
#     aFeatSeleced.append(aFeatNames[int(i)])
#
# print '\nSELECTED FEATURES\n'
# for i in range(0, n_select_feat):
#     print '{0:6} {1}'.format(jmi.selected_indicies_[i], aFeatSeleced[i])
#
# aFeatNames = aFeatSeleced
# print 'TRAIN :', XTRAIN.shape
# print 'ytrain:', ytrain.shape


''' SVM '''
# # # # cs = l1_min_c(XTRAIN, ytrain, loss='log') * np.logspace(0, 2)
# clf = SVC(kernel='linear', probability=True, class_weight='balanced', degree=3)
# parameters = {
#     # 'penalty': ['l1', 'l2'],
#     # 'C': np.exp2(range(-12, 3, 2)),
#     'C':  0.0001 * np.logspace(0, 3, 10),
#     # 'C': 0.000001 * np.logspace(0, 3, 10),
#     # 'C': [0.001, 0.01, .1, 1],
#     # 'gamma': np.exp2(range(-20, -2, 4)),
#     # 'degree': [2, 3, 4],
#     # 'C': cs
#     # 'jmi__k_feat': [5, 10, 15, 20],
# }

''' Logistic regression '''
w = 'balanced'
clflr = LogisticRegression(class_weight=w, penalty='l2', n_jobs=1)
clf = Pipeline([
    # ('jmi', JMIFeatureSelector(k_feat=50)),
    # ('pca', PCA(n_components=2)),
    ('clf', clflr)
])

# print np.logspace(0, 2.5, 12)
# cs = 0.005 * np.logspace(0, 2.5, 12)
# cs = [0.006, 0.008, 0.010, 0.012, 0.015, 0.02, 0.04, 0.06, 0.12, 0.2, 1, 2]
# cs = [0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.014, 0.016, 0.2, 0.3, 0.5, 1]
# cs = [0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.014, 0.016, 0.2, 0.3, 0.5, 1, 2, 5]
cs = [1e-6, 1e-5, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.02, 0.04, 0.06, 0.12]
# cs = np.arange(0.00001, 0.00201, 0.00005)

parameters = {
    # 'clf__penalty': ['l1', 'l2'],
    'clf__C': cs,
    # 'clf__C': np.exp2(range(-7, 4))  # stat features
    # 'clf__C': np.hstack((np.arange(0.01, 0.031, 0.001))),  # P1
    # 'clf__C': np.hstack((np.arange(0.005, 0.031, 0.001))),  # stat P1
    # 'clf__C': np.hstack((np.arange(0.006, 0.051, 0.002))),  # stat P2, P3
    # 'clf__C': np.hstack((np.arange(0.01, 0.501, 0.02))),  # spectral P1
    # 'pca__n_components': range(2, 20, 8),
}

''' Naive Bayes '''
# clfnb = GaussianNB()
# clf = Pipeline([
#     ('fs', SelectKBest(f_classif)),
#     ('clf', clfnb)
# ])
# # parameters = {'fs__k': range(2, 30, 2)}
# parameters = {'fs__k': [2, 5, 10, 20, 30]}

''' KNN '''
# knn = KNeighborsClassifier()
# clf = Pipeline([
#     # ('jmi', JMIFeatureSelector()),
#     # ('sm', SMOTE(kind='regular')),
#     ('clf', knn)
# ])
# # parameters = {'jmi__k_feat': [10, 20, min(XTRAIN.shape[1], 50), min(XTRAIN.shape[1], 100)],
# #               'clf__n_neighbors': range(20, 200, 20)}
#
# parameters = {'clf__n_neighbors': range(20, 201, 20)}
# # parameters = {'clf__n_neighbors': range(200, 501, 100)}
# # parameters = {'clf__n_neighbors': range(2, 20, 2)}
# # parameters = {'clf__n_neighbors': range(150, 500, 50)}


print "Parameters:"
for key, val in parameters.iteritems():
    print key, val

# nr = np.sum([R * K for K in K])
nr = R
nr *= np.prod([len(d) for d in parameters.itervalues()])
index = range(0, nr - 1)

l = parameters.keys()
l.append('result')
l.append('index')
df_results = pd.DataFrame(columns=l, index=index, dtype=float)

l = parameters.keys()
l.append('result')
l.append('index')
l.append('split')
nr = R * K
nr *= np.prod([len(d) for d in parameters.itervalues()])
index = range(0, nr - 1)
df_split_results = pd.DataFrame(columns=l, index=index, dtype=float)

# df_results.info()
cnt = -1
cnt_split_results = -1

auc_cv_rk = np.zeros((K, R))

for r in range(0, R):
    # cnt_fold = -1
    # cv = cross_validation.StratifiedKFold(y, n_folds=k_feat)

    # X, y, p = shuffle(XTRAIN, ytrain, plabels_tr)
    # p = rename_groups_random(p)
    # skf = GroupKFold(n_splits=k)

    X, y, p = insert_pathol_to_normal_random_keep_order(XTRAIN, ytrain, plabels_tr)
    skf = TimeSeriesSplitGroupSafe(n_splits=K)

    print 'FOLDS:', K, 'REPEAT: ', r

    ''' Grid Search '''
    grid_scores = list()
    param_grid = ParameterGrid(parameters)
    for param in param_grid:

        auc_cv = np.zeros((K, 1))
        # res_selected_cv = np.zeros((Xt.shape[1]), dtype=np.int)
        # res_selected_all[ind_selected_all] = 1
        # print [s for i, s in enumerate(aFeatNames) if ind_selected_all[i] == True]

        for i, (itrn, itst) in enumerate(skf.split(X, y, p)):
            # print itrn, itst
            Xtr, ytr, ptr = X[itrn, :], y[itrn], p[itrn]
            Xts, yts = X[itst, :], y[itst]

            # sm = RandomOverSampler()
            # Xtr, ytr = sm.fit_sample(Xtr, ytr)
            # fs = SelectKBest(f_classif, k=param['fs__k'])

            clf.set_params(**param)

            ''' prediction '''
            clf.fit(Xtr, ytr)
            yhat = clf.predict_proba(Xts)
            # print yhat.shape
            auc = compute_roc_auc_score_label_safe(yts, yhat[:, 1])
            auc_cv[i] = auc
            auc_cv_rk[i, r] = auc

            # print 'iter: ', i, ' auc= ', auc

        # print param, auc_cv.mean(), auc_cv.ravel()
        # print res_selected_cv
        # print [s for i, s in enumerate(aFeatNames) if res_selected_cv[i] > 0]
        # print 'selected features: ', sum(res_selected_cv > 0)
        grid_scores.append(_CVScoreTuple(param, auc_cv.mean(), np.array(auc_cv)))
        # print param, auc_cv.mean(), auc_cv.std()

    # print grid_scores
    best = sorted(grid_scores, key=lambda x: x.mean_validation_score, reverse=True)[0]
    best_params_ = best.parameters
    best_score_ = best.mean_validation_score

    # grid_search = GridSearchCV(pipe, parameters, verbose=1, n_jobs=2, cv=k_feat)
    # grid_search = GridSearchCV(clf, parameters, verbose=1, cv=cv, scoring='roc_auc')
    # grid_search.fit(Xt, yt)

    # print('Best features:', grid_search.best_estimator_.steps[0][1].k_feature_idx_)
    print("Best score: %0.3f" % best_score_)
    print("Best parameters set:")
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_params_[param_name]))

    print 'Grid scores'
    for score in grid_scores:
        print score
        d = score[0].copy()
        cnt += 1
        d['result'] = float(score[1])
        d['index'] = cnt
        df_results.loc[cnt] = d

        # print 'SPLIT RESULTS'
        for j, res in enumerate(score[2]):
            cnt_split_results += 1
            d = score[0].copy()
            d['result'] = float(res)
            d['index'] = cnt_split_results
            d['split'] = j
            df_split_results.loc[cnt_split_results] = d

        # print df_split_results


print auc_cv_rk

s = parameters.keys() + ['split']
print s
print df_split_results.groupby(s, as_index=False).agg({'result': ['median', 'std']})

sns.pointplot(x='split', hue='clf__C', y='result', data=df_split_results, estimator=np.median)

# print df_results.describe()
# print df_results.info()
# print df_results

print df_results.groupby(parameters.keys(), as_index=False).agg({'result': ['median', 'std']})

sns.plt.show()

# import time
# stime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
# sname = "{0}_patient_{1}_results_SLCV_{2}_NB".format(stime, nsubject, 'spectral')
# df_results.to_pickle(sname)
