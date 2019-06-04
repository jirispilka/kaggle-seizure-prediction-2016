
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.svm import SVC, l1_min_c
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.cross_validation import StratifiedKFold, LeavePLabelOut, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb

from matplotlib import pyplot as plt
from utils import *
from spp_00_load_data import load_features

nsubject = 1
# feat_select = ['stat']
feat_select = ['sp_entropy']
# feat_select = ['mfj']

# filename_tr = 'sp2016_feat_train_{0}_stat_20160915'.format(nsubject)
# Xt, y, aFeatNames, aFiles_tr, plabels, data_q = get_from_10_min(filename_tr)
X, y, aFeatNames, aFiles_tr, plabels, data_q = load_features('train', nsubject, feat_select)

ind = np.sum(np.isnan(X), axis=0) < 50
X = X[:, ind]

print 'Subject: ', nsubject
print 'Original dataset'
print X.shape
print y.shape
print list(aFeatNames)
y = y.ravel()


# clf = xgb.XGBClassifier()
# parameters = {
#     'max_depth': range(2, 31, 5),
#     'gamma': [0, 2],
#     # 'n_estimators': [10, 25, 50, 100],
#     'learning_rate': [0.01, 0.1, 0.2],
#     # 'subsample': [0.5, 1]
#     # 'learning_rate': [0.01, 0.05]
# }

# clf = DecisionTreeClassifier(class_weight='balanced')
# parameters = {
#     # 'max_leaf_nodes': [int(x) for x in np.exp2(range(1, 8))]
#     'max_depth': [1]
# }

# clf_rf = RandomForestClassifier(class_weight='balanced', random_state=2016)
# parameters = {
#     'clf__max_depth': [1] + range(2, 9, 2),
#     'clf__n_estimators': range(5, 51, 5)
# }

# fratio = sum(y == 0) / float(sum(y == 1))
# clf_rf = xgb.XGBClassifier(gamma=0, min_child_weight=1, subsample=1,
#                            colsample_bytree=0.2, learning_rate=0.1, max_depth=1, n_estimators=6,
#                            objective='binary:logistic', seed=2016, silent=1,
#                            scale_pos_weight=fratio)
#
# parameters = {
#     'clf__max_depth': range(1, 4, 1),
#     # 'min_child_weight': range(1, 6, 1),
#     'clf__n_estimators': range(2, 21, 4),
#     # 'gamma': [0,.01, .02, .03],
#     # 'scale_pos_weight': range(1, 10, 1),
#     'clf__colsample_bytree': np.arange(0.1, 0.41, 0.1),
#     # 'subsample': np.arange(0.1, 1.01, .1),
#     # 'learning_rate': [0.01, 0.1, 0.2],
#     # 'reg_alpha': [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1, 50, 100],
#     # 'clf__reg_lambda': [0.1, 1, 10, 25, 50, 100, 150]
# }

# clf = Pipeline([
#     ('ow', OutliersWinsorization()),
#     ('sc', StandardScaler()),
#     ('clf', clf_rf)
# ])

''' Logistic regression '''
clf_lr = LogisticRegression(class_weight='balanced', penalty='l1')
clf = Pipeline([
    ('ow', OutliersWinsorization()),
    ('sc', StandardScaler()),
    ('fs', SelectKBest(f_classif)),
    ('clf', clf_nb)
])
# parameters = {'clf__C': np.exp2(range(-8, 6))}
parameters = {'clf__C': np.arange(0.004, 0.0111, 0.0001)}

''' Naive Bayes '''
# clf_nb = GaussianNB()
#
# clf = Pipeline([
#     ('ow', OutliersWinsorization()),
#     ('sc', StandardScaler()),
#     ('fs', SelectKBest(f_classif)),
#     ('clf', clf_nb)
# ])
# # parameters = {'fs__k': [1, 5, 10, 20, 30, Xt.shape[1]]}  # stat
# # parameters = {'fs__k': range(1, 11, 1)}
# # parameters = {'fs__k': range(10, Xt.shape[1], 10)}
# parameters = {'fs__k': range(1, 20, 1)}

''' SVM with LR selection '''
# clfsvm = SVC(kernel='linear', probability=True, class_weight='balanced')
#
# clf_lr = LogisticRegression(class_weight='balanced', penalty='l1')
# clf_fs = SelectFromModel(estimator=clf_lr)
#
# clf = Pipeline([
#     ('ow', OutliersWinsorization()),
#     ('sc', StandardScaler()),
#     ('fs', clf_fs),
#     ('clf', clfsvm)
# ])
#
# # parameters = {'fs__estimator__C': np.arange(0.007, 0.031, 0.001),
# #               'clf__C': np.exp2(range(-35, 0, 2))
# # }
#
# parameters = {'fs__estimator__C': np.arange(0.01, 0.051, 0.01),
#               'clf__C': np.exp2(range(-24, 1)),
# }
#
# # parameters = {'fs__estimator__C': np.arange(0.005, 0.036, 0.005),
# #               'clf__C': np.exp2(range(-25, 2, 2))
# #               # 'clf__C': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
# #               }

# mask = np.any(np.isnan(Xt), axis=1)
# Xt = Xt[~mask]
# y = y[~mask].ravel()
# plabels = plabels[~mask]

# print 'Removed NaNs'
# print Xt.shape
# print y.shape
# print plabels.shape

print "Parameters:"
for key, val in parameters.iteritems():
    print key, val

# K = [5, 10, 20, 50]
# K = [5, 10]
# K = range(2, 12, 2)
K = [5]
Kinner = 5
R = 11  # repeat cross-validation
verbose_roc = 1

nr = np.sum([R*k for k in K])
index = range(0, nr - 1)

l = parameters.keys()
l.append('folds')
l.append('res_inner')
l.append('res_outer')
l.append('index')
df_results = pd.DataFrame(columns=l, index=index, dtype=float)

nr *= np.prod([len(d) for d in parameters.itervalues()])
index = range(0, nr - 1)

l = parameters.keys()
l.append('repetition')
l.append('folds')
l.append('result')
l.append('index')
df_results_inner = pd.DataFrame(columns=l, index=index, dtype=float)

# df_results.info()
cnt = -1
cnt_inner = -1

C = np.zeros((2, 2))

for k in K:
    for r in range(0, R):

        X_s, y_s, plabels_s, data_q_s = shuffle(X, y, plabels, data_q)
        cv = StratifiedKFoldPLabels(y_s, plabels=plabels_s, k=k)

        for i, (itrn, itst) in enumerate(cv):

            print 'FOLDS: ', k, '| REPEAT: ', r, '| ITER: ', i
            X_train, X_test = X_s[itrn, :], X_s[itst, :]
            y_train, y_test = y_s[itrn], y_s[itst]
            plabels_trn = plabels_s[itrn]
            data_q_trn = data_q_s[itrn]

            # delete nans here
            X_train, y_train, plabels_trn = drop_data_quality_thr(X_train, y_train, plabels_trn, data_q_trn, 10)
            # X_train, y_train, plabels_trn = drop_nan(X_train, y_train, plabels_trn)

            cv_inner = StratifiedKFoldPLabels(y_train, plabels=plabels_trn, k=Kinner)
            grid_search = GridSearchCV(clf, parameters, verbose=0, cv=cv_inner, scoring='roc_auc')
            grid_search.fit(X_train, y_train)

            print grid_search.best_params_

            ''' Inner loop results '''
            print("Inner loop: best parameters set:")
            print("Best score: %0.3f" % grid_search.best_score_)
            best_parameters = grid_search.best_estimator_.get_params()
            for param_name in sorted(parameters.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))

            print 'Grid scores'
            for score in grid_search.grid_scores_:
                print score
                d = score[0].copy()
                res = score[2]
                d['repetition'] = r
                d['folds'] = k

                for rr in res:
                    cnt_inner += 1
                    d['result'] = rr
                    d['index'] = cnt_inner
                    df_results_inner.loc[cnt_inner] = d

            # remember those with nan
            ind = np.any(np.isnan(X_test), axis=1)
            X_test[np.isnan(X_test)] = 0

            # print sum(np.any(np.isnan(X_test), axis=1))
            yhat_prob = grid_search.predict_proba(X_test)
            yhat = grid_search.predict(X_test)

            # classifty to majority class
            yhat_prob[ind] = 0
            yhat[ind] = 0

            auc = metrics.roc_auc_score(y_test, yhat_prob[:, 1])
            # auc = auc if auc > 0.5 else 1-auc

            Ct = metrics.confusion_matrix(y_test, yhat)
            C += Ct

            print 'ifold: ', i, ' - AUC inner:', grid_search.best_score_, 'AUC outer: ', auc
            print Ct
            print C
            cnt += 1
            d = grid_search.best_params_.copy()
            d['folds'] = k
            d['res_inner'] = grid_search.best_score_
            d['res_outer'] = auc
            d['index'] = cnt
            # print df_results.info()
            df_results.loc[cnt] = d

            if verbose_roc:
                fpr, tpr, dummy = metrics.roc_curve(y_train, grid_search.predict_proba(X_train)[:, 1], pos_label=1)
                roc_auc = metrics.auc(fpr, tpr)
                plt.plot(fpr, tpr, 'r', lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

                fpr, tpr, dummy = metrics.roc_curve(y_test, grid_search.predict_proba(X_test)[:, 1], pos_label=1)
                plt.plot(fpr, tpr, 'b', lw=1, label='ROC fold %d (area = %0.2f)' % (i, auc))
                # plt.show()


print df_results.info()
print df_results

print 'Confusion matrix'
print np.round(C / R)

print df_results.median()
print df_results.mad()

import time
stime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
sname = "{0}_patient_{1}_{2}".format(stime, nsubject, "results_DLCV_en_spec_XGB")
df_results.to_pickle(sname)

sname = "{0}_patient_{1}_{2}".format(stime, nsubject, "results_DLCV_INNER_en_spec_XGB")
df_results_inner.to_pickle(sname)

plt.figure()
sns.boxplot(data=df_results[['res_inner', 'res_outer']], notch=True)
sns.swarmplot(data=df_results[['res_inner', 'res_outer']], color=".25")
sns.plt.ylim([0.5, 1])

plt.figure()
sns.distplot(df_results[['res_outer']])

if verbose_roc:
    plt.show()


