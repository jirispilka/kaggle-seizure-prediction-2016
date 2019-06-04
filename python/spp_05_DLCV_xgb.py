"""
DLCV tailored to XGB boost
XGB needs to use eval_metric = 'auc' when xgb.fit is called
"""

import sys
from matplotlib import pyplot as plt

import pandas as pd
import seaborn as sns
import xgboost as xgb
from python.utils_learning import OutliersWinsorization
from sklearn import metrics
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.grid_search import ParameterGrid, _CVScoreTuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from spp_00_load_data import load_features
from utils import *

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

if len(sys.argv) > 1:
    nsubject = int(sys.argv[1])
else:
    nsubject = 2

feat_select = ['sp_entropy']
X, y, aFeatNames, aFiles_tr, plabels, data_q = load_features('train', nsubject, feat_select)

ind = np.sum(np.isnan(X), axis=0) < 50
X = X[:, ind]

print 'Subject: ', nsubject
print 'Original dataset'
print X.shape
print y.shape
print list(aFeatNames)
y = y.ravel()

fratio = sum(y == 0) / float(sum(y == 1))
clf = xgb.XGBClassifier(max_depth=3, n_estimators=22, min_child_weight=1, gamma=0,
                        learning_rate=0.1, colsample_bytree=0.2, subsample=1,
                        reg_lambda=1, reg_alpha=1, nthread=4,
                        objective='binary:logistic', seed=2016, silent=1,
                        scale_pos_weight=fratio)

parameters = {
    'max_depth': [1] + range(2, 5, 1),
    # 'max_depth': range(2, 9, 2),
    # 'n_estimators': range(5, 21, 5) + [30, 40],
    'n_estimators': range(5, 27, 5),
    # 'min_child_weight': range(1, 6, 1),
    # 'gamma': [0,.01, .02, .03, 1 , 2, 50, 100],
    'colsample_bytree': np.arange(0.1, 1.01, 0.1),
    # 'colsample_bytree': np.arange(0.02, 0.51, 0.02),
    'subsample': np.arange(0.1, 1.01, .1),
    # 'learning_rate': [0.01, 0.1, 0.2],
    # 'reg_alpha': [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1, 50, 100],
    # 'reg_lambda': [1e-4, 0.1, 1, 10, 50, 100]
    # 'max_delta_step': np.arange(0.2, 2, 0.2)
}

# parameters = {
#     'max_depth': range(2, 4, 1),
#     # 'min_child_weight': range(1, 6, 1),
#     'n_estimators': range(2, 24, 4),
#     # 'gamma': [0,.01, .02, .03],
#     # 'scale_pos_weight': range(1, 10, 1),
#     'colsample_bytree': np.arange(0.1, 0.41, 0.1),
#     # 'subsample': np.arange(0.1, 1.01, .1),
#     # 'learning_rate': [0.01, 0.1, 0.2],
#     # 'reg_alpha': [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1, 50, 100],
#     # 'clf__reg_lambda': [0.1, 1, 10, 25, 50, 100, 150]
# }

pipe = Pipeline([
    ('ow', OutliersWinsorization()),
    ('sc', StandardScaler()),
    ('fs', SelectFpr(f_classif, alpha=0.05))
])

print "Parameters:"
for key, val in parameters.iteritems():
    print key, val

K = [2, 4, 6]
# K = [5, 10]
# K = [5]
R = 11  # repeat cross-validation
verbose_roc = 0

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

cnt = -1
cnt_inner = -1

C = np.zeros((2, 2))

for k in K:
    for r in range(0, R):

        X_s, y_s, plabels_s, data_q_s = shuffle(X, y, plabels, data_q)
        cv_outer = StratifiedKFoldPLabels(y_s, plabels=plabels_s, k=k)

        ''' OUTER LOOP '''
        for i, (idxtrn, idxtst) in enumerate(cv_outer):

            print 'FOLDS: ', k, '| REPEAT: ', r, '| ITER: ', i
            X_train, X_test = X_s[idxtrn, :], X_s[idxtst, :]
            y_train, y_test = y_s[idxtrn], y_s[idxtst]
            plabels_trn = plabels_s[idxtrn]
            data_q_trn = data_q_s[idxtrn]

            # delete nans here
            X_train, y_train, plabels_trn = drop_data_quality_thr(X_train, y_train, plabels_trn, data_q_trn, 10)
            cv_inner = StratifiedKFoldPLabels(y_train, plabels=plabels_trn, k=k)

            ''' INNER LOOP '''
            grid_scores = list()
            param_grid = ParameterGrid(parameters)
            for param in param_grid:

                auc_cv = np.zeros((len(cv_inner), 1))

                for jj, (itrn, itst) in enumerate(cv_inner):

                    Xtr, ytr, ptr = X_train[itrn, :], y_train[itrn], plabels_trn[itrn]
                    Xts, yts = X_train[itst, :], y_train[itst]

                    Xtr = pipe.fit_transform(Xtr, ytr)
                    Xts = pipe.transform(Xts)
                    # print 'selected features: ', Xtr.shape

                    clf.set_params(**param)
                    clf.fit(Xtr, ytr, eval_metric='auc')

                    ''' prediction '''
                    yhat = clf.predict_proba(Xts)
                    auc = metrics.roc_auc_score(yts, yhat[:, 1])
                    auc_cv[jj] = auc
                    # auc_cv[jj] = auc if auc > 0.5 else 1 - auc

                # print param, auc_cv.mean(), auc_cv.ravel()
                grid_scores.append(_CVScoreTuple(param, auc_cv.mean(), np.array(auc_cv)))

            ''' Inner loop results '''
            print("Inner loop: best parameters set:")
            best = sorted(grid_scores, key=lambda x: x.mean_validation_score,reverse=True)[0]
            best_params_ = best.parameters
            best_score_ = best.mean_validation_score

            print("Best score: %0.3f" % best_score_)
            print("Best parameters set:")
            for param_name in sorted(parameters.keys()):
                print("\t%s: %r" % (param_name, best_params_[param_name]))

            # print 'Grid scores'
            for score in grid_scores:
                # print score
                d = score[0].copy()
                res = score[2]
                d['repetition'] = r
                d['folds'] = k

                for rr in res:
                    d = d.copy()
                    cnt_inner += 1
                    d['result'] = float(rr)
                    d['index'] = cnt_inner
                    df_results_inner.loc[cnt_inner] = d

            # remember those with nan
            ind = np.any(np.isnan(X_test), axis=1)
            X_test[np.isnan(X_test)] = 0

            ''' refit with the best params and apply on the test set '''
            X_train = pipe.fit_transform(X_train, y_train)
            X_test = pipe.transform(X_test)
            clf.set_params(**best_params_)
            clf.fit(X_train, y_train)

            # print sum(np.any(np.isnan(X_test), axis=1))
            yhat_prob = clf.predict_proba(X_test)
            yhat = clf.predict(X_test)

            # classify to majority class
            yhat_prob[ind] = 0
            yhat[ind] = 0

            auc = metrics.roc_auc_score(y_test, yhat_prob[:, 1])
            # auc = auc if auc > 0.5 else 1-auc

            Ct = metrics.confusion_matrix(y_test, yhat)
            C += Ct

            print 'AUC inner:', best_score_, 'AUC outer: ', auc
            print Ct
            print C
            cnt += 1
            d = best_params_.copy()
            d['folds'] = k
            d['res_inner'] = best_score_
            d['res_outer'] = auc
            d['index'] = cnt
            # print df_results.info()
            df_results.loc[cnt] = d

            if verbose_roc:
                fpr, tpr, dummy = metrics.roc_curve(y_train, clf.predict_proba(X_train)[:, 1], pos_label=1)
                roc_auc = metrics.auc(fpr, tpr)
                plt.plot(fpr, tpr, 'r', lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

                fpr, tpr, dummy = metrics.roc_curve(y_test, clf.predict_proba(X_test)[:, 1], pos_label=1)
                plt.plot(fpr, tpr, 'b', lw=1, label='ROC fold %d (area = %0.2f)' % (i, auc))
                plt.show()


print df_results.info()
print df_results

print 'Confusion matrix'
print np.round(C / R)

print df_results.median()
print df_results.mad()

import time
stime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
sname = "{0}_patient_{1}_{2}".format(stime, nsubject, "results_DLCV_en_spec_XGB.res")
df_results.to_pickle(sname)

sname = "{0}_patient_{1}_{2}".format(stime, nsubject, "results_DLCV_INNER_en_spec_XGB.res")
df_results_inner.to_pickle(sname)

plt.figure()
sns.boxplot(data=df_results[['res_inner', 'res_outer']], notch=True)
sns.swarmplot(data=df_results[['res_inner', 'res_outer']], color=".25")
sns.plt.ylim([0.5, 1])

plt.figure()
sns.distplot(df_results[['res_outer']])

if verbose_roc:
    plt.show()


