import pandas as pd

from sklearn.utils import shuffle
from sklearn.grid_search import ParameterGrid, _CVScoreTuple
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectFpr
import xgboost as xgb
from utils import *
from spp_ut_settings import Settings
from utils import PreprocessPipeline, drop_data_quality_thr, remove_features_by_name, rename_groups_random, \
    load_features_and_preprocess
import sys
import time

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

if len(sys.argv) > 1:
    nsubject = int(sys.argv[1])
else:
    nsubject = 2

# feat_select = ['stat']
feat_select = ['spectral']
# feat_select = ['sp_entropy']
# feat_select = ['mfj']
# feat_select = ['corr']
# feat_select = ['wav_entropy']
# feat_select = ['sp_entropy', 'stat']
# feat_select = ['stat', 'spectral', 'sp_entropy']
# feat_select = ['sp_entropy', 'spectral']
# feat_select = ['sp_entropy', 'corr']
# feat_select = ['wav_entropy']
# feat_select = ['sp_entropy','corr']
# feat_select = ['stat', 'spectral', 'sp_entropy', 'mfj', 'corr']

settings = Settings()
print settings

K = [settings.kfoldCV]
R = settings.repeatCV

# settings.remove_covariate_shift = False

# XTRAIN, ytrain, aFeatNames, aFiles_tr, plabels, data_q = load_features('train', nsubject, feat_select)
# XTEST, ytest, aFeatNames_ts, dummy4, dummy5, dummy3 = load_features('test', nsubject, feat_select)

d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select, settings=settings)
XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                              d_tr[4], d_tr[5], d_tr[6]
XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                         d_ts[4], d_ts[5], d_ts[6]

'''XGB Boost '''
fratio = sum(ytrain == 0) / float(sum(ytrain == 1))
print 'fratio:', fratio

'''default values '''
# clf = xgb.XGBClassifier(max_depth=3, n_estimators=20, min_child_weight=1, gamma=0,
#                         learning_rate=0.1, colsample_bytree=1, subsample=1,
#                         reg_lambda=1, reg_alpha=1,
#                         objective='binary:logistic', seed=2016, silent=1,
#                         scale_pos_weight=fratio)

clf = xgb.XGBClassifier(max_depth=2, n_estimators=10, min_child_weight=1, gamma=0,
                        learning_rate=0.1, colsample_bytree=0.1, subsample=0.3,
                        reg_lambda=1, reg_alpha=1, nthread=15,
                        objective='binary:logistic', seed=2016, silent=1,
                        scale_pos_weight=fratio)

parameters = {
    # 'max_depth': [1, 2, 3, 4, 5],
    # 'max_depth': range(1, 6, 1),
    'n_estimators': range(5, 31, 5) + [40, 50],
    # 'n_estimators': range(5, 27, 5),
    # 'min_child_weight': range(1, 6, 1),
    # 'gamma': [0,.01, .02, .03, 1 , 2, 50, 100],
    # 'colsample_bytree': np.arange(0.1, 1.01, 0.1),
    # 'colsample_bytree': np.hstack((np.arange(0.01, 0.151, 0.01), np.arange(0.06, 0.21, 0.02))),
    # 'colsample_bytree': np.arange(0.02, 0.51, 0.02),
    # 'subsample': np.arange(0.1, 1.01, 0.1),
    # 'learning_rate': [0.01, 0.1, 0.2],
    # 'reg_alpha': [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1, 50, 100],
    # 'reg_lambda': [1e-4, 0.1, 1, 10, 50, 100]
    # 'max_delta_step': np.arange(0.2, 2, 0.2)
}

# parameters = {
#     'max_depth': range(1, 4, 1),
#     # 'min_child_weight': range(1, 6, 1),
#     'n_estimators': range(2, 21, 4),
#     # 'gamma': [0,.01, .02, .03],
#     # 'scale_pos_weight': range(1, 10, 1),
#     'colsample_bytree': np.arange(0.1, 0.41, 0.1),
#     # 'subsample': np.arange(0.1, 1.01, .1),
#     # 'learning_rate': [0.01, 0.1, 0.2],
#     # 'reg_alpha': [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1, 50, 100],
#     # 'clf__reg_lambda': [0.1, 1, 10, 25, 50, 100, 150]
# }

print "Parameters:"
for key, val in parameters.iteritems():
    print key, val

# K = [2]
# # K = [2]
# R = 41  # repeat cross-validation

nr = np.sum([R*k for k in K])
nr *= np.prod([len(d) for d in parameters.itervalues()])
index = range(0, nr - 1)

l = parameters.keys()
l.append('repetition')
l.append('folds')
l.append('result')
l.append('index')
df_results = pd.DataFrame(columns=l, index=index, dtype=float)

cnt = -1

for k in K:
    for r in range(0, R):

        start_time = time.time()

        Xt, yt, pt = shuffle(XTRAIN, ytrain, plabels_tr)
        pt = rename_groups_random(pt)
        skf = GroupKFold(n_splits=k)

        print 'FOLDS:', k, 'REPEAT: ', r

        ''' Grid Search '''
        grid_scores = list()
        param_grid = ParameterGrid(parameters)
        print 'Number of fits: ', len(param_grid)

        for kk, param in enumerate(param_grid):

            if kk % 100 == 0:
                print kk,

            auc_cv = np.zeros((k, 1))
            # res_selected_cv = np.zeros((Xt.shape[1]), dtype=np.int)
            # res_selected_all[ind_selected_all] = 1
            # print [s for i, s in enumerate(aFeatNames) if ind_selected_all[i] == True]

            for i, (itrn, itst) in enumerate(skf.split(Xt, yt, pt)):
                Xtr, ytr, ptr = Xt[itrn, :], yt[itrn], pt[itrn]
                Xts, yts = Xt[itst, :], yt[itst]

                # fs = SelectFpr(f_classif, alpha=0.05)
                # Xtr = fs.fit_transform(Xtr, ytr)
                # Xts = fs.transform(Xts)
                # # print 'selected features: ', Xtr.shape

                clf.set_params(**param)
                clf.fit(Xtr, ytr, eval_metric='auc')

                ''' prediction '''
                yhat = clf.predict_proba(Xts)
                auc = roc_auc_score(yts, yhat[:, 1])
                # auc_cv[i] = auc if auc > 0.5 else 1 - auc
                auc_cv[i] = auc

                # ind = clf.named_steps['fs'].get_support()
                # print param
                # print 'selected features: ', sum(ind)
                # res_selected_cv[ind] += 1

            # print param, auc_cv.mean(), auc_cv.ravel()
            # print res_selected_cv
            # print [s for i, s in enumerate(aFeatNames) if res_selected_cv[i] > 0]
            # print 'selected features: ', sum(res_selected_cv > 0)
            grid_scores.append(_CVScoreTuple(param, auc_cv.mean(), np.array(auc_cv)))
            # print param, auc_cv.mean(), auc_cv.std()

        # print grid_scores
        best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        best_params_ = best.parameters
        best_score_ = best.mean_validation_score

        print("\nBest score: %0.3f" % best_score_)
        print("Best parameters set:")
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_params_[param_name]))

        print 'Grid scores'
        for score in grid_scores:
            print score
            d = score[0].copy()
            res = score[2]
            d['repetition'] = r
            d['folds'] = k

            for rr in res:
                d = d.copy()
                cnt += 1
                d['result'] = float(rr)
                d['index'] = cnt
                df_results.loc[cnt] = d

        print 'elapsed time: {0} s'.format(time.time() - start_time)


# print df_results.describe()
print df_results.info()
print df_results

print df_results.groupby(parameters.keys()).median()

import time
stime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
sname = "{0}_patient_{1}_results_SLCV_{2}_XGB_test.res".format(stime, nsubject, ''.join(feat_select))
df_results.to_pickle(sname)
