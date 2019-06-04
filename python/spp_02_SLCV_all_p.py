import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import numpy as np
# from joblib import Parallel, delayed
# import multiprocessing
import sys
import time

# from sklearn.grid_search import ParameterGrid, _CVScoreTuple
from sklearn.model_selection import ParameterGrid
from sklearn.grid_search import _CVScoreTuple
from spp_ut_feat_selection import JMIFeatureSelector

from utils import load_features_and_preprocess, create_unique_key_param_patient, \
    repeated_slcv_for_p_save_computation, compute_auc_cv_for_all_p, FeatureSelectGroup, VotingClassifierRank
from spp_ut_settings import Settings

settings = Settings()
print settings

if len(sys.argv) > 1:
    rank = str(sys.argv[1])
else:
    rank = settings.prob_calib_alg

# feat_select = [['stat'], ['stat'], ['stat']]
# feat_select = [['spectral'], ['spectral'], ['spectral']]
# feat_select = [['sp_entropy'], ['sp_entropy'], ['sp_entropy']]
# feat_select = [['mfj'], ['mfj'], ['mfj']]
f1 = ['stat', 'spectral', 'sp_entropy']
feat_select = [f1, f1, f1]
# feat_select = [['sp_entropy'], ['spectral'], ['sp_entropy']]
# feat_select = [['spectral'], ['stat', 'spectral'], ['stat']]

d_data_train = dict()
d_data_test = dict()

prob_calib_alg = settings.prob_calib_alg
# prob_calib_alg = rank

K = settings.kfoldCV
R = settings.repeatCV
Rinner = 5

a_feat_names_for_select_group = dict()
a_fratio = dict()
# a_clf = dict()

for i in range(0, 3):

    nsubject = i + 1

    d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select[i], settings=settings)

    XTRAIN, ytrain, aFeatNames_tr, plabels_tr = d_tr['X'], d_tr['y'], d_tr['aFeatNames'], d_tr['plabels']
    XTEST, ytest, aFeatNames_ts, plabels_ts = d_ts['X'], d_ts['y'], d_ts['aFeatNames'], d_ts['plabels']

    # d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select[i], settings=settings)
    # XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
    #                                                                               d_tr[4], d_tr[5], d_tr[6]
    # XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
    #                                                                             d_ts[4], d_ts[5], d_ts[6]

    a_feat_names_for_select_group[nsubject] = aFeatNames_tr
    a_fratio[nsubject] = sum(ytrain == 0) / float(sum(ytrain == 1))

    print ytest.shape
    print plabels_ts.shape

    T = np.hstack((XTRAIN, ytrain[:, np.newaxis], plabels_tr[:, np.newaxis]))
    names = list(aFeatNames_tr)
    names.append('ytrain')
    names.append('plabels')
    df = pd.DataFrame(data=T, columns=names)
    d_data_train[nsubject] = df

    T = np.hstack((XTEST, ytest[:, np.newaxis], plabels_ts[:, np.newaxis]))
    names = aFeatNames_ts
    names.append('ytest')
    names.append('plabels')
    df = pd.DataFrame(data=T, columns=names)
    d_data_test[nsubject] = df

''' LR '''
# w = 'balanced'
# clflr = LogisticRegression(class_weight=w, penalty='l1', n_jobs=1)
# clf = Pipeline([
#     # ('jmi', JMIFeatureSelector(k_feat=100)),
#     ('clf', clflr)
# ])
# verze stat - BALANCED
# cs = [0.006, 0.008, 0.010, 0.012, 0.015, 0.02, 0.04, 0.06, 0.12, 0.2, 1, 5]
# cs = [0.006, 0.008, 0.010, 0.012, 0.015, 0.02, 0.04, 0.06, 0.12, 0.2, 0.4]
# cs = [0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.014, 0.016, 0.2, 0.3, 0.5]
# cs_sp_entropy = [0.006, 0.007, 0.008, 0.01, 0.012, 0.014, 0.016, 0.1]
# cs_spectral = [0.006, 0.008, 0.01, 0.012, 0.015, 0.2, 0.5, 0.6, 1, 2]
# parameters = {'1_clf__C': cs_sp_entropy,
#               '2_clf__C': cs_spectral,
#               '3_clf__C': cs_sp_entropy,
#               }

# parameters = {'1_clf__C': np.hstack((np.arange(0.015, 0.05, 0.005), np.arange(0.05, 0.101, 0.005))),  # P1
#               '2_clf__C': np.hstack((np.arange(0.006, 0.021, 0.002))),
#               '3_clf__C': np.hstack((np.arange(0.02, 0.071, 0.005))),
#               }

# # verze spectral - BALANCED
# parameters = {'1_clf__C': np.hstack((np.arange(0.006, 0.0251, 0.002))),  # P1
#               '2_clf__C': np.hstack((np.arange(0.02, 0.111, 0.01))),
#               '3_clf__C': np.hstack(([0.004, 0.006, 0.008, 0.01], np.arange(0.012, 0.101, 0.02))),
#               }

''' NB '''
clfnb = GaussianNB()
clf = Pipeline([
    ('fs', SelectKBest(f_classif)),
    ('clf', clfnb)
])
parameters = {'1_fs__k': range(2, 18, 2),  # P1
              '2_fs__k': range(2, 18, 2),  # P1
              '3_fs__k': range(2, 18, 2),  # P1
              }

a_clf = {1: clone(clf), 2: clone(clf), 3: clone(clf)}

# clf = BernoulliNB()
# parameters = {'1__binarize': np.arange(-1.25, 1.251, 0.25),
#               '2__binarize': np.arange(-1.25, 1.251, 0.25),
#               '3__binarize': np.arange(-1.25, 1.251, 0.25)}

# clf = KNeighborsClassifier()
# parameters = {'1__n_neighbors': range(20, 201, 20),
#               '2__n_neighbors': range(20, 201, 20),
#               '3__n_neighbors': range(20, 201, 20)}

''' -------------------------- VOTING CLASSIFIER --------------------------------'''
# clf_nb = Pipeline([
#     ('gr', FeatureSelectGroup()),
#     ('fs', SelectKBest(f_classif)),
#     ('nb', GaussianNB())
# ])
#
# clf_lr = Pipeline([
#     ('gr', FeatureSelectGroup()),
#     ('lr', LogisticRegression(class_weight='balanced', penalty='l1', n_jobs=1))
# ])
#
# clf_xgb = Pipeline([
#     ('gr', FeatureSelectGroup()),
#     ('xgb', xgb.XGBClassifier(max_depth=2, n_estimators=10, gamma=0, colsample_bytree=0.3, subsample=0.9, seed=2016))
# ])
#
# # P1
# est = [('clfa', clone(clf_nb)), ('clfb', clone(clf_lr)), ('clfc', clone(clf_xgb))]
# clf1 = VotingClassifierRank(estimators=est, voting='rank')
#
# # P2
# est = [('clfa', clone(clf_nb)), ('clfb', clone(clf_nb)), ('clfc', clone(clf_lr)), ('clfd', clone(clf_xgb))]
# clf2 = VotingClassifierRank(estimators=est, voting='rank')
#
# # P3
# est = [('clfa', clone(clf_lr)), ('clfb', clone(clf_nb)), ('clfc', clone(clf_lr)), ('clfd', clone(clf_lr))]
# clf3 = VotingClassifierRank(estimators=est, voting='rank')
#
# a_clf[1] = clf1
# a_clf[2] = clf2
# a_clf[3] = clf3
#
# for key, clf in a_clf.iteritems():
#     for _, c in clf.estimators:
#         assert isinstance(c, Pipeline)
#         c.named_steps['gr'].set_feature_names(a_feat_names_for_select_group[key])
#
# parameters = {'1_clfa__fs__k': [10],
#               '1_clfa__gr__group':  ['spectral'],
#               '1_clfb__lr__C':  [0.016],
#               '1_clfb__gr__group': ['sp_entropy'],
#               '1_clfc__xgb__max_depth': [2],
#               '1_clfc__xgb__n_estimators': [10],
#               '1_clfc__xgb__scale_pos_weight': [a_fratio[1]],
#               '1_clfc__gr__group': ['spectral'],
#               '2_clfa__fs__k': [10],
#               '2_clfa__gr__group':  ['stat'],
#               '2_clfb__fs__k':  [10],
#               '2_clfb__gr__group': ['sp_entropy'],
#               '2_clfc__lr__C': [0.006],
#               '2_clfc__gr__group': ['spectral'],
#               '2_clfd__xgb__n_estimators':  [1, 2],
#               '2_clfd__xgb__max_depth': [40, 90],
#               '2_clfd__xgb__scale_pos_weight': [a_fratio[2]],
#               '2_clfd__gr__group': ['sp_entropy'],
#               '3_clfa__lr__C': [0.014],
#               '3_clfa__gr__group': ['stat'],
#               '3_clfb__fs__k': [28],
#               '3_clfb__gr__group': ['stat'],
#               '3_clfc__lr__C': [1],
#               '3_clfc__gr__group': ['stat'],
#               '3_clfd__lr__C': [.5],
#               '3_clfd__gr__group': ['spectral'],
#               }

print "Parameters:"
for param_name in sorted(parameters.keys()):
    print param_name, parameters[param_name]

nr = R*K
nr *= np.prod([len(d) for d in parameters.itervalues()])
index = range(0, nr - 1)

l = parameters.keys()
l.append('repetition')
l.append('folds')
l.append('result')
l.append('result_1')
l.append('result_2')
l.append('result_3')
l.append('index')
df_results = pd.DataFrame(columns=l, index=index, dtype=float)

verbose = False
cnt = -1

for r in range(0, R):  # repaat all

    start_time = time.time()

    # using GroupKFold
    # acv = dict()
    # for key, df in d_data_train.iteritems():
    #     assert isinstance(df, pd.DataFrame)
    #     skf = GroupKFold(n_splits=k)
    #     y = df['ytrain'].get_values().astype(int)
    #     p = df['plabels'].get_values().astype(int)
    #     p = rename_groups_random(p)
    #     df['plabels'] = p
    #
    #     itr, its = list(), list()
    #     for train, test in skf.split(y.copy(), y, p):
    #         itr.append(train)
    #         its.append(test)
    #
    #     cv = zip(itr, its)
    #     acv[key] = cv

    print 'FOLDS:', K, 'REPEAT: ', r

    ''' Grid Search '''
    gs_1 = list()
    gs_2 = list()
    gs_3 = list()
    grid_scores_all = list()

    param_grid = ParameterGrid(parameters)
    print 'Number of param. combinations: ', len(param_grid)

    d_yhat_valid_to_save_computation = dict()  # store predicted yhat
    d_auc_valid_to_save_computation = dict()

    for iparam, param in enumerate(param_grid):

        if verbose:
            print param

        sys.stdout.write("Processed params: %d%%, %d\r" % (100*iparam/len(param_grid), iparam))
        sys.stdout.flush()

        ''' 1) compute predictions for every subject '''
        # print ifold
        # _fit_score_and_predict()

        auc_single_cv = np.zeros((K, 3))

        # paralelni smycka je ve skutecnosti pomalalejsi, protoze dela kopii celeho dataframu!!!
        # num_cores = multiprocessing.cpu_count()
        # results = Parallel(n_jobs=3)(delayed(repeated_slcv_for_p_save_computation)
        #                                     (estimator=clone(clf), n_splits=K, nr_repeat=Rinner, df=df,
        #                                      param=param, key=key,
        #                                      y_hat_valid_precomputed=d_yhat_valid_to_save_computation,
        #                                      auc_valid_precomputed=d_auc_valid_to_save_computation)
        #                              for key, df in d_data_train.iteritems())

        # for key, res in zip(d_data_train.iterkeys(), results):
        #     clf_param_key = create_unique_key_param_patient(key, param)
        #     d_auc_valid_to_save_computation[clf_param_key] = res[0]
        #     d_yhat_valid_to_save_computation[clf_param_key] = res[1]

        for key, df in d_data_train.iteritems():
            out = repeated_slcv_for_p_save_computation(estimator=a_clf[key], n_splits=K, nr_repeat=Rinner, df=df,
                                                       param=param, key=key,
                                                       y_hat_valid_precomputed=d_yhat_valid_to_save_computation,
                                                       auc_valid_precomputed=d_auc_valid_to_save_computation)

            clf_param_key = create_unique_key_param_patient(key, param)
            d_auc_valid_to_save_computation[clf_param_key] = out[0]
            d_yhat_valid_to_save_computation[clf_param_key] = out[1]
            auc_single_cv[:, key - 1] = out[0]

        # print d_yhat_valid_to_save_computation.keys()
        ''' 2) add together prediction from each patient and compute AUC across all patients '''
        auc_all_p = compute_auc_cv_for_all_p(d_data_train, d_yhat_valid_to_save_computation, K, param,
                                             prob_calib_alg)

        if verbose:
            print 'auc results all p'
            print auc_all_p
        # ''' --------------------------------'''
        # ''' stacked generalization '''

        # print param, auc_cv.mean(), auc_cv.ravel()
        gs_1.append(_CVScoreTuple(param, auc_single_cv[:, 0].mean(), np.array(auc_single_cv[:, 0])))
        gs_2.append(_CVScoreTuple(param, auc_single_cv[:, 1].mean(), np.array(auc_single_cv[:, 1])))
        gs_3.append(_CVScoreTuple(param, auc_single_cv[:, 2].mean(), np.array(auc_single_cv[:, 2])))
        grid_scores_all.append(_CVScoreTuple(param, auc_all_p.mean(), np.array(auc_all_p)))

    # print grid_scores
    best = sorted(grid_scores_all, key=lambda x: x.mean_validation_score, reverse=True)[0]
    best_params_ = best.parameters
    best_score_ = best.mean_validation_score

    print("Best score ALL: %0.3f" % best_score_)
    print("Best parameters set for ALL:")
    for param_name in sorted(parameters.keys()):
        print '\t{0}: {1}'.format(param_name, best_params_[param_name])

    # print 'Grid scores ALL'
    for score_all, score1, score2, score3 in zip(grid_scores_all, gs_1, gs_2, gs_3):
        # print score_all
        d = score_all[0].copy()
        res_all = score_all[2]
        res1 = score1[2]
        res2 = score2[2]
        res3 = score3[2]
        d['repetition'] = R
        d['folds'] = K

        for r_all, r1, r2, r3 in zip(res_all, res1, res2, res3):
            d = d.copy()
            cnt += 1
            d['result'] = float(r_all)
            d['result_1'] = float(r1)
            d['result_2'] = float(r2)
            d['result_3'] = float(r3)
            d['index'] = cnt
            df_results.loc[cnt] = d

    print 'Elapsed time:', time.time() - start_time, ' [s]'

# print df_results.describe()
# print df_results.info()
# print df_results

print df_results.groupby(parameters.keys()).median()

import time
stime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
sname = "{0}_all_p_res_SLCV_KNN_stat_spec_{1}_{2}k.res".format(stime, prob_calib_alg, K)
df_results.to_pickle(sname)
