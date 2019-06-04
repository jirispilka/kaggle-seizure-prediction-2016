import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
# from matplotlib import pyplot as plt
# from joblib import Parallel, delayed
# import multiprocessing
import sys
import time

# from sklearn.grid_search import ParameterGrid, _CVScoreTuple
from sklearn.model_selection import ParameterGrid, GroupKFold
from sklearn.grid_search import _CVScoreTuple
import xgboost as xgb

from utils import load_features_and_preprocess, create_unique_key_param_patient, get_cv_groups_folds_for_all_p, \
    repeated_slcv_for_p_save_computation, compute_auc_cv_for_all_p, get_params_for_patient, \
    get_params_for_specific_clf, probability_calibration, compute_roc_auc_score_label_safe, FeatureSelectGroup, \
    VotingClassifierRank
from spp_ut_settings import Settings

# feat_select = [['stat'], ['stat'], ['stat']]
# feat_select = [['spectral'], ['spectral'], ['spectral']]
# feat_select = [['sp_entropy'], ['sp_entropy'], ['sp_entropy']]
# feat_select = [['mfj'], ['mfj'], ['mfj']]
f1 = ['stat', 'spectral', 'sp_entropy']
feat_select = [f1, f1, f1]
# feat_select = [['sp_entropy'], ['spectral'], ['stat']]
# feat_select = [['sp_entropy'], ['spectral'], ['sp_entropy']]
# feat_select = [['spectral'], ['stat', 'spectral'], ['stat']]

d_data_train = dict()
d_data_outer_test = dict()

settings = Settings()
print settings

K = settings.kfoldDLCV
R = 10  # settings.repeatDLCV

Kinner = settings.kfoldCV
Rinner = 5  # settings.repeatCV
prob_calib_alg = settings.prob_calib_alg
# prob_calib_alg = 'median_centered'

a_feat_names_for_select_group = dict()
a_fratio = dict()
a_clf = dict()

for i in range(0, 3):

    nsubject = i + 1

    d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select[i], settings=settings)
    XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                                  d_tr[4], d_tr[5], d_tr[6]
    XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                                d_ts[4], d_ts[5], d_ts[6]

    a_feat_names_for_select_group[nsubject] = aFeatNames_tr
    a_fratio[nsubject] = sum(ytrain == 0) / float(sum(ytrain == 1))

    T = np.hstack((XTRAIN, ytrain[:, np.newaxis], plabels_tr[:, np.newaxis]))
    names = list(aFeatNames_tr)
    names.append('ytrain')
    names.append('plabels')
    df = pd.DataFrame(data=T, columns=names)
    d_data_train[nsubject] = df

    # T = np.hstack((XTEST, ytest[:, np.newaxis], plabels_ts[:, np.newaxis]))
    # names = aFeatNames_ts
    # names.append('ytest')
    # names.append('plabels')
    # df = pd.DataFrame(data=T, columns=names)
    # d_data_test[nsubject] = df

''' LR '''
# w = 'balanced'
# clflr = LogisticRegression(class_weight=w, penalty='l1', n_jobs=1)
# clf = Pipeline([
#     ('clf', clflr)
# ])

# verze stat - BALANCED
# cs = [0.006, 0.008, 0.010, 0.012, 0.015, 0.02, 0.04, 0.06, 0.12, 0.2, 1, 5]
# cs = [0.006, 0.008, 0.010, 0.012, 0.015, 0.02, 0.04, 0.06, 0.12, 0.2, 0.4]
# cs = [0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.014, 0.016, 0.2, 0.3, 0.5]
# parameters = {'1_clf__C': cs,
#               '2_clf__C': cs,
#               '3_clf__C': cs,
#               }

# cs_sp_entropy = [0.006, 0.007, 0.008, 0.01, 0.012, 0.014, 0.016, 0.1]
# cs_spectral = [0.006, 0.008, 0.01, 0.012, 0.015, 0.2, 0.5, 0.6, 1, 2]
# parameters = {'1_clf__C': cs_sp_entropy,
#               '2_clf__C': cs_spectral,
#               '3_clf__C': cs_sp_entropy,
#               }

# verze spectral - BALANCED
# parameters = {'1_clf__C': np.hstack((np.arange(0.006, 0.0251, 0.002))),  # P1
#               '2_clf__C': np.hstack((np.arange(0.02, 0.111, 0.01))),
#               '3_clf__C': np.hstack(([0.004, 0.006, 0.008, 0.01], np.arange(0.012, 0.101, 0.02))),
#               }

# parameters = {'1_clf__C': np.hstack((np.arange(0.006, 0.121, 0.03))),  # P1
#               '2_clf__C': np.hstack((np.arange(0.02, 0.151, 0.04))),
#               '3_clf__C': np.hstack(([0.004, 0.006, 0.008, 0.05])),
#               }

''' NB '''
# clfnb = GaussianNB()
# clf = Pipeline([
#     ('fs', SelectKBest(f_classif)),
#     ('clf', clfnb)
# ])
# parameters = {'1_fs__k': range(2, 18, 2),  # P1
#               '2_fs__k': range(2, 18, 2),  # P1
#               '3_fs__k': range(2, 18, 2),  # P1
#               }

# clf = BernoulliNB()
# parameters = {'1__binarize': np.arange(-1.25, 1.251, 0.25),
#               '2__binarize': np.arange(-1.25, 1.251, 0.25),
#               '3__binarize': np.arange(-1.25, 1.251, 0.25)}

# clf = KNeighborsClassifier()
# parameters = {'1__n_neighbors': range(100, 161, 20),
#               '2__n_neighbors': range(100, 501, 100),
#               '3__n_neighbors': range(100, 161, 20)}

''' -------------------------- VOTING CLASSIFIER --------------------------------'''
clf_nb = Pipeline([
    ('gr', FeatureSelectGroup()),
    ('fs', SelectKBest(f_classif)),
    ('nb', GaussianNB())
])

clf_lr = Pipeline([
    ('gr', FeatureSelectGroup()),
    ('lr', LogisticRegression(class_weight='balanced', penalty='l1', n_jobs=1))
])

clf_xgb = Pipeline([
    ('gr', FeatureSelectGroup()),
    ('xgb', xgb.XGBClassifier(max_depth=2, n_estimators=10, gamma=0, colsample_bytree=0.3, subsample=0.9, seed=2016))
])

# P1
est = [('clfa', clone(clf_nb)), ('clfb', clone(clf_lr)), ('clfc', clone(clf_xgb))]
clf1 = VotingClassifierRank(estimators=est, voting='rank')

# P2
est = [('clfa', clone(clf_nb)), ('clfb', clone(clf_nb)), ('clfc', clone(clf_lr)), ('clfd', clone(clf_xgb))]
clf2 = VotingClassifierRank(estimators=est, voting='rank')

# P3
est = [('clfa', clone(clf_lr)), ('clfb', clone(clf_nb)), ('clfc', clone(clf_lr)), ('clfd', clone(clf_lr))]
clf3 = VotingClassifierRank(estimators=est, voting='rank')

a_clf[1] = clf1
a_clf[2] = clf2
a_clf[3] = clf3

for key, clftemp in a_clf.iteritems():
    for _, c in clftemp.estimators:
        assert isinstance(c, Pipeline)
        c.named_steps['gr'].set_feature_names(a_feat_names_for_select_group[key])

parameters = {'1_clfa__fs__k': [10],
              '1_clfa__gr__group':  ['spectral'],
              '1_clfb__lr__C':  [0.016],
              '1_clfb__gr__group': ['sp_entropy'],
              '1_clfc__xgb__max_depth': [2],
              '1_clfc__xgb__n_estimators': [10],
              '1_clfc__xgb__scale_pos_weight': [a_fratio[1]],
              '1_clfc__gr__group': ['spectral'],
              '2_clfa__fs__k': [10],
              '2_clfa__gr__group':  ['stat'],
              '2_clfb__fs__k':  [10],
              '2_clfb__gr__group': ['sp_entropy'],
              '2_clfc__lr__C': [0.006],
              '2_clfc__gr__group': ['spectral'],
              '2_clfd__xgb__n_estimators':  [1],
              '2_clfd__xgb__max_depth': [90],
              '2_clfd__xgb__scale_pos_weight': [a_fratio[2]],
              '2_clfd__gr__group': ['sp_entropy'],
              '3_clfa__lr__C': [0.014],
              '3_clfa__gr__group': ['stat'],
              '3_clfb__fs__k': [28],
              '3_clfb__gr__group': ['stat'],
              '3_clfc__lr__C': [1],
              '3_clfc__gr__group': ['stat'],
              '3_clfd__lr__C': [.5],
              '3_clfd__gr__group': ['spectral'],
              }

print "Parameters:"
for param_name in sorted(parameters.keys()):
    print param_name, parameters[param_name]

verbose_roc = 0

nr = R*K
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
l.append('result_1')
l.append('result_2')
l.append('result_3')
l.append('index')
df_results_inner = pd.DataFrame(columns=l, index=index, dtype=float)

cnt = -1
cnt_inner = -1

verbose = False

auc_all_p_cv_r = np.zeros((K, R))

for r in range(0, R):  # repeat double-loop CV

    start_time = time.time()
    CV_all_p = get_cv_groups_folds_for_all_p(d_data_train, K)

    for ifold in range(0, K):

        print '\n --- REPEAT: {0}/{1} | FOLDS: {2}/{3} | '.format(r, R, ifold, K)
        # split data to train/test
        # train data = inner cross-validation loop
        d_data_inner_cv = dict()
        d_data_outer_test = dict()
        for key, df in d_data_train.iteritems():
            cv = CV_all_p[key]
            itrn = cv[ifold][0]
            itst = cv[ifold][1]
            # print len(itrn), len(itst)
            d_data_inner_cv[key] = df.iloc[itrn].copy()
            d_data_outer_test[key] = df.iloc[itst].copy()

        ''' ---------------------------------- '''
        ''' INNER CROSS-VALIDATION Grid Search '''
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

            sys.stdout.write("Inner loop processed params: %d%%, %d\r" % (100 * iparam / len(param_grid), iparam))
            sys.stdout.flush()

            ''' 1) compute predictions for every subject '''
            auc_single_cv = np.zeros((Kinner, 3))
            for key, df in d_data_train.iteritems():
                out = repeated_slcv_for_p_save_computation(estimator=clone(a_clf[key]), n_splits=Kinner, nr_repeat=Rinner,
                                                           df=df, param=param, key=key,
                                                           y_hat_valid_precomputed=d_yhat_valid_to_save_computation,
                                                           auc_valid_precomputed=d_auc_valid_to_save_computation)

                clf_param_key = create_unique_key_param_patient(key, param)
                d_auc_valid_to_save_computation[clf_param_key] = out[0]
                d_yhat_valid_to_save_computation[clf_param_key] = out[1]
                auc_single_cv[:, key-1] = out[0]

            ''' 2) add together prediction from each patient and compute AUC across all patients '''
            auc_all_p = compute_auc_cv_for_all_p(d_data_train, d_yhat_valid_to_save_computation,
                                                 Kinner, param, prob_calib_alg)

            gs_1.append(_CVScoreTuple(param, auc_single_cv[:, 0].mean(), np.array(auc_single_cv[:, 0])))
            gs_2.append(_CVScoreTuple(param, auc_single_cv[:, 1].mean(), np.array(auc_single_cv[:, 1])))
            gs_3.append(_CVScoreTuple(param, auc_single_cv[:, 2].mean(), np.array(auc_single_cv[:, 2])))
            grid_scores_all.append(_CVScoreTuple(param, auc_all_p.mean(), np.array(auc_all_p)))

        best = sorted(grid_scores_all, key=lambda x: x.mean_validation_score, reverse=True)[0]
        best_params_ = best.parameters
        best_score_ = best.mean_validation_score

        print("Inner loop: best parameters set:")
        print("Best AUC for all p: %0.3f" % best_score_)
        print("Best parameters all p:")
        for param_name in sorted(parameters.keys()):
            print '\t{0}: {1}'.format(param_name, best_params_[param_name])
            # print("\t%s: %2.5f" % (param_name, best_params_[param_name]))

        print 'Best AUC for single patient :'
        tmp = [gs_1, gs_2, gs_3]
        for i, gs in enumerate(tmp):
            best_single = sorted(gs, key=lambda x: x.mean_validation_score, reverse=True)[0]
            print("\t%d: %2.3f" % (i+1, best_single.mean_validation_score))

        # print 'Grid scores ALL'
        for score_all, score1, score2, score3 in zip(grid_scores_all, gs_1, gs_2, gs_3):
            # print score_all
            d = score_all[0].copy()
            res_all = score_all[2]
            res1 = score1[2]
            res2 = score2[2]
            res3 = score3[2]
            d['repetition'] = Rinner
            d['folds'] = Kinner

            for r_all, r1, r2, r3 in zip(res_all, res1, res2, res3):
                d = d.copy()
                cnt_inner += 1
                d['result'] = float(r_all)
                d['result_1'] = float(r1)
                d['result_2'] = float(r2)
                d['result_3'] = float(r3)
                d['index'] = cnt_inner
                df_results_inner.loc[cnt_inner] = d

        ''' ---------------------------------- '''
        ''' PREDICTION ON THE TEST DATA '''
        # predict on the test data
        # fit the clfs to trainning data

        yhat_all_p = 0
        ytrue_all_p = 0

        # auc_single_p_trn = np.zeros((3, 1))
        auc_single_p = np.zeros((3, 1))
        for key, df in d_data_inner_cv.iteritems():

            afeatnames = [col for col in df.columns if col not in ['ytrain', 'plabels']]
            XTRAIN = df[afeatnames].get_values()
            ytrain = df['ytrain'].get_values()

            df_test = d_data_outer_test[key]
            afeatnames = [col for col in df_test.columns if col not in ['ytrain', 'plabels']]
            XTEST = df_test[afeatnames].get_values()
            ytest = df_test['ytrain'].get_values()

            clf = a_clf[key]

            param_for_patient = get_params_for_patient(key, best_params_)
            param_for_clf = get_params_for_specific_clf(param_for_patient, clf.get_params())
            # print param_for_clf
            clf.set_params(**param_for_clf)

            clf.fit(XTRAIN, ytrain)
            # prediction - train
            # y_hat = clf.predict_proba(XTRAIN)
            # y_hat = probability_calibration(y_hat, prob_calib_alg)
            # auc_single_p_trn[key - 1, 0] = compute_roc_auc_score_label_safe(ytrain, y_hat[:, 1])

            y_hat = clf.predict_proba(XTEST)
            y_hat = probability_calibration(y_hat[:, 1], prob_calib_alg)

            auc = compute_roc_auc_score_label_safe(ytest, y_hat)
            auc_single_p[key - 1, 0] = auc

            yhat_all_p = y_hat if key == 1 else np.hstack((yhat_all_p, y_hat))
            ytrue_all_p = ytest if key == 1 else np.hstack((ytrue_all_p, ytest))

        auc = compute_roc_auc_score_label_safe(ytrue_all_p, yhat_all_p)
        auc_all_p_cv_r[ifold, r] = auc

        # print 'AUC single patient trn: ', auc_single_p_trn.ravel().T
        print 'AUC single patient: ', auc_single_p.ravel().T
        print 'AUC all p: inner: {0:3.3}  outer: {1:3.3}'.format(best_score_, auc)

        cnt += 1
        d = best_params_.copy()
        d['folds'] = K
        d['res_inner'] = best_score_
        d['res_outer'] = auc
        d['index'] = cnt
        df_results.loc[cnt] = d

        # if verbose_roc:
        #     fpr, tpr, dummy = roc_curve(ytrue_all_p, yhat_all_p, pos_label=1)
        #     plt.plot(fpr, tpr, 'r', lw=1, label='ROC fold %d (area = %0.2f)' % (ifold, auc))
        #     plt.grid()
        #     plt.show()

    print '\nAUC outer: mean (std): {0:3.3} ({1:3.3})\n'.format(auc_all_p_cv_r[:, r].mean(), auc_all_p_cv_r[:, r].std())
    print 'Elapsed time:', time.time() - start_time, ' [s]'

print df_results.median()

import time
stime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
sname = "{0}_all_p_{1}_{2}.res".format(stime, "results_KNN_DLCV_stat_spec", prob_calib_alg)
df_results.to_pickle(sname)

sname = "{0}_all_p_{1}_{2}.res".format(stime, "results_KNN_DLCV_INNER_stat_spec", prob_calib_alg)
df_results_inner.to_pickle(sname)

# if verbose_roc:
#     plt.show()


