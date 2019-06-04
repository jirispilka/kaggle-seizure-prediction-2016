import pandas as pd
# from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import sys
import time
from scipy.stats.kde import gaussian_kde
from scipy.stats import entropy

# from sklearn.grid_search import ParameterGrid, _CVScoreTuple
from sklearn.model_selection import ParameterGrid, GroupKFold
from sklearn.grid_search import _CVScoreTuple

from utils import PreprocessPipeline, drop_data_quality_thr, load_features_and_preprocess, remove_features_by_name, \
    get_params_for_patient, get_params_for_specific_clf, drop_nan_single, rename_groups_random, auc_score_mannwhitneyu
from spp_00_load_data import load_features, load_removed_features
from spp_ut_settings import Settings


# if len(sys.argv) > 2:
#     K = [int(sys.argv[1])]
#     R = int(sys.argv[2])
# else:
#     K = [5]
#     R = 20  # repeat cross-validation

# def main():
# feat_select = [['sp_entropy'],
#                ['spectral'],
#                ['sp_entropy']]

# feat_select = [['stat'], ['stat'], ['stat']]
# feat_select = [['spectral'], ['spectral'], ['spectral']]
feat_select = [['sp_entropy'], ['sp_entropy'], ['sp_entropy']]
# feat_select = [['mfj'], ['mfj'], ['mfj']]

# sall = ['stat', 'spectral', 'sp_entropy', 'mfj', 'corr']
# feat_select = [sall, sall, sall]

d_data_train = dict()
d_data_test = dict()

settings = Settings()
print settings

for i in range(0, 3):

    nsubject = i + 1

    K = [settings.kfoldCV]
    R = settings.repeatCVouter

    # XTRAIN, ytrain, aFeatNames, aFiles_tr, plabels, data_q = load_features('train', nsubject, feat_select)
    # XTEST, ytest, aFeatNames_ts, dummy4, dummy5, dummy3 = load_features('test', nsubject, feat_select)

    d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select[i], settings=settings)
    XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                                  d_tr[4], d_tr[5], d_tr[6]
    XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                                d_ts[4], d_ts[5], d_ts[6]

    # XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr = load_features('train', nsubject, feat_select[i])
    # XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts = load_features('test', nsubject, feat_select[i])

    # pp.fit(XTRAIN, XTEST, drop_nan=True)
    #
    # print '####### Subject: ', nsubject
    # print '-- Original dataset'
    # print XTRAIN.shape
    # print ytrain.shape
    #
    # thr = 0
    # XTRAIN, ytrain, plabels_tr = drop_data_quality_thr(XTRAIN, ytrain, plabels_tr, data_q_tr, thr)
    # XTEST, ytest, plabels_ts = drop_data_quality_thr(XTEST, ytest, plabels_ts, data_q_ts, thr)
    #
    # XTRAIN = pp.transform(XTRAIN)
    # XTEST = pp.transform(XTEST)
    #
    # print '-- Transformed and removed NaNs'
    # print XTRAIN.shape
    # print ytrain.shape
    #
    # if REMOVE_COVARIATE_SHIFT:
    #     l_feat_remove = load_removed_features(nsubject, feat_select[i])
    #     XTRAIN, aFeatNames_tr, ind_remove = remove_features_by_name(XTRAIN, aFeatNames_tr, l_feat_remove)
    #     XTEST, aFeatNames_ts, ind_remove = remove_features_by_name(XTEST, aFeatNames_ts, l_feat_remove)
    #
    #     print '-- Removed features with covariate shift: '
    #     print 'TRAIN :', XTRAIN.shape
    #     print 'XTEST :', XTEST.shape

    T = np.hstack((XTRAIN, ytrain[:, np.newaxis], plabels_tr[:, np.newaxis]))
    names = aFeatNames_tr
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

# print d_data_train
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

''' Naive Bayes '''
# # p = [0.5, 0.5]
# p = [0.9, 0.1]
# clfnb = GaussianNB(priors=p)
# clf = Pipeline([
#     ('fs', SelectKBest(f_classif)),
#     ('clf', clfnb)
# ])
# # stat
# parameters = {'1_fs__k': range(2, 10, 1) + range(10, 26, 5),
#               '2_fs__k': range(2, 10, 1) + range(10, 21, 5),
#               '3_fs__k': range(2, 10, 1) + range(10, 26, 5),
#               }

# spectral
# parameters = {'1_fs__k': range(1, 10, 1) + range(10, 30, 4),
#               '2_fs__k': range(1, 10, 1) + range(10, 30, 4),
#               '3_fs__k': range(1, 10, 1) + range(10, 30, 4),
#               }

# spectral entropy
# parameters = {'1_fs__k': range(2, 10, 1) + range(10, 30, 4),
#               '2_fs__k': range(2, 10, 1) + range(10, 30, 4),
#               '3_fs__k': range(2, 10, 1) + range(10, 30, 4),
#               }

# # mfj
# parameters = {'1_fs__k': range(2, 10, 2) + range(10, 32, 4),
#               '2_fs__k': range(2, 10, 2) + range(10, 32, 4),
#               '3_fs__k': range(2, 10, 2) + range(10, 32, 4),
#               }


''' LR '''
# w = 'balanced'
w = None
print 'WEIGHTS: ', w

clflr = LogisticRegression(class_weight=w, penalty='l1', n_jobs=1)
clf = Pipeline([
    ('clf', clflr)
])

# parameters = {'1_clf__C': np.hstack((np.arange(0.007, 0.01, 0.0005), np.arange(0.01, 0.051, 0.005))),  # P1
#               '2_clf__C': np.hstack((np.arange(0.007, 0.02, 0.0005), np.arange(0.02, 0.061, 0.01))),
#               '3_clf__C': np.hstack((np.arange(0.007, 0.02, 0.0005), np.arange(0.02, 0.061, 0.01))),  # P3
#               '1_shift_prob': [-0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15],
#               # '1_shift_prob': [0],
#               '2_shift_prob': [0],
#               '3_shift_prob': [0],
#               }

# verze stat - not balanced
# parameters = {'1_clf__C': np.arange(0.08, 0.221, 0.01),  # P1
#               '2_clf__C': np.hstack((np.arange(0.1, 1.001, 0.05))),
#               '3_clf__C': np.hstack((np.arange(0.05, 0.401, 0.05), [0.21])),
#               }

# verze spectral - not balanced
# parameters = {'1_clf__C': np.hstack((np.arange(0.01, 0.311, 0.025))),  # P1
#               '2_clf__C': np.hstack((np.arange(0.01, 0.501, 0.05), [0.14])),
#               '3_clf__C': np.hstack((np.arange(0.1, 0.3, 0.03), np.arange(0.3, 0.401, 0.1))),
#               }

# verze entropy - not balanced
# parameters = {'1_clf__C': np.hstack(([0.018], np.arange(0.01, 0.101, 0.01))),  # P1
#               '2_clf__C': np.hstack(([0.032], np.arange(0.01, 0.080, 0.01), np.arange(0.08, 0.241, 0.04))),
#               '3_clf__C': np.hstack(([0.022], np.arange(0.01, 0.121, 0.01))),
#               }

# verze mfj - not balanced
# parameters = {'1_clf__C': np.hstack((np.arange(0.01, 0.04, 0.01), np.arange(0.04, 0.3, 0.02))),  # P1
#               '2_clf__C': np.hstack(([0.018], np.arange(0.01, 0.03, 0.005), np.arange(0.03, 0.411, 0.02))),
#               '3_clf__C': np.hstack(([0.22], np.arange(0.10, 0.31, 0.02))),
#               }

# verze spectral - BALANCED
parameters = {'1_clf__C': np.hstack(([0.008], np.arange(0.01, 0.131, 0.005))),  # P1
              '2_clf__C': np.hstack(([0.090], np.arange(0.02, 0.121, 0.005))),
              '3_clf__C': np.hstack(([0.008], np.arange(0.01, 0.0151, 0.005), np.arange(0.015, 0.121, 0.01))),
              }

# 'clf__C': np.hstack((np.arange(0.005, 0.031, 0.001))),  # P1
# 'clf__C': np.hstack((np.arange(0.005, 0.0151, 0.001), np.arange(0.015, 0.101, 0.005))),  # P1

# verze 49
# parameters = {'1_clf__C': np.hstack((np.arange(0.004, 0.0101, 0.0004), np.arange(0.01, 0.031, 0.004))),  # P1
#               '2_clf__C': np.hstack((np.arange(0.005, 0.0101, 0.0005), np.arange(0.01, 0.061, 0.004))),
#               '3_clf__C': np.hstack((np.arange(0.004, 0.0101, 0.0004), np.arange(0.01, 0.031, 0.004))),  # P3
#               }
#
# # # verze SLCV kazdy zvlast
# parameters = {'1_clf__C': np.hstack((np.arange(0.004, 0.0101, 0.0001), np.arange(0.01, 0.021, 0.002))),  # P1
#               '2_clf__C': np.hstack((np.arange(0.003, 0.0101, 0.0002), np.arange(0.01, 0.041, 0.002))),
#               '3_clf__C': np.hstack((np.arange(0.003, 0.0101, 0.0002), np.arange(0.01, 0.031, 0.002))),  # P3
#               }

print "Parameters:"
for key, val in parameters.iteritems():
    print key, val

# K = [3]
# R = 30  # repeat cross-validation

nr = np.sum([R*k for k in K])
nr *= np.prod([len(d) for d in parameters.itervalues()])
index = range(0, nr - 1)

l = parameters.keys()
l.append('repetition')
l.append('folds')
l.append('result')
l.append('result_1')
l.append('result_2')
l.append('result_3')
# l.append('dkl')
# l.append('dkl_1')
# l.append('dkl_2')
# l.append('dkl_3')
l.append('index')
df_results = pd.DataFrame(columns=l, index=index, dtype=float)

# df_results.info()
cnt = -1
verbose = False

for k in K:
    for r in range(0, R):

        start_time = time.time()

        # using StratifiedKFoldPLabels
        # acv = dict()
        # for key, df in d_data_train.iteritems():
        #     assert isinstance(df, pd.DataFrame)
        #     cv = StratifiedKFoldPLabels(y=df['ytrain'].get_values().astype(int),
        #                                 plabels=df['plabels'].get_values().astype(int), k=k)
        #     acv[key] = cv

        # using GroupKFold
        acv = dict()
        for key, df in d_data_train.iteritems():
            assert isinstance(df, pd.DataFrame)
            skf = GroupKFold(n_splits=k)
            y = df['ytrain'].get_values().astype(int)
            p = df['plabels'].get_values().astype(int)
            p = rename_groups_random(p)
            df['plabels'] = p

            itr, its = list(), list()
            for train, test in skf.split(y.copy(), y, p):
                itr.append(train)
                its.append(test)

            cv = zip(itr, its)
            acv[key] = cv

        print 'FOLDS:', k, 'REPEAT: ', r

        ''' Grid Search '''
        gs_1 = list()
        gs_2 = list()
        gs_3 = list()
        grid_scores_all = list()
        dkl = list()

        param_grid = ParameterGrid(parameters)
        print 'Number of param. combinations: ', len(param_grid)

        d_predicted_yhat_to_save_computation = dict()  # store predicted yhat
        d_ytest_to_save_computation = dict()
        d_auc_save_computation = dict()

        for iparam, param in enumerate(param_grid):

            auc_single_cv = np.zeros((len(cv), 3))
            auc_all_cv = np.zeros((len(cv), 1))

            if verbose:
                print param

            sys.stdout.write("Processed params: %d%%, %d\r" % (100*iparam/len(param_grid), iparam))
            # sys.stdout.write("Processed params: %d%% \r" % (iparam))
            sys.stdout.flush()

            # d_yhat_for_dkl_computation = dict()
            # d_yts_for_sanity_check = dict()
            #
            # for ip in range(1, 4):
            #     d_yhat_for_dkl_computation[ip] = np.zeros((len(d_data_train[ip]['ytrain']), 1))
            #     d_yts_for_sanity_check[ip] = np.zeros((len(d_data_train[ip]['ytrain']), 1))

            ''' cross validation '''
            for ifold in range(0, k):

                ''' for every subject '''
                yhat_for_auc_across_patients = 0
                ytest_for_auc_across_patients = 0

                # print ifold
                # _fit_score_and_predict()

                for key, df in d_data_train.iteritems():
                    assert isinstance(df, pd.DataFrame)
                    # print 'ifold: {0}, patient: {1}'.format(i, key)

                    ''' select parameters for this patient only'''
                    param_for_patient = get_params_for_patient(key, param)

                    # print param_for_patient
                    # clf_param_key = str(key) + '_' + '_'.join(param_for_patient)
                    s = '_'.join('{0}_{1}'.format(k, v) for k, v in param_for_patient.items())
                    clf_param_key = str(key) + '_' + str(ifold) + '_fold_' + s

                    if clf_param_key not in d_predicted_yhat_to_save_computation:

                        afeatnames = [col for col in df.columns if col not in ['ytrain', 'plabels']]
                        XTRAIN = df[afeatnames].get_values()
                        ytrain = df['ytrain'].get_values()

                        cv = acv[key]

                        itrn = cv[ifold][0]
                        itst = cv[ifold][1]

                        Xtr, ytr = XTRAIN[itrn, :], ytrain[itrn]
                        Xts, yts = XTRAIN[itst, :], ytrain[itst]

                        # print param_for_patient
                        # print clf.get_params()

                        ''' select parameters for clf'''
                        param_for_clf = get_params_for_specific_clf(param_for_patient, clf.get_params())
                        clf.set_params(**param_for_clf)

                        ''' prediction '''
                        clf.fit(Xtr, ytr)
                        yhat = clf.predict_proba(Xts)
                        yhat = yhat[:, 1]

                        if 'shift_prob' in param_for_patient:
                            yhat = yhat + param_for_patient['shift_prob']
                            yhat[yhat < 0] = 0
                            yhat[yhat > 1] = 1

                        auc = roc_auc_score(yts, yhat)

                        d_predicted_yhat_to_save_computation[clf_param_key] = yhat
                        d_ytest_to_save_computation[clf_param_key] = yts
                        d_auc_save_computation[clf_param_key] = auc

                    else:
                        yhat = d_predicted_yhat_to_save_computation[clf_param_key]
                        yts = d_ytest_to_save_computation[clf_param_key]
                        auc = d_auc_save_computation[clf_param_key]

                    # auc = roc_auc_score(yts, yhat)
                    # auc = auc_score_mannwhitneyu(yts, yhat)
                    auc_single_cv[ifold, key-1] = auc

                    # save the predicted labels
                    # cv = acv[key]
                    # itst = cv[ifold][1]
                    # d_yhat_for_dkl_computation[key][itst, 0] = yhat
                    # d_yts_for_sanity_check[key][itst, 0] = yts

                    # # print auc, auc2
                    # if abs(auc - auc2) > 1e-1:
                    #     print auc, auc2
                    #     raise Exception('Error in auc')
                    # auc_single_cv[i, key] = auc if auc > 0.5 else 1 - auc

                    yhat_for_auc_across_patients = yhat if key == 1 else np.hstack((yhat_for_auc_across_patients, yhat))
                    ytest_for_auc_across_patients = yts if key == 1 else np.hstack((ytest_for_auc_across_patients, yts))

                auc = roc_auc_score(ytest_for_auc_across_patients, yhat_for_auc_across_patients)
                # auc = auc_score_mannwhitneyu(ytest_for_auc_across_patients, yhat_for_auc_across_patients)
                auc_all_cv[ifold] = auc

            # TOHLE HODNE ZPOMALUJE !!!!!
            # ''' compute KL divergence for predicted probabilities and XTEST (unlabelled set)'''
            # kl = dict()
            # for ip in range(1, 4):
            #     assert sum(d_data_train[ip]['ytrain'].ravel() - d_yts_for_sanity_check[ip].ravel()) == 0
            #
            #     param_p = get_params_for_patient(key, param)
            #     param_for_clf = get_params_for_specific_clf(param_p, clf.get_params())
            #     clf.set_params(**param_for_clf)
            #
            #     df = d_data_train[key]
            #     afeatnames = [col for col in df.columns if col not in ['ytrain', 'plabels']]
            #     Xtr = df[afeatnames].get_values()
            #     ytr = df['ytrain']
            #     clf.fit(Xtr, ytr)
            #
            #     df = d_data_test[key]
            #     Xt_unlabelled = df[afeatnames].get_values()
            #     yts_unlabelled = clf.predict_proba(Xt_unlabelled)
            #
            #     # Estimating the pdf and plotting
            #     pdf_tr = gaussian_kde(d_yhat_for_dkl_computation[ip].ravel())
            #     pdf_ts = gaussian_kde(yts_unlabelled[:, 1])
            #     x = np.linspace(0, 1, 100)
            #
            #     en = entropy(pdf_tr(x), pdf_ts(x))
            #     kl[ip] = en

            # print auc_single_cv
            # print auc_all_cv
            if verbose:
                print kl
                print auc_single_cv.mean(axis=0)
                print auc_all_cv.mean()

            # print param, auc_cv.mean(), auc_cv.ravel()
            gs_1.append(_CVScoreTuple(param, auc_single_cv[:, 0].mean(), np.array(auc_single_cv[:, 0])))
            gs_2.append(_CVScoreTuple(param, auc_single_cv[:, 1].mean(), np.array(auc_single_cv[:, 1])))
            gs_3.append(_CVScoreTuple(param, auc_single_cv[:, 2].mean(), np.array(auc_single_cv[:, 2])))
            grid_scores_all.append(_CVScoreTuple(param, auc_all_cv.mean(), np.array(auc_all_cv)))
            # dkl.append(kl)

        # print grid_scores
        best = sorted(grid_scores_all, key=lambda x: x.mean_validation_score, reverse=True)[0]
        best_params_ = best.parameters
        best_score_ = best.mean_validation_score

        # grid_search = GridSearchCV(pipe, parameters, verbose=1, n_jobs=2, cv=k_feat)
        # grid_search = GridSearchCV(clf, parameters, verbose=1, cv=cv, scoring='roc_auc')
        # grid_search.fit(Xt, yt)

        # print('Best features:', grid_search.best_estimator_.steps[0][1].k_feature_idx_)
        print("Best score ALL: %0.3f" % best_score_)
        print("Best parameters set for ALL:")
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_params_[param_name]))

        # print 'Grid scores ALL'
        for score_all, score1, score2, score3 in zip(grid_scores_all, gs_1, gs_2, gs_3):
            # print score_all
            d = score_all[0].copy()
            res_all = score_all[2]
            res1 = score1[2]
            res2 = score2[2]
            res3 = score3[2]
            d['repetition'] = r
            d['folds'] = k
            d['folds'] = k
            # d['dkl'] = np.mean([v for v in kl.itervalues()])
            # d['dkl_1'] = kl[1]
            # d['dkl_2'] = kl[2]
            # d['dkl_3'] = kl[3]

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
print df_results.info()
print df_results

print df_results.groupby(parameters.keys()).median()

import time
stime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
sname = "{0}_all_p_res_SLCV_LR_mfj_not_balanced_{1}k.res".format(stime, K)
df_results.to_pickle(sname)


# if __name__ == '__main__':
#     import cProfile
#     import pstats
#     cProfile.run('main()', 'profile_data')
#     p = pstats.Stats('profile_data')
#     p.sort_stats('time').print_stats(40)
#     p.sort_stats('cumulative').print_stats(40)
