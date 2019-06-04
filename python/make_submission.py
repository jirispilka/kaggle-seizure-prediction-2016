from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, SelectFpr, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spp_00_load_data import load_features
from utils import load_features_and_preprocess, probability_calibration
from spp_ut_settings import Settings

# submission v2
# clf_lr1 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.25)
# clf_lr2 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.25)
# clf_lr3 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.25)

# submission v6
# clf_lr1 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.015625)
# clf_lr2 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.0625)
# clf_lr3 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.03125)

# submission v7
# clf_lr1 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.0625)
# clf_lr2 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.0078125)
# clf_lr3 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.015625)

# # # submission v8
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=48)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=48)), ('clf', GaussianNB())])

# submission v9
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=48)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
# clf3 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.015625)

# # submission v10
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=48)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=4)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=48)), ('clf', GaussianNB())])

# # # submission v11
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=48)), ('clf', GaussianNB())])

# # # submission v12
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=3)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=48)), ('clf', GaussianNB())])

# # # submission v13
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # # submission v14
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=4)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # # submission v15
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=4)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])

# # # # submission v16
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = RandomForestClassifier(class_weight='balanced', max_depth=2, n_estimators=20, random_state=2016)
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])

# # # # submission v17
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = xgb.XGBClassifier(max_depth=1, n_estimators=14, colsample_bytree=0.3, seed=2016)
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # submission v18
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = xgb.XGBClassifier(max_depth=2, n_estimators=6, colsample_bytree=0.2, seed=2016)
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # submission v19
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = xgb.XGBClassifier(max_depth=1, n_estimators=14, colsample_bytree=0.3, seed=2016)
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # # submission v20
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = xgb.XGBClassifier(max_depth=2, n_estimators=6, colsample_bytree=0.2, seed=2016)
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # # submission v21
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # submission v22-v25
# clf1 = xgb.XGBClassifier(max_depth=1, n_estimators=10, colsample_bytree=0.6, subsample=0.5, seed=2016)  # v22
# clf1 = xgb.XGBClassifier(max_depth=1, n_estimators=20, colsample_bytree=0.1, subsample=0.7, seed=2016)  # v23
# clf1 = xgb.XGBClassifier(max_depth=2, n_estimators=10, colsample_bytree=0.1, subsample=0.5, seed=2016)  # v24
# clf1 = xgb.XGBClassifier(max_depth=1, n_estimators=30, colsample_bytree=0.1, subsample=0.7, seed=2016)  # v25
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # # submission v26-27
# # clf1 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])  # 26
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=15)), ('clf', GaussianNB())])  # 27
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # # # submission v28-30
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# # clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])  # v28
# # clf2 = Pipeline([('fs', SelectKBest(f_classif, k=30)), ('clf', GaussianNB())])  # v29
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=8)), ('clf', GaussianNB())])  # v30
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # # submission v31
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # # # submission v32-34
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=8)), ('clf', GaussianNB())])
# # clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])  # v32
# # clf3 = Pipeline([('fs', SelectKBest(f_classif, k=5)), ('clf', GaussianNB())])  # v33
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])  # v34

# # # # submission v35
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=8)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])

# # # v36 ....
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=8)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])

# # # v37
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=8)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=8)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])

# v39
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])

# # v41
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])

# v42 zkouska preuceni JMI
# clf1 = Pipeline([('jmi', JMIFeatureSelector(k_feat=24)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])

# # # v43
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=26)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])

# # v44
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=18)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=8)), ('clf', GaussianNB())])

# # # v45
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=7)), ('clf', GaussianNB())])

# # # v46
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=5)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # # v47
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=5)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=5)), ('clf', GaussianNB())])

# # # v48
# clf1 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.014)
# clf2 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.026)
# clf3 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.018)

# # # # submission v62
# p = [0.9, 0.1]
# # p=None
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB(priors=p))])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=8)), ('clf', GaussianNB(priors=p))])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB(priors=p))])

from submissions_clfs import a_clf_v93, parameters # v93

# aclf = [clf_lr1, clf_lr2, clf_lr3]
# afeat_sets = [['stat'], ['stat'], ['stat']] # v8
# afeat_sets = [['stat'], ['spectral'], ['stat']] # v10
# afeat_sets = [['sp_entropy'], ['stat'], ['stat']] # v11
# afeat_sets = [['sp_entropy'], ['sp_entropy'], ['stat']] # v12
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v13 (SCORE = 0.64413 !!!)
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v14
# afeat_sets = [['sp_entropy'], ['spectral'], ['mfj']]  # v15
# afeat_sets = [['sp_entropy'], ['sp_entropy'], ['mfj']]  # v16
# afeat_sets = [['sp_entropy'], ['sp_entropy'], ['sp_entropy']]  # v17, v18, v19, v20, v21
# afeat_sets = [['sp_entropy'], ['sp_entropy'], ['sp_entropy']]  # v26-27
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v28-29
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v31
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v32
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v33-34,35
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v36-41
# afeat_sets = [['stat', 'spectral', 'sp_entropy', 'mfj', 'corr'], ['spectral'], ['sp_entropy']]  # v42
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v43-47
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v43-47

# sall = ['stat', 'spectral', 'sp_entropy', 'mfj', 'corr'] # v 48
# afeat_sets = [sall, sall, sall] # v48
f1 = ['stat', 'spectral', 'sp_entropy']  # 94
afeat_sets = [f1, f1, f1]  # 94

# aclf = [clf1, clf2, clf3]
aclf = a_clf_v93
normalize_all = False

settings = Settings()
print settings
prob_calibration_algorithm = settings.prob_calib_alg
# prob_calibration_algorithm = None

print '------------------------'
print prob_calibration_algorithm
print '------------------------'

submission = pd.DataFrame()
acolors = ['r', 'g', 'b']

for i in range(0, 3):

    nsubject = i+1

    print '###### PATIENT: {0}'.format(nsubject)

    nsubject = i + 1

    d_tr, d_ts = load_features_and_preprocess(nsubject, afeat_sets[i], settings=settings)
    XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                                  d_tr[4], d_tr[5], d_tr[6]
    XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                                d_ts[4], d_ts[5], d_ts[6]

    clf = aclf[nsubject]
    # print clf.get_params()
    for _, c in clf.estimators:
        assert isinstance(c, Pipeline)
        c.named_steps['gr'].set_feature_names(aFeatNames_tr)

    # if normalize_all is True:
    #     pp = PreprocessPipeline(remove_outliers=True, standardize=True)
    #     pp.fit(XTRAIN, XTEST)

    # XTRAIN, ytrain, dummy4 = drop_data_quality_thr(XTRAIN, ytrain, plabels_tr, data_q_tr, 10)
    # Xtrain, ytrain, dummy4 = drop_nan(Xtrain, ytrain, ytrain.copy())

    # if normalize_all is True:
    #     XTRAIN = pp.transform(XTRAIN)

    # modify for XGB
    # if hasattr(aclf[i], 'colsample_bytree'):
    #     pipe = Pipeline([
    #         ('ow', OutliersWinsorization()),
    #         ('sc', StandardScaler()),
    #         ('fs', SelectFpr(f_classif, alpha=0.05))
    #     ])
    #     XTRAIN = pipe.fit_transform(XTRAIN, ytrain)
    #
    #     clf = aclf[i]
    #     fratio = sum(ytrain == 0) / float(sum(ytrain == 1))
    #     param = {'scale_pos_weight': fratio}
    #     clf.set_params(**param)
    #     clf.fit(XTRAIN, ytrain, eval_metric='auc')
    # else:
    #     clf = Pipeline([
    #         ('ow', OutliersWinsorization()),
    #         ('sc', StandardScaler()),
    #         ('clf', aclf[i])
    #     ])
    #     clf.fit(XTRAIN, ytrain)
    #
    #     # print clf.steps[2][1].steps[1][1].class_prior_

    # clf = aclf[nsubject]
    # print clf

    clf.fit(XTRAIN, ytrain)
    yhat_tr = clf.predict_proba(XTRAIN)
    yhat_tr = yhat_tr[:, 1]
    yhat_tr = probability_calibration(yhat_tr, prob_calibration_algorithm)

    auc_tr = metrics.roc_auc_score(ytrain, yhat_tr)
    print 'Patient {0}, auc tr={1:3.2}'.format(nsubject, auc_tr)

    # ind = np.any(np.isnan(XTEST), axis=1)
    # XTEST[ind] = 0
    # if hasattr(aclf[i], 'colsample_bytree'):
    #     XTEST = pipe.transform(XTEST)
    # else:
    #     if normalize_all is True:
    #         XTEST = pp.transform(XTEST)

    yhat_ts = clf.predict_proba(XTEST)
    yhat_ts = yhat_ts[:, 1]
    # if nsubject == 2 or nsubject == 3:
    #     import warnings
    #     warnings._show_warning('the prediction for p2 and p3 are set to zero', Warning, 'make_submission.py', 1)
    #     ind = range(0, yhat_ts.shape[0])

    yhat_ts[ind_nan_ts] = 0
    yhat_ts = probability_calibration(yhat_ts, prob_calibration_algorithm)

    # import sys
    # sys.exit()

    print 'Recs: {0}, nans in test set: {1}'.format(XTEST.shape[0], sum(ind_nan_ts))

    plt.figure()
    plt.hist(yhat_tr[ytrain == 1], 20, histtype='step', color='r', linewidth=2)
    plt.hist(yhat_tr[ytrain == 0], 20, histtype='step', color='b')
    plt.hist(yhat_ts, 20, histtype='step', color='g')
    plt.title('Patient {0}'.format(nsubject))
    plt.grid()
    plt.legend(['train == 1', 'ytrain == 0', 'test'])

    plt.figure(100)
    plt.hist(yhat_ts, 20, histtype='step', color=acolors[i], linewidth=2, label='test{0}'.format(nsubject))
    plt.grid()
    plt.legend()

    df = pd.DataFrame({"File": aFiles_ts, "Class": yhat_ts})
    submission = submission.append(df)

''' podivat se na rozdil '''
# df_orig = pd.read_csv('../submissions/submission_sp2016_35_20161135_001145_0.64974.csv')
df_orig = pd.read_csv('submission_sp2016_93_20161025_104000.csv')

# print submission

df_all = df_orig
df_all['File_new'] = submission['File'].get_values()
df_all['Class_new'] = submission['Class'].get_values()

print 'ROZDIL:'
print np.sum(df_all['Class'].get_values() - submission['Class'].get_values())

import seaborn as sns
sns.pairplot(data=df_all[['Class', 'Class_new']])
# sns.plt.show()

''' submission '''
# print 'Final submission'
# print submission
print submission.describe()

submission[["File", "Class"]].to_csv('submission_sp2016_94_20161016_114500.csv', index=False, header=True)

plt.show()