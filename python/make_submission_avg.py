from matplotlib import pyplot as plt

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, SelectFpr, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GroupKFold

from spp_00_load_data import load_features, load_removed_features
from utils import load_features_and_preprocess, rename_groups_random, probability_calibration, \
    compute_roc_auc_score_label_safe
from spp_ut_settings import Settings

# # # submission v13
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # # # submission v35
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=8)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])

# # # v48
# clf1 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.014)
# clf2 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.026)
# clf3 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.018)

# # # v49
# clf1 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.014)
# clf2 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.03)
# clf3 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.018)

# # # 50 - v35-prumerovani
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=8)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])

# # # v51 -v13-prumerovani
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=2)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])

# # # v52 - v45- prumerovani
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=7)), ('clf', GaussianNB())])

# ### v53
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=5)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=4)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=4)), ('clf', GaussianNB())])

# ### v54
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=18)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=7)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=22)), ('clf', GaussianNB())])

# #### v55
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=5)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=4)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=5)), ('clf', GaussianNB())])

# #### v56
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=4)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=5)), ('clf', GaussianNB())])

# # #### v57
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=30)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])

# # #### v57
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=8)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=4)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=7)), ('clf', GaussianNB())])

# #### v59
# p = [0.5, 0.5]
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=20)), ('clf', GaussianNB(priors=p))])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=4)), ('clf', GaussianNB(priors=p))])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=25)), ('clf', GaussianNB(priors=p))])

# #### v60
# p = [0.9, 0.1]
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=15)), ('clf', GaussianNB(priors=p))])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=4)), ('clf', GaussianNB(priors=p))])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=25)), ('clf', GaussianNB(priors=p))])

# # # # 50 - v35-prumerovani a nastaveni prior
# p = [0.9, 0.1]
# # p = None
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB(priors=p))])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=8)), ('clf', GaussianNB(priors=p))])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=9)), ('clf', GaussianNB(priors=p))])

# c1, c2, c3 = 0.11, 0.15, 0.15  # 63 - stat all p
# c1, c2, c3 = 0.11, 0.15, 0.21  # 64 - stat SLCV
# c1, c2, c3 = 0.085, 0.41, 0.4  # 65 - spectral all p
# c1, c2, c3 = 0.060, 0.14, 0.19  # 66 - spectral SLCV
# c1, c2, c3 = 0.06, 0.04, 0.03  # 67 - spectral entropy all p
# c1, c2, c3 = 0.018, 0.032, 0.022  # 68 - spectral entropy SLCV
# c1, c2, c3 = 0.14, 0.17, 0.18  # 69 - mfj all p
# c1, c2, c3 = 0.024, 0.018, 0.22  # 70 - mfj SLCV
# c1, c2, c3 = 0.02, 0.04, 0.12  # 71 - stat - bez normalizace
# c1, c2, c3 = 0.012, 0.01, 0.01  # 72 - stat - probability calib ranked
# c1, c2, c3 = 0.016, 0.015, 0.015  # 73 - stat - probability calib median centered
# c1, c2, c3 = 0.02, 0.06, 0.02  # 74 - spect - probability calib ranked
# c1, c2, c3 = 1, 0.06, 0.06  # 75 - spect - probability calib median centered
# c1, c2, c3 = 0.02, 0.2, 0.008  # 76 - spectral entropy - probability calib ranked
# c1, c2, c3 = 0.02, 0.008, 0.008  # 77 - spectral entropy - probability median centered
# c1, c2, c3 = 0.01, 0.016, 0.5  # 78 - spectral entropy - probability calib ranked
# c1, c2, c3 = 0.016, 0.01, 0.007  # 79 - spectral entropy - probability median centered
# c1, c2, c3 = 0.014, 0.2, 0.007  # # stat + spec + spectral en calib ranked
# c1, c2, c3 = 0.014, 0.007, 0.007  # # stat + spec + spectral en calib median centerd
# c1, c2, c3 = 0.014, 0.007, 0.007  # # stat + spec + spectral en calib ranked
# c1, c2, c3 = 0.016, 0.2, 0.014  # # 83
# c1, c2, c3 = 0.014, 0.012, 0.008  # # 84
# clf1 = LogisticRegression(class_weight='balanced', penalty='l1', C=c1)
# clf2 = LogisticRegression(class_weight='balanced', penalty='l1', C=c2)
# clf3 = LogisticRegression(class_weight='balanced', penalty='l1', C=c3)

# k1, k2, k3 = 4, 4, 14  # 85
# k1, k2, k3 = 4, 6, 6  # 86
# k1, k2, k3 = 4, 6, 6  # 87 + preset priors
# k1, k2, k3 = 4, 6, 6  # 88 - selekce priznaku mimo CV
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=k1)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=k2)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=k3)), ('clf', GaussianNB())])
# # verze 89
# clf1 = BernoulliNB(alpha=1, binarize=0)
# clf2 = BernoulliNB(alpha=1, binarize=-.25)
# clf3 = BernoulliNB(alpha=1, binarize=0)

# k1, k2, k3 = 4, 6, 6  # 90 - selekce priznaku mimo CV, CV=10
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=k1)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=k2)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=k3)), ('clf', GaussianNB())])

# k1, k2, k3 = 140, 160, 40  # 91
# k1, k2, k3 = 100, 300, 60  # 92
# clf1 = KNeighborsClassifier(n_neighbors=k1)
# clf2 = KNeighborsClassifier(n_neighbors=k2)
# clf3 = KNeighborsClassifier(n_neighbors=k3)

# v93
from submissions_clfs import a_clf_v93, parameters

# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v13 (SCORE = 0.64413 !!!)
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v33-34,35
# sall = ['stat', 'spectral', 'sp_entropy', 'mfj', 'corr']  # v 48
# afeat_sets = [sall, sall, sall] # v48, v49
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v50-v52
# afeat_sets = [['stat'], ['stat'], ['stat']]  # v53-54
# afeat_sets = [['sp_entropy'], ['sp_entropy'], ['sp_entropy']]  # v55
# afeat_sets = [['spectral'], ['spectral'], ['spectral']]  # v56
# afeat_sets = [['mfj'], ['mfj'], ['mfj']]  # v57
# afeat_sets = [['stat'], ['stat'], ['stat']]  # v58
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v33-34,35
# afeat_sets = [['stat'], ['stat'], ['stat']]  # v63, 64
# afeat_sets = [['spectral'], ['spectral'], ['spectral']]  # v65, 66
# afeat_sets = [['sp_entropy'], ['sp_entropy'], ['sp_entropy']]  # v67,68
# afeat_sets = [['mfj'], ['mfj'], ['mfj']]  # v69, 70
# afeat_sets = [['stat'], ['stat'], ['stat']]  # v71, 72, 73
# afeat_sets = [['spectral'], ['spectral'], ['spectral']]  # v74, 75
# afeat_sets = [['sp_entropy'], ['sp_entropy'], ['sp_entropy']]  # v76, 77
# afeat_sets = [['sp_entropy'], ['spectral'], ['spectral']]  # v78, 79
# f1 = ['stat', 'spectral', 'sp_entropy']
# afeat_sets = [f1, f1, f1] # v80, 81, 82
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v83, 84
# afeat_sets = [['sp_entropy'], ['sp_entropy'], ['sp_entropy']]  # v85
# afeat_sets = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v86, 87, 89, 90
# f1 = ['stat', 'spectral'] # 88
# afeat_sets = [f1, f1, f1] # v88
# afeat_sets = [['spectral'], ['spectral'], ['spectral']]  # v91
# f1 = ['stat', 'spectral'] # 92
# afeat_sets = [f1, f1, f1] # 92
f1 = ['stat', 'spectral', 'sp_entropy']  # 93
afeat_sets = [f1, f1, f1]  # 93

# aclf = [clf1, clf2, clf3]
aclf = a_clf_v93
submission = pd.DataFrame()
df_yvalid = pd.DataFrame()
acolors = ['r', 'g', 'b']

settings = Settings()
print settings

prob_calibration_algorithm = 'none'
# prob_calibration_algorithm = 'rank'
# prob_calibration_algorithm = settings.prob_calib_alg
# prob_calibration_algorithm = 'median_centered'

print '------------------------'
print prob_calibration_algorithm
print '------------------------'

K = settings.kfoldCV
R = settings.repeatCV

for i in range(0, 3):

    nsubject = i + 1

    d_tr, d_ts = load_features_and_preprocess(nsubject, afeat_sets[i], settings=settings)
    XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                                  d_tr[4], d_tr[5], d_tr[6]
    XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                                d_ts[4], d_ts[5], d_ts[6]

    # clf = aclf[i]
    clf = aclf[nsubject]
    # print clf.get_params()
    for _, c in clf.estimators:
        assert isinstance(c, Pipeline)
        c.named_steps['gr'].set_feature_names(aFeatNames_tr)

    # XTRAIN = clf.steps[0][1].fit_transform(XTRAIN, ytrain)
    # XTEST = clf.steps[0][1].transform(XTEST)
    # print XTRAIN.shape, XTEST.shape

    yhat_ts_rk = np.zeros((XTEST.shape[0], K * R))
    aauc = np.zeros((K*R, 2))
    yhat_valid_r = np.zeros((XTRAIN.shape[0], R))

    cnt = -1
    for r in range(0, R):

        # cv = StratifiedKFoldPLabels(y=ytrain, plabels=plabels_tr, k=K)
        # yhat_valid_k = np.zeros((XTRAIN.shape[0], 1))
        p = rename_groups_random(plabels_tr)
        cv = GroupKFold(n_splits=K).split(ytrain.copy(), ytrain, p)

        for ifold, (itrn, itst) in enumerate(cv):
            cnt += 1

            xt = XTRAIN[itrn, :]
            yt = ytrain[itrn]

            ''' prediction '''
            clf.fit(xt, yt)

            yhat_tr = clf.predict_proba(xt)
            yhat_valid = clf.predict_proba(XTRAIN[itst, :])
            yhat_ts = clf.predict_proba(XTEST)
            yhat_ts[ind_nan_ts, :] = 0

            yhat_tr = yhat_tr[:, 1]
            yhat_valid = yhat_valid[:, 1]
            yhat_ts = yhat_ts[:, 1]

            # rank probabilities
            yhat_tr = probability_calibration(yhat_tr, prob_calibration_algorithm)
            yhat_valid = probability_calibration(yhat_valid, prob_calibration_algorithm)
            yhat_ts = probability_calibration(yhat_ts, prob_calibration_algorithm)

            auc_tr = metrics.roc_auc_score(yt, yhat_tr)
            auc_valid = compute_roc_auc_score_label_safe(ytrain[itst], yhat_valid)
            aauc[cnt, 0] = auc_tr
            aauc[cnt, 1] = auc_valid
            # print 'Patient {0}, auc_tr={1:3.2}, auc_valid={2:3.2}'.format(nsubject, auc_tr, auc_valid)

            yhat_ts_rk[:, cnt] = yhat_ts
            yhat_valid_r[itst, r] = yhat_valid

            # yhat_valid_k[itst, 0] = yhat_valid

            # plt.figure(nsubject)
            # plt.hist(yhat_ts, range=[0, 1], histtype='step', color='r', linewidth=1,
            #          label='test{0}_{1}fold'.format(nsubject, ifold))
            # plt.title('Patient {0}'.format(nsubject))
            # plt.xlim(0, 1)
            # plt.grid()
            # plt.legend()

        # yhat_valid_r[:, r] = yhat_valid_k[:, 0]

    yhat_ts_avg = yhat_ts_rk.mean(axis=1)
    yhat_valid_avg = yhat_valid_r.mean(axis=1)

    yhat_ts_avg = probability_calibration(yhat_ts_avg, method=prob_calibration_algorithm)
    yhat_valid_avg = probability_calibration(yhat_valid_avg, method=prob_calibration_algorithm)

    df = pd.DataFrame({"yhat_valid": yhat_valid_avg, "ytrain": ytrain})
    df_yvalid = df_yvalid.append(df)

    print 'Patient {0} AUC AVG, auc_tr={1:3.2}, auc_valid={2:3.2}'.format(nsubject, aauc[:, 0].mean(), aauc[:, 1].mean())

    ''' prumer z jednotlivych foldu '''
    plt.figure(10)
    plt.hist(yhat_ts_avg, 50, range=[0, 1], histtype='step', color=acolors[i], linewidth=2,
             label='test-unlabelled_{0}_avg'.format(nsubject))
    plt.xlim(0, 1)
    plt.grid()
    plt.legend()

    plt.figure(70 + nsubject)
    plt.hist(yhat_ts_avg, 50, range=[0, 1], histtype='step', color='k', linewidth=1,
             label='test-unlabelled_{0}_avg'.format(nsubject))
    plt.hist(yhat_valid_avg, 50, range=[0, 1], histtype='step', color='g', linewidth=1, label='valid_{0}_avg'.format(nsubject))
    plt.title('Patient {0} (validation vs unlabelled)'.format(nsubject))
    plt.xlim(0, 1)
    plt.grid()
    plt.legend()

    plt.figure(24)
    fpr, tpr, dummy = metrics.roc_curve(ytrain, yhat_valid_avg, pos_label=1)
    plt.plot(fpr, tpr, color=acolors[i], lw=1, label='patient: %d auc = %0.5f' % (nsubject, aauc[:, 1].mean()))
    plt.title('ROC for all patients')

    df = pd.DataFrame({"File": aFiles_ts, "Class": yhat_ts_avg})
    submission = submission.append(df)

''' vyhodnoceni na cele mnozine '''
yhat_valid = df_yvalid['yhat_valid'].get_values()
ytrain = df_yvalid['ytrain'].get_values()
auc = metrics.roc_auc_score(ytrain, yhat_valid)

plt.figure(24)
fpr, tpr, dummy2 = metrics.roc_curve(ytrain, yhat_valid, pos_label=1)
plt.plot(fpr, tpr, 'k', lw=2, label='auc = %0.5f' % (auc))
plt.title('ROC for all patients')
plt.grid()
plt.legend()

print '----------------------------------'
print '| AUC valid for P1-P2-P3: {0:2.3}  |'.format(auc)
print '----------------------------------'

''' Rozdil oproti minule submission '''
# df_orig = pd.read_csv('../submissions/submission_sp2016_13_20160923_221500_0.64413.csv')
# df_orig = pd.read_csv('../submissions/submission_sp2016_35_20161135_001145_0.64974.csv')
# df_orig = pd.read_csv('../submissions/submission_sp2016_45_20161006_101800_0.64194.csv')
# df_orig = pd.read_csv('../submissions/submission_sp2016_48_20161008_114000_0.60245.csv')
# df_orig = pd.read_csv('../submissions/submission_sp2016_50_20161008_232100_0.64687.csv')
# df_orig = pd.read_csv('submission_sp2016_63_20161013_104500.csv')
# df_orig = pd.read_csv('../submissions/submission_sp2016_65_20161013_105500.csv')
# df_orig = pd.read_csv('../submissions/submission_sp2016_74_20161018_120000.csv')
df_orig = pd.read_csv('submission_sp2016_93_20161025_104000.csv')

df_all = df_orig
df_all['File_new'] = submission['File'].get_values()
df_all['Class_new'] = submission['Class'].get_values()

print 'ROZDIL:'
print np.sum(df_all['Class'].get_values() - submission['Class'].get_values())

sns.pairplot(data=df_all[['Class', 'Class_new']])
# sns.plt.show()

''' submission '''
# print submission
print submission.describe()

submission[["File", "Class"]].to_csv('submission_sp2016_97_20161025_122800.csv', index=False, header=True)

plt.show()