import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt
import numpy as np
import sys
import time
from scipy.stats.kde import gaussian_kde
from scipy.stats import entropy

from utils import PreprocessPipeline, drop_data_quality_thr, drop_nan_single, remove_features_by_name, \
    rename_groups_random, load_features_and_preprocess, probability_calibration
from spp_00_load_data import load_features, load_removed_features
from spp_ut_settings import Settings
from discretization import MDLP


# sall = ['stat', 'spectral', 'sp_entropy', 'mfj', 'corr']
# feat_select = [sall, sall, sall]
# feat_select = [['sp_entropy'], ['spectral'], ['sp_entropy']]  # v36-41
# feat_select = [['stat'], ['stat'], ['stat']]  # v36-41
feat_select = [['spectral'], ['spectral'], ['spectral']]  # v36-41
# feat_select = [['sp_entropy'], ['sp_entropy'], ['sp_entropy']]  # v36-41
# feat_select = [['mfj'], ['mfj'], ['mfj']]  # v36-41
# feat_select = [['corr'], ['corr'], ['corr']]  # v36-41
# feat_select = [['sp_entropy'], ['spectral'], ['spectral']]  # v36-41
# f1 = ['stat', 'spectral']
# feat_select = [f1, f1, f1]
# feat_select = [['sp_entropy'], ['spectral'], ['sp_entropy']]
# pp = PreprocessPipeline(remove_outliers=True, standardize=True)

'''balanced'''
w = 'balanced'
# clf1 = LogisticRegression(class_weight=w, penalty='l1', C=0.014)
# clf2 = LogisticRegression(class_weight=w, penalty='l1', C=0.028)
# clf3 = LogisticRegression(class_weight=w, penalty='l1', C=0.012)

# c1, c2, c3 = 0.06, 0.012, 0.012  # # stat opt all p
# c1, c2, c3 = 0.02, 0.06, 0.02  # # spec opt all p
# c1, c2, c3 = 0.014, 0.007, 0.007  # # spec opt all p
# c1, c2, c3 = 0.014, 0.012, 0.008  # # spec opt all p
# clf1 = LogisticRegression(class_weight=w, penalty='l1', C=c1)
# clf2 = LogisticRegression(class_weight=w, penalty='l1', C=c2)
# clf3 = LogisticRegression(class_weight=w, penalty='l1', C=c3)

# stat 2) opt all p
# w = None
# clf1 = LogisticRegression(class_weight=w, penalty='l1', C=0.11)
# clf2 = LogisticRegression(class_weight=w, penalty='l1', C=0.15)
# clf3 = LogisticRegression(class_weight=w, penalty='l1', C=0.15)

# # spec 1) opt SLCV
# w = None
# clf1 = LogisticRegression(class_weight=w, penalty='l1', C=0.06)
# clf2 = LogisticRegression(class_weight=w, penalty='l1', C=0.14)
# clf3 = LogisticRegression(class_weight=w, penalty='l1', C=0.19)

# spec 2) opt all p
# w = None
# clf1 = LogisticRegression(class_weight=w, penalty='l1', C=0.085)
# clf2 = LogisticRegression(class_weight=w, penalty='l1', C=0.41)
# clf3 = LogisticRegression(class_weight=w, penalty='l1', C=0.40)

# spec entropy 1)
# c1, c2, c3 = 0.014, 0.04, 0.01 # opt all p
# # c1, c2, c3 = 0.014, 0.028, 0.012  # opt slcv
# clf1 = LogisticRegression(class_weight='balanced', penalty='l1', C=c1)
# clf2 = LogisticRegression(class_weight='balanced', penalty='l1', C=c2)
# clf3 = LogisticRegression(class_weight='balanced', penalty='l1', C=c3)

# spec entropy 1) opt all p
# w = None
# clf1 = LogisticRegression(class_weight=w, penalty='l1', C=0.06)
# clf2 = LogisticRegression(class_weight=w, penalty='l1', C=0.04)
# clf3 = LogisticRegression(class_weight=w, penalty='l1', C=0.03)

# mfj 1) SLCV
# w = None
# clf1 = LogisticRegression(class_weight=w, penalty='l1', C=0.028) # pada pokud C=0.024
# clf2 = LogisticRegression(class_weight=w, penalty='l1', C=0.038) # pada pokud C=0.018
# clf3 = LogisticRegression(class_weight=w, penalty='l1', C=0.22)

# w = None
# clf1 = LogisticRegression(class_weight=w, penalty='l1', C=0.14)
# clf2 = LogisticRegression(class_weight=w, penalty='l1', C=0.17)
# clf3 = LogisticRegression(class_weight=w, penalty='l1', C=0.18)

# corr SLCV
# w = None
# clf1 = LogisticRegression(class_weight=w, penalty='l1', C=0.055)
# clf2 = LogisticRegression(class_weight=w, penalty='l1', C=0.16)
# clf3 = LogisticRegression(class_weight=w, penalty='l1', C=0.80)

# p1_offset = +0.0
# p2_offset = -0.0
# clf1 = LogisticRegression(class_weight=w, penalty='l1', C=0.03)
# clf2 = LogisticRegression(class_weight=w, penalty='l1', C=0.035)
# clf3 = LogisticRegression(class_weight=w, penalty='l1', C=0.051)

# clf1 = LogisticRegression(class_weight=w, penalty='l1', C=0.02)
# clf2 = LogisticRegression(class_weight=w, penalty='l1', C=0.03)
# clf3 = LogisticRegression(class_weight=w, penalty='l1', C=0.04)

# p = [0.9, 0.1]
# # p = [0.5, 0.5]
p = None
# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=4)), ('clf', GaussianNB(priors=p))])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=4)), ('clf', GaussianNB(priors=p))])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=14)), ('clf', GaussianNB(priors=p))])

# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=4)), ('clf', GaussianNB(priors=p))])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB(priors=p))])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=6)), ('clf', GaussianNB(priors=p))])

# clf1 = Pipeline([('fs', SelectKBest(f_classif, k=18)), ('clf', GaussianNB())])
# clf2 = Pipeline([('fs', SelectKBest(f_classif, k=7)), ('clf', GaussianNB())])
# clf3 = Pipeline([('fs', SelectKBest(f_classif, k=22)), ('clf', GaussianNB())])

# from sklearn.neighbors import KNeighborsClassifier
clf1 = KNeighborsClassifier(n_neighbors=140)
clf2 = KNeighborsClassifier(n_neighbors=160)
clf3 = KNeighborsClassifier(n_neighbors=40)

# print np.unique(XTRAIN)
# sys.exit()

# clf = MultinomialNB()
# clf1 = BernoulliNB(alpha=1, binarize=0)
# clf2 = BernoulliNB(alpha=1, binarize=-.25)
# clf3 = BernoulliNB(alpha=1, binarize=0)

aclf = [clf1, clf2, clf3]
acolors = ['r', 'g', 'b' ]
submission = pd.DataFrame()

# K = 3
R = 3

settings = Settings()
print settings

K = settings.kfoldCV
R = settings.repeatCV

nr_bins = 25
prob_calib_alg = 'rank'
# prob_calib_alg = settings.prob_calib_alg
# prob_calib_alg = None
# prob_calib_alg = 'median_centered'

for i in range(0, 3):

    nsubject = i + 1

    d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select[i], settings=settings)
    XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                                  d_tr[4], d_tr[5], d_tr[6]
    XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                                d_ts[4], d_ts[5], d_ts[6]

    # from spp_ut_feat_selection import JMIFeatureSelector
    # n_select_feat = 5
    # jmi = JMIFeatureSelector(k_feat=n_select_feat)
    # jmi.fit(XTRAIN, ytrain)
    # XTRAIN = jmi.fit_transform(XTRAIN, ytrain)
    # XTEST = jmi.transform(XTEST)

    ''' discretize '''
    # mdlp = MDLP(shuffle=False)
    # XTRAIN = mdlp.fit_transform(XTRAIN, ytrain)
    # XTEST = mdlp.transform(XTEST)
    # ind = np.asarray([len(cut) for cut in mdlp.cut_points_.itervalues()], dtype=int)
    # XTRAIN, XTEST = XTRAIN[:, ind > 0], XTEST[:, ind > 0]

    yhat_ts_rk = np.zeros((XTEST.shape[0], K * R))
    aauc = np.zeros((K*R, 2))
    yhat_valid_r = np.zeros((XTRAIN.shape[0], R))
    adkl = np.zeros((K * R, 2))

    cnt = -1
    for r in range(0, R):

        yhat_valid_k = np.zeros((XTRAIN.shape[0], 1))
        # cv = StratifiedKFoldPLabels(y=ytrain, plabels=plabels_tr, k=K)

        p = rename_groups_random(plabels_tr)
        cv = GroupKFold(n_splits=K).split(ytrain.copy(), ytrain, p)
        # cv = K

        for ifold, (itrn, itst) in enumerate(cv):
            cnt += 1
            x_tr, x_valid = XTRAIN[itrn, :], XTRAIN[itst, :]
            y_tr, y_valid = ytrain[itrn], ytrain[itst]

            ''' prediction '''
            clf = aclf[i]
            # clf.fit(XTRAIN, ytrain)
            clf.fit(x_tr, y_tr)
            yhat_tr = clf.predict_proba(x_tr)
            yhat_valid = clf.predict_proba(x_valid)
            yhat_ts = clf.predict_proba(XTEST)

            yhat_tr = yhat_tr[:, 1]
            yhat_valid = yhat_valid[:, 1]
            yhat_ts = yhat_ts[:, 1]

            yhat_valid = probability_calibration(yhat_valid, prob_calib_alg)
            yhat_ts = probability_calibration(yhat_ts, prob_calib_alg)

            # if nsubject == 1:
            #     yhat_valid[:, 1] = yhat_valid[:, 1] + p1_offset
            #     yhat_ts[:, 1] = yhat_ts[:, 1] + p1_offset
            #
            # if nsubject == 2:
            #     yhat_valid[:, 1] = yhat_valid[:, 1] + p2_offset
            #     yhat_ts[:, 1] = yhat_ts[:, 1] + p2_offset

            auc_tr = roc_auc_score(y_tr, yhat_tr)
            auc_valid = roc_auc_score(y_valid, yhat_valid)
            aauc[cnt, 0] = auc_tr
            aauc[cnt, 1] = auc_valid
            # print 'Patient {0}, auc_tr={1:3.2}, auc_valid={2:3.2}'.format(nsubject, auc_tr, auc_valid)

            yhat_ts_rk[:, cnt] = yhat_ts
            yhat_valid_k[itst, 0] = yhat_valid

            ''' ROC curves '''
            # plt.figure(20+nsubject)
            # fpr, tpr, dummy = roc_curve(y_valid, yhat_valid[:, 1], pos_label=1)
            # plt.plot(fpr, tpr, 'b', lw=1, label='ROC fold %d (area = %0.2f)' % (ifold, auc_valid))
            # plt.title('Patient {0}'.format(nsubject))
            # plt.grid()

            ''' discrete prob '''
            # plt.figure(nsubject)
            # plt.hist(yhat_valid[y_valid == 0, 1], 25, histtype='step', color='b', linewidth=1, label='valid0_{0}_{1}fold'.format(nsubject, ifold))
            # plt.hist(yhat_valid[y_valid == 1, 1], 25, histtype='step', color='r', linewidth=1, label='valid1_{0}_{1}fold'.format(nsubject, ifold))
            # plt.title('Patient {0} (validation data)'.format(nsubject))
            # plt.xlim(0, 1)
            # plt.grid()
            # # plt.legend()
            # # plt.show()

            # plt.figure(nsubject)
            # plt.hist(yhat_tr[:, 1], 20, histtype='step', color='r', linewidth=2, label='train'+str(nsubject))
            # plt.hist(yhat_ts[:, 1], 20, histtype='step', color='b', linewidth=2, label='test'+str(nsubject))
            # plt.title('Patient {0}'.format(nsubject))
            # plt.xlim(0, 1)
            # plt.grid()
            # plt.legend()

            # plt.figure(100)
            # plt.hist(yhat_tr[:, 1], 20, histtype='step', color=acolors[i], linewidth=2, label='train{0}_{1}fold'.format(nsubject, ifold))
            # plt.xlim(0, 1)
            # plt.grid()
            # plt.legend()

            # plt.figure(51)
            # plt.hist(yhat_ts[:, 1], 20, histtype='step', color=acolors[i], linewidth=1, label='test'+str(nsubject))
            # plt.xlim(0, 1)
            # plt.grid()
            # plt.legend()

            ''' continuous prob '''
            # Estimating the pdf and plotting
            # pdf_tr = gaussian_kde(yhat_tr[:, 1])
            # pdf_valid = gaussian_kde(yhat_valid[:, 1])
            # pdf_ts = gaussian_kde(yhat_ts[:, 1])
            # x = np.linspace(0, 1, 100)
            #
            # adkl[cnt, 0] = entropy(pdf_tr(x), pdf_ts(x))
            # adkl[cnt, 1] = entropy(pdf_valid(x), pdf_ts(x))

            # plt.figure(nsubject)
            # plt.plot(x, pdf_tr(x), color='r', linewidth=1, label='train{0}_{1}fold'.format(nsubject, ifold))
            # plt.plot(x, pdf_valid(x), color='b', linewidth=1, label='valid{0}_{1}fold'.format(nsubject, ifold))
            # plt.plot(x, pdf_ts(x), color='k', linewidth=1, label='test{0}_{1}fold'.format(nsubject, ifold))
            # plt.title('Patient {0}'.format(nsubject))
            # plt.legend()
            # plt.grid()

            # plt.figure(100)
            # plt.plot(x, pdf_ts(x), color='k', linewidth=1, label='test{0}_{1}fold'.format(nsubject, ifold))
            # plt.legend()
            # plt.grid()

            # plt.figure(1000)
            # plt.plot(x, pdf_tr(x), color=acolors[i], linewidth=2, label='train'+str(nsubject))
            # plt.legend()
            # plt.grid()
            #
            # plt.figure(51)
            # plt.plot(x, pdf_ts(x), color=acolors[i], linewidth=2, label='test'+str(nsubject))
            # plt.legend()
            # plt.grid()

        yhat_valid_r[:, r] = yhat_valid_k[:, 0]

    yt = yhat_valid_r.mean(axis=1)
    # yt = probability_calibration(yt[:, np.newaxis]).ravel()

    df = pd.DataFrame({"yhat_valid": yt, "ytrain": ytrain})
    submission = submission.append(df)

    print 'P{0} AUC AVG, auc_tr={1:3.2}, auc_valid={2:3.2}'.format(nsubject, aauc[:, 0].mean(), aauc[:, 1].mean())
    # print 'P{0} DKL, dkl_tr={1:3.2}, dkl_valid={2:3.2}'.format(nsubject, adkl[:, 0].mean(), adkl[:, 1].mean())

    # ''' na cele mnozine '''
    # clf = aclf[i]
    # clf.fit(XTRAIN, ytrain)
    # yhat_ts = clf.predict_proba(XTEST)
    # pdf_ts = gaussian_kde(yhat_ts[:, 1])

    # print 'Confussion matrix: '
    # print confusion_matrix(ytrain, clf.predict(XTRAIN))

    # plt.figure(1001)
    # plt.plot(x, pdf_ts(x), color='c', linewidth=4, label='test{0}_all'.format(nsubject))

    # plt.figure(nsubject)
    # plt.hist(yhat_ts[:, 1], 25, histtype='step', color='c', linewidth=3, label='test{0}_all'.format(nsubject))
    # plt.xlim(0, 1)
    # plt.grid()
    #
    # plt.figure(10)
    # plt.hist(yhat_ts[:, 1], 25, histtype='step', color='c', linewidth=3, label='test{0}_all'.format(nsubject))
    # plt.xlim(0, 1)
    # plt.grid()

    ''' prumer z jednotlivych foldu '''
    yhat_ts_avg = yhat_ts_rk.mean(axis=1)
    # pdf_ts = gaussian_kde(yhat_ts_avg)
    #
    # plt.figure(201)
    # plt.plot(x, pdf_ts(x), color=acolors[i], linewidth=3, label='test{0}_avg'.format(nsubject))
    # plt.grid()
    # plt.legend()

    plt.figure(10)
    plt.hist(yt[ytrain == 0], nr_bins, range=[0, 1], histtype='step', color=acolors[i], linewidth=1,
             linestyle='solid', label='valid0_{0}_avg'.format(nsubject))
    plt.hist(yt[ytrain == 1], nr_bins, range=[0, 1], histtype='step', color=acolors[i], linewidth=2,
             linestyle='solid', label='valid1_{0}_avg'.format(nsubject))
    plt.xlim(0, 1)
    plt.grid()
    plt.legend()

    plt.figure(70+nsubject)
    plt.hist(yhat_ts_avg, nr_bins, range=[0, 1], histtype='step', color='k', linewidth=1, label='test-unlabelled_{0}_avg'.format(nsubject))
    plt.hist(yt, nr_bins, range=[0, 1], histtype='step', color='g', linewidth=1, label='valid_{0}_avg'.format(nsubject))
    plt.title('Patient {0} (validation vs unlabelled)'.format(nsubject))
    plt.xlim(0, 1)
    plt.grid()
    plt.legend()

    # plt.figure(51)
    # plt.hist(yhat_ts_avg, 25, histtype='step', color='k', linewidth=3, label='test{0}_avg'.format(nsubject))
    # plt.xlim(0, 1)
    # plt.grid()
    # plt.legend()

yhat_valid = submission['yhat_valid'].get_values()
ytrain = submission['ytrain'].get_values()
auc = roc_auc_score(ytrain, yhat_valid)

''' ROC curve '''
plt.figure(24)
fpr, tpr, dummy = roc_curve(ytrain, yhat_valid, pos_label=1)
plt.plot(fpr, tpr, 'r', lw=1, label='auc = %0.5f' % (auc))
plt.title('ROC for all patients')
plt.grid()
plt.legend()

print 'AUC valid for P1-P2-P3: {0:3.3}'.format(auc)

plt.show()