import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import GroupKFold
from matplotlib import pyplot as plt
import numpy as np
import sys
import time
from scipy.stats.kde import gaussian_kde
from scipy.stats import entropy

from utils import rename_groups_random, load_features_and_preprocess, probability_calibration
from spp_ut_settings import Settings


feat_select = [['sp_entropy'], ['sp_entropy'], ['sp_entropy']]

w = 'balanced'
clf1 = LogisticRegression(class_weight=w, penalty='l1', C=0.5)
clf2 = LogisticRegression(class_weight=w, penalty='l1', C=0.05)
clf3 = LogisticRegression(class_weight=w, penalty='l1', C=0.045)

aclf = [clf1, clf2, clf3]
acolors = ['r', 'g', 'b' ]
submission = pd.DataFrame()

settings = Settings()
print settings

K = settings.kfoldCV
R = 1  # settings.repeatCV

prob_calib_alg = 'median_centered'

for i in range(0, 3):

    nsubject = i + 1

    d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select[i], settings=settings)
    XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                                  d_tr[4], d_tr[5], d_tr[6]
    XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                                d_ts[4], d_ts[5], d_ts[6]

    auc_all_p = np.zeros((K * R, 3))
    yhat_valid_r = np.zeros((XTRAIN.shape[0], R))
    yhat_valid_ranked_r = np.zeros((XTRAIN.shape[0], R))

    cnt = -1
    for r in range(0, R):

        yhat_valid_k = np.zeros((XTRAIN.shape[0], 1))
        yhat_valid_ranked_k = np.zeros((XTRAIN.shape[0], 1))

        p = rename_groups_random(plabels_tr)
        cv = GroupKFold(n_splits=K).split(ytrain.copy(), ytrain, p)

        for ifold, (itrn, itst) in enumerate(cv):

            cnt += 1
            x_tr, x_valid = XTRAIN[itrn, :], XTRAIN[itst, :]
            y_tr, y_valid = ytrain[itrn], ytrain[itst]

            ''' prediction '''
            clf = aclf[i]
            clf.fit(x_tr, y_tr)
            yhat_tr = clf.predict_proba(x_tr)
            yhat_valid = clf.predict_proba(x_valid)
            yhat_ts = clf.predict_proba(XTEST)

            yhat_valid_ranked = probability_calibration(yhat_valid[:, 1], prob_calib_alg)

            auc_tr = roc_auc_score(y_tr, yhat_tr[:, 1])
            auc_valid = roc_auc_score(y_valid, yhat_valid[:, 1])
            auc_valid_ranked = roc_auc_score(y_valid, yhat_valid_ranked)

            auc_all_p[cnt, 0] = auc_tr
            auc_all_p[cnt, 1] = auc_valid
            auc_all_p[cnt, 2] = auc_valid_ranked

            # yhat_ts_rk[:, cnt] = yhat_ts[:, 1]
            yhat_valid_k[itst, 0] = yhat_valid[:, 1]
            yhat_valid_ranked_k[itst, 0] = yhat_valid_ranked

            ''' ROC curves '''
            # plt.figure(10+nsubject)
            # fpr, tpr, dummy = roc_curve(y_valid, yhat_valid[:, 1], pos_label=1)
            # plt.plot(fpr, tpr, 'b', lw=1, label='ROC fold %d (area = %0.2f)' % (ifold, auc_valid))
            # plt.title('Patient {0}'.format(nsubject))
            # plt.grid()

            ''' discrete prob '''
            plt.figure(20+nsubject)
            plt.hist(yhat_valid[y_valid == 0, 1], 25, histtype='step', color='b', linewidth=1, label='valid0_{0}_{1}fold'.format(nsubject, ifold))
            plt.hist(yhat_valid[y_valid == 1, 1], 25, histtype='step', color='r', linewidth=1, label='valid1_{0}_{1}fold'.format(nsubject, ifold))
            plt.title('Patient {0} (validation data)'.format(nsubject))
            plt.xlim(0, 1)
            plt.grid()
            # plt.legend()
            # plt.show()

            # plt.figure(30+nsubject)
            # plt.hist(yhat_valid[:, 1], 20, histtype='step', color='g', linewidth=2, label='train'+str(nsubject))
            # plt.hist(yhat_ts[:, 1], 20, histtype='step', color='k', linewidth=2, label='test_unlabelled'+str(nsubject))
            # plt.title('Patient {0}'.format(nsubject))
            # plt.xlim(0, 1)
            # plt.grid()
            # plt.legend()

        yhat_valid_r[:, r] = yhat_valid_k[:, 0]
        yhat_valid_ranked_r[:, r] = yhat_valid_ranked_k[:, 0]

    yt = yhat_valid_r.mean(axis=1)
    yrt = yhat_valid_ranked_r.mean(axis=1)

    ''' discrete prob '''
    plt.figure(40)
    plt.hist(yrt[ytrain == 0], 50, range=[0, 1], histtype='step', color=acolors[i], lw=2, label='valid_{0}'.format(nsubject))
    plt.hist(yrt[ytrain == 1], 50, range=[0, 1], histtype='step', color=acolors[i], lw=2,
             label='valid_{0}'.format(nsubject))
    plt.title('Validation data)'.format(nsubject))
    plt.xlim(0, 1)
    plt.legend()
    plt.grid()

    auc = roc_auc_score(ytrain, yt)
    aucr = roc_auc_score(ytrain, yrt)

    print 'AUC averaged across folds: valid {0:3.3}, ranked {1: 3.3}'.format(auc, aucr)
    # print auc_all_p

    df = pd.DataFrame({"yhat_valid": yt, "yhat_valid_ranked": yrt, "ytrain": ytrain})
    submission = submission.append(df)

    print 'P{0} AUC AVG, auc_tr={1:3.3}, auc_valid={2:3.3}, auc_valid_ranked={2:3.3}'.format(
        nsubject, auc_all_p[:, 0].mean(), auc_all_p[:, 1].mean(), auc_all_p[:, 2].mean())


yhat_valid = submission['yhat_valid'].get_values()
yhat_valid_ranked = submission['yhat_valid_ranked'].get_values()
ytrain = submission['ytrain'].get_values()
auc = roc_auc_score(ytrain, yhat_valid)
aucr = roc_auc_score(ytrain, yhat_valid_ranked)

''' ROC curve '''
plt.figure(24)
fpr, tpr, dummy3 = roc_curve(ytrain, yhat_valid, pos_label=1)
plt.plot(fpr, tpr, 'r', lw=1, label='auc all p valid = %0.5f' % (auc))
fpr, tpr, dummy4 = roc_curve(ytrain, yhat_valid_ranked, pos_label=1)
plt.plot(fpr, tpr, 'k', lw=1, label='auc all p ranked = %0.5f' % (aucr))
plt.title('ROC for all patients')
plt.grid()
plt.legend()

print 'AUC valid for P1-P2-P3: valid: {0:2.3}, ranked: {1:2.3}'.format(auc, aucr)

plt.show()