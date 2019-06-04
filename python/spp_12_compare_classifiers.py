import pandas as pd
import seaborn as sns
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GroupKFold
import xgboost as xgb
import itertools

from matplotlib import pyplot as plt
import numpy as np
from utils import rename_groups_random, load_features_and_preprocess, select_feature_group, \
    probability_calibration, VotingClassifierRank
from spp_ut_settings import Settings


# feat_select = ['stat', 'spectral', 'sp_entropy', 'mfj', 'corr']
# feat_select = ['stat', 'sp_entropy', 'mfj', 'corr']
# feat_select = ['stat', 'spectral', 'sp_entropy']
# feat_select = ['stat', 'sp_entropy']
# feat_select = ['sp_entropy', 'mfj']
# feat_select = ['mfj']
# feat_select = ['stat', 'spectral', 'sp_entropy', 'mfj', 'corr']
# feat_select = ['spectral', 'spectral', 'spectral', 'spectral']
# feat_select = ['spectral', 'spectral', 'spectral', 'spectral']

feat_select_unique = ['stat', 'spectral', 'sp_entropy']

feat_select = ['spectral', 'sp_entropy', 'spectral'] # P1
# feat_select = ['stat', 'sp_entropy', 'spectral', 'sp_entropy']  # P2
# feat_select = ['stat', 'stat', 'stat', 'spectral']  # P3

nsubject = 1

if nsubject == 1:
    clf1 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
    clf2 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.016)
    clf3 = xgb.XGBClassifier(max_depth=2, n_estimators=10, min_child_weight=1, gamma=0,
                             learning_rate=0.1, colsample_bytree=0.3, subsample=0.9,
                             reg_lambda=1, reg_alpha=1, nthread=15,
                             objective='binary:logistic', seed=2016, silent=1,
                             scale_pos_weight=8)
    # clf4 = RandomForestClassifier(max_depth=2, n_estimators=10, random_state=2016)

elif nsubject == 2:

    clf1 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
    clf2 = Pipeline([('fs', SelectKBest(f_classif, k=10)), ('clf', GaussianNB())])
    clf3 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.006)
    clf4 = xgb.XGBClassifier(max_depth=1, n_estimators=90, min_child_weight=1, gamma=0,
                             learning_rate=0.1, colsample_bytree=0.3, subsample=0.9,
                             reg_lambda=1, reg_alpha=1, nthread=15,
                             objective='binary:logistic', seed=2016, silent=1,
                             scale_pos_weight=14)

else:

    clf1 = LogisticRegression(class_weight='balanced', penalty='l1', C=0.014)
    clf2 = Pipeline([('fs', SelectKBest(f_classif, k=28)), ('clf', GaussianNB())])
    clf3 = LogisticRegression(class_weight='balanced', penalty='l1', C=1)
    clf4 = LogisticRegression(class_weight='balanced', penalty='l1', C=.5)
    # clf4 = xgb.XGBClassifier(max_depth=1, n_estimators=90, min_child_weight=1, gamma=0,
    #                          learning_rate=0.1, colsample_bytree=0.3, subsample=0.9,
    #                          reg_lambda=1, reg_alpha=1, nthread=15,
    #                          objective='binary:logistic', seed=2016, silent=1,
    #                          scale_pos_weight=8)

aclf = [clf1, clf2, clf3]
# aclf = [clf1, clf2, clf3, clf4]
# aclf = [clf1, clf2]
# aclf = [clf1, clf2, clf3, clf4]
acolors = ['r', 'g', 'b', 'y', 'c']

df_results = pd.DataFrame()
# df_results = pd.DataFrame(columns=['feat_group', 'electrode', 'total', 'removed'], index=range(0, 200))

settings = Settings()
print settings

K = settings.kfoldCV
R = settings.repeatCV

# settings.remove_covariate_shift = False

nr_bins = 25
# prob_calib_alg = settings.prob_calib_alg
# prob_calib_alg = None
prob_calib_alg = 'rank'
# prob_calib_alg = 'median_centered'

d_tr, d_ts = load_features_and_preprocess(nsubject,feat_select_unique, settings=settings, verbose=False)
XTRAIN_ALL, ytrain, aFeatNames_tr_all, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                              d_tr[4], d_tr[5], d_tr[6]
XTEST_ALL, ytest, aFeatNames_ts_all, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                            d_ts[4], d_ts[5], d_ts[6]

y_all_clf = np.zeros((XTRAIN_ALL.shape[0], len(aclf)))
auc_all = np.zeros((len(aclf), 1))

for i, clf in enumerate(aclf):

    print XTRAIN_ALL.shape
    XTRAIN, aFeatNames_tr, dummy3 = select_feature_group(XTRAIN_ALL, aFeatNames_tr_all, feature_group=feat_select[i],
                                                         verbose=False)
    XTEST, aFeatNames_ts, dummy2 = select_feature_group(XTRAIN_ALL, aFeatNames_ts_all, feature_group=feat_select[i])

    print feat_select[i], XTRAIN.shape, XTEST.shape

    yhat_ts_rk = np.zeros((XTEST.shape[0], K * R))
    aauc = np.zeros((K*R, 1))
    yhat_valid_r = np.zeros((XTRAIN.shape[0], R))

    cnt = -1
    for r in range(0, R):

        p = rename_groups_random(plabels_tr)
        cv = GroupKFold(n_splits=K).split(ytrain.copy(), ytrain, p)
        # cv = K

        for ifold, (itrn, itst) in enumerate(cv):
            cnt += 1
            x_tr, x_valid = XTRAIN[itrn, :], XTRAIN[itst, :]
            y_tr, y_valid = ytrain[itrn], ytrain[itst]

            ''' prediction '''
            clf.fit(x_tr, y_tr)
            yhat_valid = clf.predict_proba(x_valid)
            yhat_ts = clf.predict_proba(XTEST)

            yhat_valid = yhat_valid[:, 1]
            yhat_ts = yhat_ts[:, 1]

            yhat_valid = probability_calibration(yhat_valid, prob_calib_alg)
            yhat_ts = probability_calibration(yhat_ts, prob_calib_alg)

            auc_valid = roc_auc_score(y_valid, yhat_valid)
            aauc[cnt, 0] = auc_valid

            yhat_ts_rk[:, cnt] = yhat_ts
            yhat_valid_r[itst, r] = yhat_valid

            ''' discrete prob '''
            # plt.figure(nsubject)
            # plt.hist(yhat_valid[y_valid == 0], bins=nr_bins, range=[0, 1], histtype='step', color='b', linewidth=1,
            #          label='valid0_{0}_{1}fold'.format(nsubject, ifold))
            # plt.hist(yhat_valid[y_valid == 1], bins=nr_bins, range=[0, 1], histtype='step', color='r', linewidth=1,
            #          label='valid1_{0}_{1}fold'.format(nsubject, ifold))
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

    yt = yhat_valid_r.mean(axis=1)

    y_all_clf[:, i] = yt

    # df = pd.DataFrame({"yhat_valid": yt, "ytrain": ytrain})

    # print feat_select
    # print feat_select[0], yt.shape

    # df_results['ytrain'] = ytrain
    # df_results[feat_select[i]] = yt
    # df_results = df_results.append(df)

    auc_all[i, 0] = aauc[:, 0].mean()

    print 'P{0}, feat: {1} -- AUC AVG, auc_valid = {2:3.3}'.format(nsubject, feat_select[i], aauc[:, 0].mean())

    ''' ROC curves '''
    plt.figure(1)
    fpr, tpr, dummy = roc_curve(ytrain, yt, pos_label=1)
    plt.plot(fpr, tpr, color=acolors[i], lw=1, label='ROC feat: %s (area = %0.3f)' % (feat_select[i],  aauc[:, 0].mean()))
    plt.title('Patient {0}'.format(nsubject))
    plt.grid()
    plt.legend(loc=4)

    ''' prumer z jednotlivych foldu '''
    yhat_ts_avg = yhat_ts_rk.mean(axis=1)

    plt.figure(10)
    plt.hist(yt[ytrain == 0], nr_bins, range=[0, 1], histtype='step', color=acolors[i], linewidth=1,
             linestyle='solid', label='%s' % feat_select[i])
    plt.hist(yt[ytrain == 1], nr_bins, range=[0, 1], histtype='step', color=acolors[i], linewidth=2,
             linestyle='solid', label='%s' % feat_select[i])
    plt.xlim(0, 1)
    plt.grid()
    plt.legend()

    # plt.figure(70+nsubject)
    # plt.hist(yhat_ts_avg, nr_bins, range=[0, 1], histtype='step', color='k', linewidth=1, label='test-unlabelled_{0}_avg'.format(nsubject))
    # plt.hist(yt, nr_bins, range=[0, 1], histtype='step', color='g', linewidth=1, label='valid_{0}_avg'.format(nsubject))
    # plt.title('Patient {0} (validation vs unlabelled)'.format(nsubject))
    # plt.xlim(0, 1)
    # plt.grid()
    # plt.legend()
    #
    # plt.figure(51)
    # plt.hist(yhat_ts_avg, 25, histtype='step', color='k', linewidth=3, label='test{0}_avg'.format(nsubject))
    # plt.xlim(0, 1)
    # plt.grid()
    # plt.legend()


T = np.hstack((y_all_clf, ytrain[:, np.newaxis]))
feat_select = [s+str(i) for i, s in enumerate(feat_select)]
F = feat_select
F.append('y')
df = pd.DataFrame(columns=F, data=T)

sns.pairplot(data=df[feat_select], hue='y', diag_kind='kde')

''' ROC curve '''
for i in range(0, y_all_clf.shape[1]):
    y_all_clf[:, i] = probability_calibration(y_all_clf[:, i], prob_calib_alg)

# yhat_valid = np.average(y_all_clf, axis=1, weights=auc_all.ravel())
yhat_valid = np.average(y_all_clf, axis=1)
auc = roc_auc_score(ytrain, yhat_valid)

print 'Classifiers combination: %0.5f' % auc

plt.figure(1)
fpr, tpr, dummy21 = roc_curve(ytrain, yhat_valid, pos_label=1)
plt.plot(fpr, tpr, 'k', lw=2, label='combination auc = %0.4f' % (auc))
plt.title('ROC for combination')
plt.grid()
plt.legend(loc=4)

corr = df.corr(method='spearman')
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, annot=True)

plt.show()