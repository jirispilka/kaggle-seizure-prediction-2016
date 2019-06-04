import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold, StratifiedKFold
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import sys

from utils import rename_groups_random, load_features_and_preprocess
from spp_ut_settings import Settings

nsubject = 1

# feat_select = ['stat']
feat_select = ['spectral']
# feat_select = ['sp_entropy']
# feat_select = ['mfj']
# feat_select = ['corr']

settings = Settings()
print settings

K = 3
R = 40  # settings.repeatCV

d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select, settings=settings)
XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                              d_tr[4], d_tr[5], d_tr[6]
XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                         d_ts[4], d_ts[5], d_ts[6]

# clf = RandomForestClassifier(class_weight='balanced', random_state=2016)
# parameters = {
#     'max_depth': [1] + range(2, 9, 2),
#     # 'max_leaf_nodes': [int(x) for x in np.exp2(range(1, 8))],
#     # 'n_estimators': range(5, 21, 5) + range(20, 101, 20) + [150]
# }

''' Adaboost '''
# clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1))
# parameters = {
#     'n_estimators': range(10, 201, 50),
# }


''' SVM '''
# # # # cs = l1_min_c(XTRAIN, ytrain, loss='log') * np.logspace(0, 2)
# clf = SVC(kernel='linear', probability=True, class_weight='balanced')
# # # clf = LinearSVC(class_weight='balanced', loss='hinge')
# parameters = {
#     # 'penalty': ['l1', 'l2'],
#     'C': np.exp2(range(-24, 5, 3)),
#     # 'C': np.arange(0.000001, 0.01, 0.0001),
#     # 'C': [0.1, 1, 10, 100, 1000],
#     # 'gamma': np.exp2(range(-20, -2, 1)),
#     # 'degree': [2],
#     # 'C': cs
# }

''' Logistic regression '''
w = 'balanced'
# w = None
clf = LogisticRegression(class_weight=w, penalty='l1', n_jobs=1, C=0.08)

''' Naive Bayes '''
# clf = GaussianNB(priors=None)

''' KNN '''
# clf = KNeighborsClassifier(n_neighbors=100)

p = rename_groups_random(plabels_tr)
skf = GroupKFold(n_splits=K)
# skf = StratifiedKFold(n_splits=10)

# sm = SMOTE(kind='regular')
# XTRAIN, ytrain = sm.fit_sample(XTRAIN, ytrain)

Y = np.zeros((len(ytrain), 1))

for i, (itrn, itst) in enumerate(skf.split(XTRAIN, ytrain, p)):
# for i, (itrn, itst) in enumerate(skf.split(XTRAIN, ytrain)):

    print len(itrn), len(itst)

    xtr, xvalid = XTRAIN[itrn, :], XTRAIN[itst, :]
    ytr, yvalid = ytrain[itrn], ytrain[itst]
    # pvalid = p[itst]

    clf.fit(xtr, ytr)

    y_hat_valid = clf.predict_proba(xvalid)
    # yhat_ts = clf.predict_proba(XTEST)

    Y[itst, 0] = y_hat_valid[:, 1]

    # fr_of_pos, mean_pred_val = calibration_curve(yvalid, y_hat_valid[:, 1], n_bins=10)
    # xtmp = np.arange(0, 1, 0.01)

    # plt.figure(1500)
    # plt.plot(xtmp, xtmp, 'k--')
    # plt.plot(mean_pred_val, fr_of_pos, color='b', label='fold_{0}'.format(i), marker='x')
    #
    # plt.figure(10)
    # plt.hist(y_hat_valid[yvalid == 0, 1], range=[0, 1], bins=50, histtype='step', color='b', linewidth=1, label='valid0_{0}fold'.format(i))
    # plt.hist(y_hat_valid[yvalid == 1, 1], range=[0, 1], bins=50, histtype='step', color='r', linewidth=1, label='valid0_{0}fold'.format(i))
    # plt.hist(yhat_ts[:, 1], range=[0, 1], bins=50, histtype='step', color='k', linewidth=1, label='unlabelled0_{0}fold'.format(i))
    # plt.title('Patient {0} (validation data)'.format(nsubject))
    # plt.xlim(0, 1)
    # plt.grid()
    # # plt.legend()
    # # plt.show()

    # cv_gen = GroupKFold(n_splits=10)
    # itr, its = list(), list()
    # for train, test in cv_gen.split(xvalid, yvalid, pvalid):
    #     itr.append(train)
    #     its.append(test)
    #
    # cv = zip(itr, its)
    #
    # clf_cp = CalibratedClassifierCV(base_estimator=clf, method='isotonic', cv=cv)
    # clf_cp.fit(!!!!!, yvalid)
    # y_hat_calibrated = clf_cp.predict_proba(xvalid)
    # fr_of_pos, mean_pred_val = calibration_curve(yvalid, y_hat_calibrated[:, 1], n_bins=10)
    #
    # plt.figure(1500)
    # plt.plot(xtmp, xtmp, 'k--')
    # plt.plot(mean_pred_val, fr_of_pos, color='r', label='fold_{0}'.format(i), marker='x')
    # plt.grid()
    # plt.legend()
    #
    # yhat_ts_calibrated = clf_cp.predict_proba(XTEST)
    #
    # plt.figure(1501)
    # plt.hist(y_hat_calibrated[:, 1], range=[0, 1], bins=50, histtype='step', color='b', linewidth=1, label='valid0_{0}fold'.format(i))
    # plt.hist(yhat_ts_calibrated[:, 1], range=[0, 1], bins=50, histtype='step', color='k', linewidth=1, label='unlabelled0_{0}fold'.format(i))
    # plt.title('Patient {0} (po kalibraci)'.format(nsubject))
    # plt.xlim(0, 1)
    # plt.grid()

Y = Y.ravel()
clf.fit(XTRAIN, ytrain)
yhat_ts = clf.predict_proba(XTEST)

plt.figure(10)
plt.hist(Y[ytrain == 0], range=[0, 1], bins=50, histtype='step', color='b', linewidth=1, label='valid0')
plt.hist(Y[ytrain == 1], range=[0, 1], bins=50, histtype='step', color='r', linewidth=1, label='valid1')
plt.hist(yhat_ts[:, 1], range=[0, 1], bins=50, histtype='step', color='k', linewidth=1, label='unlabelled0')
plt.xlim(0, 1)
plt.title('probability distribution original')
plt.grid()

''' ROC curve '''
auc = roc_auc_score(ytrain, Y)
plt.figure(120)
fpr, tpr, dummy = roc_curve(ytrain, Y, pos_label=1)
plt.plot(fpr, tpr, 'b', lw=3, label='auc original = %0.5f' % (auc))
plt.grid()
plt.title('ROC for original')

''' probability ranking '''
from scipy.stats import rankdata
y_ranked = rankdata(Y) / float(len(Y))


''' probability calibration '''
fr_of_pos, mean_pred_val = calibration_curve(ytrain, Y, n_bins=10)
xtmp = np.arange(0, 1, 0.01)

# cv_gen = GroupKFold(n_splits=5)
cv_gen = StratifiedKFold(n_splits=10)
itr, its = list(), list()
# for train, test in cv_gen.split(XTRAIN, ytrain, p):
for train, test in cv_gen.split(XTRAIN, ytrain):
    itr.append(train)
    its.append(test)

cv = zip(itr, its)

print XTRAIN.shape, ytrain.shape

# clf = LogisticRegression(class_weight=w, penalty='l1', n_jobs=1, C=0.008)
clf_cp = CalibratedClassifierCV(base_estimator=clf, method='sigmoid', cv=cv)
clf_cp.fit(XTRAIN, ytrain)
y_hat_calibrated = clf_cp.predict_proba(XTRAIN)
fr_of_pos_cp, mean_pred_val_cp = calibration_curve(ytrain, y_hat_calibrated[:, 1], n_bins=10)

plt.figure(1500)
plt.plot(xtmp, xtmp, 'k--')
plt.plot(mean_pred_val, fr_of_pos, color='b', label='valid_{0}', marker='x')
plt.plot(mean_pred_val_cp, fr_of_pos_cp, color='r', marker='o', label='probability calibrated')
plt.grid()
plt.legend()

y_hat_calibrated = y_hat_calibrated[:, 1]
y_ts_calibrated = clf_cp.predict_proba(XTEST)
# y_ts_ranked = rankdata(y_ts_calibrated[:, 1]) / float(len(y_ts_calibrated[:, 1]))

plt.figure(11)
plt.hist(y_hat_calibrated[ytrain == 0], range=[0, 1], bins=50, histtype='step', color='b', linewidth=1, label='valid0')
plt.hist(y_hat_calibrated[ytrain == 1], range=[0, 1], bins=50, histtype='step', color='r', linewidth=1, label='valid1')
plt.hist(y_ts_calibrated[:, 1], range=[0, 1], bins=50, histtype='step', color='k', linewidth=1, label='unlabelled0')
plt.xlim(0, 1)
plt.title('Probablity calibration')
plt.grid()

plt.figure(12)
plt.hist(y_ranked[ytrain == 0], range=[0, 1], bins=50, histtype='step', color='b', linewidth=1, label='valid0')
plt.hist(y_ranked[ytrain == 1], range=[0, 1], bins=50, histtype='step', color='r', linewidth=1, label='valid1')
# plt.hist(y_ts_ranked, range=[0, 1], bins=50, histtype='step', color='k', linewidth=1, label='unlabelled0')
plt.xlim(0, 1)
plt.title('Probability ranking')
plt.grid()

''' ROC curve '''
auc = roc_auc_score(ytrain, y_ranked)
plt.figure(120)
fpr, tpr, dummy23 = roc_curve(ytrain, y_ranked, pos_label=1)
plt.plot(fpr, tpr, 'r', lw=1, label='auc ranked = %0.5f' % (auc))
plt.grid()
plt.legend()
plt.title('ROC for ranked')

# plt.legend()
# plt.show()

plt.show()