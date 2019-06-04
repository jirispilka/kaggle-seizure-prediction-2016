from matplotlib import pyplot as plt

import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.feature_selection import SelectFpr, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from spp_00_load_data import load_features, load_removed_features
from utils import load_features_and_preprocess
from spp_ut_settings import Settings
from spp_ut_feat_selection import JMIFeatureSelector


nsubject = 3
feat_select = ['stat']
# feat_select = ['sp_entropy']
# feat_select = ['spectral']
# feat_select = ['sp_entropy']
# feat_select = ['mfj']
# feat_select = ['corr']
# feat_select = ['stat', 'spectral', 'sp_entropy']

c = 1e-2

settings = Settings()
print settings

# d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select, settings=settings)
# XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
#                                                                               d_tr[4], d_tr[5], d_tr[6]
# XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
#                                                                          d_ts[4], d_ts[5], d_ts[6]

d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select, settings=settings)

XTRAIN, ytrain, aFeatNames_tr, plabels_tr = d_tr['X'], d_tr['y'], d_tr['aFeatNames'], d_tr['plabels']
XTEST, ytest, aFeatNames_ts = d_ts['X'], d_ts['y'], d_ts['aFeatNames']

# from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import RandomOverSampler
# from imblearn.combine.smote_enn import SMOTEENN
# from imblearn.combine.smote_tomek import SMOTETomek
# # sm = SMOTE(kind='regular', k=3, m=5)
# # sm = RandomOverSampler()
# sm = SMOTETomek()
# XTRAIN, ytrain = sm.fit_sample(XTRAIN, ytrain)


# ''' FEATURE SELECTION '''
# n_select_feat = 100
# jmi = JMIFeatureSelector(k_feat=n_select_feat)
# jmi.fit(XTRAIN, ytrain)
# XTRAIN = jmi.transform(XTRAIN)
# XTEST = jmi.transform(XTEST)
#
# aFeatSeleced = list()
# for i in jmi.selected_indicies_:
#     aFeatSeleced.append(aFeatNames_tr[int(i)])
#
# print '\nSELECTED FEATURES\n'
# for i in range(0, n_select_feat):
#     print '{0:6} {1}'.format(jmi.selected_indicies_[i], aFeatSeleced[i])
#
# aFeatNames_tr = aFeatSeleced
# print 'TRAIN :', XTRAIN.shape
# print 'ytrain:', ytrain.shape
# ''' -------------------- '''

# clf = RandomForestClassifier(class_weight='balanced', max_leaf_nodes=32, n_estimators=10, random_state=0)
# clf = LogisticRegression(class_weight='balanced', penalty='l1', C=0.125)
# clf_lr = LogisticRegression(class_weight='balanced', penalty='l1', C=0.006)
# clf_lr = LogisticRegression(class_weight='balanced', penalty='l1', C=0.0031)
# clf_lr = LogisticRegression(class_weight='balanced', penalty='l1', C=0.0075)
# rfe = RFE(estimator=clf_lr, step=5, verbose=2)

# clf_lr = RandomForestClassifier(class_weight='balanced', max_depth=2, n_estimators=20, random_state=2016)


w = 'balanced'
# w = None
clf = LogisticRegression(class_weight=w, penalty='l1', C=c)

# clf2 = xgb.XGBClassifier(max_depth=2, n_estimators=6, colsample_bytree=0.2, seed=2016)

# fratio = sum(y_train == 0) / float(sum(y_train == 1))
# clfxgb = xgb.XGBClassifier(max_depth=1, n_estimators=100, min_child_weight=1, gamma=0,
#                            learning_rate=0.1, colsample_bytree=0.1, subsample=0.7,
#                            reg_lambda=1, reg_alpha=0,
#                            objective='binary:logistic', seed=2016, silent=1,
#                            scale_pos_weight=fratio)

# clf = Pipeline([
#     ('ow', OutliersWinsorization()),
#     ('sc', StandardScaler()),
#     ('fs', SelectFpr(f_classif, alpha=0.05)),
#     # ('clflr', clf_lr),
# ])

# clf_transform = Pipeline([
#     ('ow', OutliersWinsorization()),
#     ('sc', StandardScaler())
# ])

# remove_feat = load_removed_features(nsubject, feat_select)
# XTRAIN, aFeatNames_tr, ind_to_remove = remove_features_by_name(XTRAIN, aFeatNames_tr, remove_feat)
# XTEST = np.delete(XTEST, ind_to_remove, axis=1)

# print len(aFeatNames)
# print ind_to_remove
# print 'Removed features:'
# print 'TRAIN: ', XTRAIN.shape
# print 'TEST: ', XTEST.shape

# X = np.vstack((X_train, X_test))
# y = np.vstack((np.zeros((len(y_train), 1)), np.ones((len(y_test), 1))))
# y.ravel()
# clf.fit(X, y)

clf.fit(XTRAIN, ytrain)
y_hat_train = clf.predict_proba(XTRAIN)
y_hat_test = clf.predict_proba(XTEST)

# # xgb boost
# X_train = clf.fit_transform(X_train, y_train)
# X_test = clf.transform(X_test)
# ind = clf.named_steps['fs'].pvalues_ <= 0.05
# aFeatNames = [s for i, s in enumerate(aFeatNames) if ind[i] == True]
#
# clfxgb.fit(X_train, y_train, eval_metric='auc')
# xgb.plot_tree(clfxgb, 0)
#
# y_hat_train = clfxgb.predict_proba(X_train)
# y_hat_test = clfxgb.predict_proba(X_test)
#
# X_train = clf_transform.fit_transform(X_train)
# X_test = clf_transform.transform(X_test)

''' create pandas dataframe '''
# ytrain = np.expand_dims(ytrain, axis=1)
# ytest = np.expand_dims(ytest, axis=1)

print XTRAIN.shape
print XTEST.shape
print ytrain.shape
print ytest.shape

T = np.hstack((XTRAIN, ytrain[:, np.newaxis]))
F = aFeatNames_tr
F.append('y')

tmp = np.hstack((XTEST, ytest[:, np.newaxis]))
T = np.vstack((T, tmp))

df = pd.DataFrame(data=T, columns=F)
# df.dropna(inplace=True)

ytrain = ytrain.ravel()
ytest = ytest.ravel()

# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data, feature_names=featnames, class_names=['Not survived', 'Survived'],
#                      filled=True, rounded=True, special_characters=True)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("classifier.pdf")


# feat_importance = clf.feature_importances_
feat_importance = clf.coef_.ravel()
# feat_importance = clf.steps[3][1].feature_importances_.ravel()
# feat_importance = clfxgb.feature_importances_.ravel()
# feat_importance = clf.steps[2][1].coef_.ravel()
# feat_importance = clf.steps[3][1].coef_.ravel()
feat_importance = abs(feat_importance)
ind = np.argsort(feat_importance)
ind = ind[::-1]
# print ind
print 'Pocet priznaku: ', sum(feat_importance > 1e-5)

plt.figure()
plt.stem(feat_importance)
plt.xticks(range(0, len(feat_importance)), aFeatNames_tr)
plt.grid(True)
dummy, labels = plt.xticks()
plt.setp(labels, rotation=90)


# print 'Features in order'
# for i, (name, val) in enumerate(zip(aFeatNames, feat_importance)):
#     print '{0:3},{1:30}: {2}'.format(i, name, val)
#
print '\n\n #### Sorted by importance'
for i in range(0, len(ind)):
    print '{0:30}: {1}'.format(aFeatNames_tr[ind[i]], feat_importance[ind[i]])

print y_hat_train.shape
print ytrain.shape
print y_hat_test.shape

plt.figure()
plt.hist(y_hat_train[ytrain == 1, 1], 50, range=[0, 1], histtype='step', color='r', linewidth=2)
plt.hist(y_hat_train[ytrain == 0, 1], 50, range=[0, 1], histtype='step', color='b', linewidth=2)
plt.hist(y_hat_test[:, 1], 50, range=[0, 1], histtype='step', color='g', linewidth=2)
plt.title('Patient {0}'.format(nsubject))
plt.legend(['train 1', 'train 0', 'test set'])
plt.grid()

# from scipy.stats.kde import gaussian_kde
# from scipy.stats import entropy
#
# # Estimating the pdf and plotting
# pdf_tr = gaussian_kde(y_hat_train[:, 1])
# pdf_ts = gaussian_kde(y_hat_test[:, 1])
# x = np.linspace(0, 1, 100)
#
# en = entropy(pdf_tr(x), pdf_ts(x))
# print en
#
# plt.figure()
# plt.plot(x, pdf_tr(x), color='r', linewidth=2, label='train')
# plt.plot(x, pdf_ts(x), color='b', linewidth=2, label='test')
# # plt.hist(d1_np,normed=1,color="cyan",alpha=.8)
# # plt.plot(x,norm.pdf(x,mu,stdv),label="parametric distribution",color="red")
# plt.legend()
# plt.grid()
# # plt.show()


# print df
# print df.info()
# print df.describe()

names = [aFeatNames_tr[ind[i]] for i in range(0, 5)]
# names = ['7-energy_0002', '16-spectrum_slope', '8-e_beta']
names += 'y'

# sns.pairplot(data=df[['3-std', '15-std', '5-std', 'y']], hue='y', diag_kind='kde')
sns.pairplot(data=df[names], hue='y', diag_kind='kde')
# sns.pairplot(df)
# sns.boxplot(data=df, x=[1], y=['3-std'], hue='y')

xt = XTRAIN[:, feat_importance > 1e-5]

pca = PCA(n_components=2)
xt = pca.fit_transform(xt)

print 'xt', xt.shape

plt.figure()
plt.plot(xt[ytrain == 0, 0], xt[ytrain == 0, 1], marker='.', color='g', linestyle='None')
plt.plot(xt[ytrain == 1, 0], xt[ytrain == 1, 1], marker='.', color='r', linestyle='None')
plt.title('PCA of the selected features')

plt.show()