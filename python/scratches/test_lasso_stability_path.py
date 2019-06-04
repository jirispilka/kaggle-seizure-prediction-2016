import warnings
from matplotlib import pyplot as plt

from python.utils_learning import OutliersWinsorization
from sklearn.linear_model import lasso_stability_path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spp_00_load_data import load_features
from utils import *

nsubject = 2
bappend_test = False
# feat_select = ['stat']
feat_select = ['mfj', 'sp_entropy']
# feat_select = ['spectral', 'sp_entropy']

X_train, y_train, aFeatNames, aFiles_tr, p, data_q = load_features('train', nsubject, feat_select)
# X_test, y_test, dummy1, aFiles_ts, dummy2, dummy3 = load_feaures('test', nsubject, feat_select)

X_test = X_train.copy()
y_test = y_train.copy()
aFiles_ts = aFiles_tr

ind = np.sum(np.isnan(X_train), axis=0) < 50
X_train = X_train[:, ind]
aFeatNames = [s for i, s in enumerate(aFeatNames) if ind[i] == True]
X_test = X_test[:, ind]

''' drop nans '''
X_train, y_train, dummy4 = drop_nan(X_train, y_train, y_train.copy())
X_test, y_test, dummy5 = drop_nan(X_test, y_test, y_test.copy())

print 'Subject: ', nsubject
print 'Original dataset'
print X_train.shape
print y_train.shape

# # clf = RandomForestClassifier(class_weight='balanced', max_leaf_nodes=32, n_estimators=10, random_state=0)
# # clf = LogisticRegression(class_weight='balanced', penalty='l1', C=0.125)
# clf_lr = LogisticRegression(class_weight='balanced', penalty='l1', C=0.01)
# # rfe = RFE(estimator=clf_lr, step=5, verbose=2)
#
# clf = Pipeline([
#     ('ow', OutliersWinsorization()),
#     ('sc', StandardScaler()),
#     ('clf', clf_lr)
# ])

clf_transform = Pipeline([
    ('ow', OutliersWinsorization()),
    ('sc', StandardScaler())
])

# clf.fit(X_train, y_train)
# y_hat_train = clf.predict_proba(X_train)
# y_hat_test = clf.predict_proba(X_test)

X_train = clf_transform.fit_transform(X_train)
X_test = clf_transform.transform(X_test)

''' create pandas dataframe '''
# y_train = np.expand_dims(y_train, axis=1)
# y_test = np.expand_dims(y_test, axis=1)
#
# print X_train.shape
# print y_train.shape
#
# T = np.hstack((X_train, y_train))
# F = aFeatNames
# F.append('y')
#
# tmp = np.hstack((X_test, y_test))
# T = np.vstack((T, tmp))
#
# df = pd.DataFrame(data=T, columns=F)
# # df.dropna(inplace=True)

y_train = y_train.ravel()
y_test = y_test.ravel()


###########################################################################
# Plot stability selection path, using a high eps for early stopping
# of the path, to save computation time
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    alpha_grid, scores_path = lasso_stability_path(X_train, y_train, random_state=42, eps=0.5, verbose=1)

print alpha_grid
print scores_path.shape
# print scores_path.T[1:]

plt.figure()
# We plot the path as a function of alpha/alpha_max to the power 1/3: the
# power 1/3 scales the path less brutally than the log, and enables to
# see the progression along the path
# hg = plt.plot(alpha_grid[1:] ** .333, scores_path[coef != 0].T[1:], 'r')
hb = plt.plot(alpha_grid[1:] ** .333, scores_path.T[1:], 'k_feat')
ymin, ymax = plt.ylim()
plt.xlabel(r'$(\alpha / \alpha_{max})^{1/3}$')
plt.ylabel('Stability score: proportion of times selected')
# plt.title('Stability Scores Path - Mutual incoherence: %.1f' % mi)
plt.axis('tight')
# plt.legend((hg[0], hb[0]), ('relevant features', 'irrelevant features'),
#            loc='best')

# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data, feature_names=featnames, class_names=['Not survived', 'Survived'],
#                      filled=True, rounded=True, special_characters=True)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("classifier.pdf")

plt.show()

import sys
sys.exit(0)


# feat_importance = clf.feature_importances_
feat_importance = clf.steps[2][1].coef_.ravel()
# feat_importance = clf.steps[3][1].coef_.ravel()
feat_importance = abs(feat_importance)
ind = np.argsort(feat_importance)
ind = ind[::-1]
# print ind
# print feat_importance

plt.figure()
plt.stem(feat_importance)
plt.xticks(range(0, len(feat_importance)), aFeatNames)
plt.grid(True)
dummy, labels = plt.xticks()
plt.setp(labels, rotation=90)


print 'Features in order'
for name, val in zip(aFeatNames, feat_importance):
    print '{0:30}: {1}'.format(name, val)

print '\n\n #### Sorted by importance'
for i in range(0, len(ind)):
    print '{0:30}: {1}'.format(aFeatNames[ind[i]], feat_importance[ind[i]])

print y_hat_train.shape
print y_train.shape
print y_hat_test.shape

plt.figure()
plt.hist(y_hat_train[y_train == 1, 1], 20, histtype='step', color='r', linewidth=2)
plt.hist(y_hat_train[y_train == 0, 1], 20, histtype='step', color='b', linewidth=2)
plt.hist(y_hat_test[:, 1], 20, histtype='step', color='g', linewidth=2)
plt.title('Patient {0}'.format(nsubject))
plt.legend(['train 1', 'train 0', 'test set'])
plt.grid()


# print df
# print df.info()
# print df.describe()

names = [aFeatNames[ind[i]] for i in range(0, 4)]
# names = ['14-high_gamma_H_sh_ann', '14-low_gamma_H_sh_knn', '12-std']
names += 'y'

# sns.pairplot(data=df[['3-std', '15-std', '5-std', 'y']], hue='y', diag_kind='kde')
sns.pairplot(data=df[names], hue='y', diag_kind='kde')
# sns.pairplot(df)
# sns.boxplot(data=df, x=[1], y=['3-std'], hue='y')


plt.show()

