from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import pandas as pd
import xgboost as xgb
import pydot
from sklearn.externals.six import StringIO
from sklearn.linear_model import LogisticRegression

import numpy as np
from matplotlib import pyplot as plt
from utils import get_from_all_10_min

nsubject = 2

filename_tr = 'sp2016_feat_train_{0}_stat_20160909'.format(nsubject)
filename_ts = 'sp2016_feat_test_{0}_stat_20160909'.format(nsubject)

X_train, y_train, aFeatNames, aFiles_tr = get_from_all_10_min(filename_tr)
mask = np.any(np.isnan(X_train), axis=1)
X_train = X_train[~mask]
y_train = y_train[~mask].ravel()

# clf = RandomForestClassifier(class_weight='balanced', max_leaf_nodes=32, n_estimators=10, random_state=0)
# clf = LogisticRegression(class_weight='balanced', penalty='l2', C=1)
clf = DecisionTreeClassifier(class_weight='balanced', max_depth=4)

clf.fit(X_train, y_train)
y_hat_train = clf.predict(X_train)

# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data, feature_names=featnames, class_names=['Not survived', 'Survived'],
#                      filled=True, rounded=True, special_characters=True)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph.write_pdf("classifier.pdf")


feat_importance = clf.feature_importances_
# feat_importance = clf.coef_.ravel()
tmp = np.sort(feat_importance)

print feat_importance

plt.figure()
plt.stem(feat_importance)
plt.xticks(range(0, len(feat_importance)), aFeatNames)
plt.grid(True)
# plt.labe(featnames)
# dummy, labels = plt.xticks()
# plt.setp(labels, rotation=90)


for name, val in zip(aFeatNames, feat_importance):
    print name, val

plt.show()