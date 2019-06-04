# from sklearn.cross_validation import StratifiedKFold
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GroupKFold
import numpy as np
from sklearn.utils import shuffle

from utils import StratifiedKFoldPLabels

plabels = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5])  # i.e. we have patients six patients (1 to 6)
y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])  # two classes

# plabels, y = shuffle(plabels, y)

cv = StratifiedKFoldPLabels(y, plabels, k=2)
# cv = GroupKFold(n_splits=2).split(y, y, plabels)
# cv = GroupKFoldEnsureSafe(y, plabels, k=2)

for itrn, itst in cv:
    print('ind train, test', itrn, itst)
    print('p train, test  ', plabels[itrn], plabels[itst])
    print('ttrain, test   ', y[itrn], y[itst])



# print plabels
# print plabels_new

# print p, ind
#
# newp = p.copy()
# print newp
#
# possible_group_vales = range(0, max(p))
#
#
# for i in range(0, len(p)):
#     val = p[i]
#     ind = plabels == val
#     print val, ind
#     new_val = np.random.randint(low=0, high=max(p))
#     print 'new_value', new_val


import sys
sys.exit()

# p, ind = np.unique(plabels, return_index=True)
#
# # print plabels[ind], ind
# # print p, y[ind]
# cv = StratifiedKFold(y[ind], n_folds=3, shuffle=True)
#
# itrn = list()
# itst = list()
# for tr, ts in cv:
#     print tr, ts
#     print p[tr], p[ts]
#
#     # najdu vsechny segmenty (z plabels), ktere odpovidaji danemu pacientovi
#     t1 = np.where(ismember(plabels, p[tr]))[0]
#     t2 = np.where(ismember(plabels, p[ts]))[0]
#     itrn.append(t1)
#     itst.append(t2)
#
#     print t1, t2
#     print plabels[t1], plabels[t2]
#     # for t in p[tr]:
#     #     print t in plabels
#
#

nsubject = 1
filename_tr = 'sp2016_feat_train_{0}_stat_20160912'.format(nsubject)
# filename_ts = 'features/sp2016_feat_test_{0}_stat_20160909'.format(nsubject)

X, y, aFeatNames, aFiles_tr, plabels = get_from_all_10_min(filename_tr)
# StratifiedKFoldPLabels(y, plabels, k_feat=10, seg=6)

y = y.ravel()
cv = StratifiedKFoldPLabels(y, plabels, k=10)

# for tr, ts in cv:
#     # print tr
#     # print ts
#     print plabels[tr]
#     print plabels[ts]
#     print 'labels count OUT ', np.bincount(y[tr]), np.bincount(y[ts])

# print y
#
# cv = StratifiedKFold(y, n_folds=60, shuffle=True)
#
# print cv

import sys
sys.exit(0)

# Xt = np.array([1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
# Xt = np.vstack((Xt, Xt)).T
# print Xt.shape

train_tmp = []
test_tmp = []
for train, test in cv:

    # train_tmp.append(train)
    # test_tmp.append(test)
    # print train, test
    # print y[train], y[test]

    label_cnt_tr = np.bincount(y[train])
    label_cnt_ts = np.bincount(y[test])
    # min_labels = np.min(label_counts)

    # print 'labels count ', label_cnt_tr, label_cnt_ts
    # print 'plabels train, plabels test', plabels[train], plabels[test]
    # print len(train) / float(len(test))

    # print plabels[train] in plabels[test]
    # for i in range(0, len(y)):
    loc = ismember(plabels[train], plabels[test])

    train_new = train[~loc]
    test_new = np.hstack((test, train[loc]))
    # print train_new, test_new
    # print plabels[train_new], plabels[test_new]

    label_cnt_tr = np.bincount(y[train_new])
    label_cnt_ts = np.bincount(y[test_new])
    # min_labels = np.min(label_counts)

    # print 'labels count OUT ', label_cnt_tr, label_cnt_ts
    # print len(train_new) / float(len(test_new))
    # print

    train_tmp.append(train_new)
    test_tmp.append(test_new)

    # projdu trenovaci sadu, pokud dany segment jak v trenovaci, tak v testovaci -> dam ho do testovaci
    # for val in plabels[train]:
    #     print val, val in plabels[test]

    # for i, val in enumerate(plabels[train]):
    #     print i, val

cv = zip(train_tmp, test_tmp)

# Check the division based on labels
if sum([sum(ismember(plabels[tr], plabels[ts])) for tr, ts in cv]) > 0:
    raise Exception('The same labels is in training and test set!')

import sys
sys.exit(0)

# for train, test in cv:
#
#     loc = ismember(plabels[train], plabels[test])
#     print loc



# print train_tmp, test_tmp



# for train, test in cv:
#     print train, test
