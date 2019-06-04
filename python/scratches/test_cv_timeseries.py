from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import random
import sys
from utils import insert_pathol_to_normal_random_keep_order, TimeSeriesSplitGroupSafe, ismember

# X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
# y = np.array([1, 2, 3, 4])
# tscv = TimeSeriesSplit(n_splits=3)
# tscv = TimeSeriesSplitGroupSafe(n_splits=3)
# print X.shape
# print(tscv)
#
# for train_index, test_index in tscv.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     # print X[train_index]
#     # print X[test_index]
#     # X_train, X_test = X[train_index], X[test_index]
#     # y_train, y_test = y[train_index], y[test_index]

from utils import load_features_and_preprocess
from spp_ut_settings import Settings
# from sklearn.utils import shuffle

settings = Settings()
settings.remove_outliers = False
settings.standardize = False
settings.drop_nan = False

d_tr, d_ts = load_features_and_preprocess(3, ['stat'], settings, verbose=True)
XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                              d_tr[4], d_tr[5], d_tr[6]

ytrain = ytrain.ravel()
XTRAIN, ytrain, plabels_tr = insert_pathol_to_normal_random_keep_order(XTRAIN, ytrain, plabels_tr)

tscv = TimeSeriesSplitGroupSafe(n_splits=100)

p = np.unique(plabels_tr)

for train_index, test_index in tscv.split(XTRAIN, ytrain, plabels_tr):
    # pass
    print("TRAIN:", train_index, "TEST:", test_index)
    print("Groups TRAIN:", plabels_tr[train_index], "Groups TEST:", plabels_tr[test_index])

    # sys.exit()


# print np.asarray(ind_for_resampling)

# p_0 = np.unique(plabels_tr[ytrain == 0])
# p_1 = np.unique(plabels_tr[ytrain == 1])
#
# print p_0
# print p_1
#
# p_to_shift = np.random.randint(min(p_0), max(p_0), len(p_1))
# p_to_shift = np.sort(p_to_shift)
# print p_to_shift
#
# p0_new = np.zeros((len(p_0) + len(p_1), 1), dtype=np.int)
#
# cnt = -1
# for i in range(0, len(p_0)):
#     cnt += 1
#     if p_0[i] not in p_to_shift:
#         p0_new[cnt] = p_0[i]
#     else:
#         cnt += 1
#         p0_new[cnt] = p_0[i]
#
# print p0_new

# # p_shifted = p_0.copy()
# for i in range(0, len(p_0)):
#     if p_0[i] in ind_to_shift:
#         print p_0[i]




# p_unique = np.unique(p, return_index=False)
# p_unique_shuffled = shuffle(p_unique)
#
# # print p_unique
# # print p_unique_shuffled
#
# p_renamed = p.copy()
#
# for i in range(0, len(p_unique)):
#     ind = np.where(p == p_unique[i])[0]
#     p_renamed[ind] = p_unique_shuffled[i]
#
# assert max(p) == max(p_renamed)
# assert min(p) == min(p_renamed)
#
# return p_renamed