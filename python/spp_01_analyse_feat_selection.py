import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# from feast import JMI

from utils import PreprocessPipeline, drop_data_quality_thr, remove_features_by_name
from spp_ut_feat_selection import JMIFeatureSelector
from spp_00_load_data import load_features, load_removed_features
from sklearn.feature_selection import f_classif, SelectFpr
from utils import PreprocessPipeline, drop_data_quality_thr, remove_features_by_name, rename_groups_random, \
    load_features_and_preprocess
from spp_ut_feat_selection import JMIFeatureSelector
from spp_00_load_data import load_features, load_removed_features
from spp_ut_settings import Settings

nsubject = 3

# feat_select = ['stat']
# feat_select = ['spectral']
# feat_select = ['sp_entropy']
# feat_select = ['mfj']
# feat_select = ['corr']
feat_select = ['stat', 'spectral']
# feat_select = ['stat', 'spectral', 'sp_entropy']
# feat_select = ['stat', 'spectral', 'sp_entropy', 'mfj', 'corr', 'wav_entropy']
# feat_select = ['wav_entropy']

settings = Settings()
# settings.remove_outliers = False
print settings

d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select, settings=settings)
XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                              d_tr[4], d_tr[5], d_tr[6]
XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                         d_ts[4], d_ts[5], d_ts[6]

ytrain = ytrain[:, np.newaxis]
ytest = ytest[:, np.newaxis]

T = np.hstack((XTRAIN, ytrain)).copy()
F = list(aFeatNames_tr)
F.append('y')

T2 = np.hstack((XTEST, ytest))
T = np.vstack((T, T2))

print 'Joined train and test set:'
print T.shape
print len(F)

df = pd.DataFrame(data=T, columns=F)
df.dropna(inplace=True)

ytrain = ytrain.ravel()
ytest = ytest.ravel()

''' FPR f_classif '''
fs = SelectFpr(f_classif, alpha=0.05)
tmp = fs.fit_transform(XTRAIN, ytrain)
feat_importance = fs.pvalues_
ind = np.argsort(feat_importance)
# ind = ind[::-1]
print 'f_classif: Selected features:', sum(feat_importance <= 0.05)

print '\n\n #### f_classif: sorted by importance'
for i in range(0, len(ind)):
    print '{0:30}: {1}'.format(aFeatNames_tr[ind[i]], feat_importance[ind[i]])

aFeatSelectedFclassif = [s for i, s in enumerate(aFeatNames_tr) if i in ind[0:5]]

''' FEATURE SELECTION '''
n_select_feat = 10
jmi = JMIFeatureSelector(k_feat=n_select_feat)
jmi.fit(XTRAIN, ytrain)
selected_indicies = jmi.selected_indicies_

aFeatSelected = list()
for i in selected_indicies:
    aFeatSelected.append(aFeatNames_tr[int(i)])

print '\nSELECTED FEATURES\n'
for i in range(0, n_select_feat):
    print '{0:6} {1}'.format(selected_indicies[i], aFeatSelected[i])

# print selected_indicies
# print aFeatSeleced
# print 'order alphabetically:', [s for i, s in enumerate(aFeatNames) if i in selected_indicies]
# print 'sort indicies + 1 !!!:', np.sort(selected_indicies)+1

names = aFeatSelected[0:5] + ['y']
sns.pairplot(data=df[names], hue='y', diag_kind='kde')

#
sns.set(style="white")

corr = df[aFeatSelected[0:8]].corr(method='spearman')
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, annot=True)

sns.plt.show()

