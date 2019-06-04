import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from utils import PreprocessPipeline, drop_data_quality_thr, remove_features_by_name, load_features_and_preprocess
from spp_00_load_data import load_features, load_removed_features
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, SelectFpr
from spp_ut_settings import Settings

settings = Settings()
print settings

nsubject = 3

# feat_select = ['stat']
# feat_select = ['spectral']
# feat_select = ['sp_entropy']
# feat_select = ['mfj']
# feat_select = ['corr']
feat_select = ['wav_entropy']
# feat_select = ['spectral']

d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select, settings=settings)
XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                              d_tr[4], d_tr[5], d_tr[6]
XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                            d_ts[4], d_ts[5], d_ts[6]

T = np.hstack((XTRAIN, ytrain[:, np.newaxis])).copy()
F = list(aFeatNames_tr)
F.append('y')

T2 = np.hstack((XTEST, ytest[:, np.newaxis]))
T = np.vstack((T, T2))

print 'Joined data:'
print T.shape
print len(F)

df = pd.DataFrame(data=T, columns=F)
# df.dropna(inplace=True)

print XTRAIN.shape
print ytrain.shape
print len(aFeatNames_tr)

fs = SelectFpr(f_classif, alpha=0.05)
tmp = fs.fit_transform(XTRAIN, ytrain)
feat_importance = fs.pvalues_
ind = np.argsort(feat_importance)
# ind = ind[::-1]
print 'Selected features:', sum(feat_importance <= 0.05)

print '\n\n #### Sorted by importance'
for i in range(0, len(ind)):
    print '{0:30}: {1}'.format(aFeatNames_tr[ind[i]], feat_importance[ind[i]])

# plt.figure()
# plt.stem(feat_importance)
# plt.xticks(range(0, len(feat_importance)), aFeatNames)
# plt.grid(True)
# dummy2, labels = plt.xticks()
# plt.setp(labels, rotation=90)
# # plt.show()

names = [s for i, s in enumerate(aFeatNames_tr) if i in ind[0:5]]
names.append('y')
sns.pairplot(data=df[names], hue='y', diag_kind='kde')


corr = df[names].corr(method='spearman')
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1,
            square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax, annot=True)

sns.plt.show()

