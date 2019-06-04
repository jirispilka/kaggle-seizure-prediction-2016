import numpy as np
import pandas as pd
import sys

from utils import load_removed_features, load_features_and_preprocess
from spp_ut_settings import Settings

# afeat_select = ['stat', 'spectral', 'sp_entropy', 'mfj']
afeat_select = ['stat', 'spectral', 'sp_entropy']
# afeat_select = ['stat', 'sp_entropy']

settings = Settings()
print settings
settings.remove_covariate_shift = False
settings.qthr = -1


def match_electrode(i, s):
    ind = s.index('-')
    return int(s[0:ind]) == i


def get_nr_feat_electrode(feat_names2):

    vals = np.zeros((16, 1))
    for iiel in range(0, 16):
        nr = sum(map(lambda s: match_electrode(iiel+1, s), feat_names2))
        vals[iiel, 0] = nr

    return vals

df = pd.DataFrame(columns=['feat_group', 'electrode', 'total', 'removed'], index=range(0, 200))

# print df


index = -1
nsubject = 2

for sfeat in afeat_select:

    d_tr, dummy1 = load_features_and_preprocess(nsubject, [sfeat], settings=settings, verbose=False)
    feat_names = d_tr[2]

    feat_names_removed = load_removed_features(nsubject, [sfeat])
    # feat_names_removed += load_removed_features(nsubject, ['stat_spectral_sp_entropy_mfj_corr'])

    nr_total = get_nr_feat_electrode(feat_names)
    nr_removed = get_nr_feat_electrode(feat_names_removed)

    for iel in range(0, 16):
        index += 1
        df['feat_group'].loc[index] = sfeat
        df['total'].loc[index] = nr_total[iel, 0]
        df['electrode'].loc[index] = iel+1
        df['removed'].loc[index] = nr_removed[iel, 0]

df.dropna(inplace=True, axis=0)
print df

print df.groupby('electrode').sum()

