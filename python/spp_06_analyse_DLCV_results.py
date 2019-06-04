import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats


def print_mean_std(df, s):
    m, se = df.mean(), df.std()
    m *= 100
    se *= 100
    print '{0}: '.format(s),
    print '{0:3.3}+-{1:3.3} ({2:3.3}-{3:3.3})'.format(m, se, m - se, m + se)


# sname = '20160915_174448_patient_1_results_DLCV_LR'
# sname = '20160915_171500_patient_2_results_DLCV_LR'
# sname = '20160915_172554_patient_3_results_DLCV_LR'

# sname = '20160915_183434_patient_1_results_DLCV_NB'
# sname = '20160915_183529_patient_2_results_DLCV_NB'
# sname = '20160915_183955_patient_3_results_DLCV_NB'

# sname = '20160918_103501_patient_1_results_DLCV_spectral_NB'
# sname = '20160918_104532_patient_2_results_DLCV_spectral_NB'
# sname = '20160918_104910_patient_3_results_DLCV_spectral_NB'

# sname = '20160928_102918_patient_1_results_DLCV_stat_XGB'
# sname = '20160928_110513_patient_2_results_DLCV_stat_XGB'
# sname = '20160928_112438_patient_3_results_DLCV_stat_XGB'

sname = '20161024_224355_all_p_results_KNN_DLCV_stat_spec_rank.res'

df_results = pd.read_pickle(sname)

not_param = ['repetition', 'folds', 'result', 'result_1', 'result_2', 'result_3',
             'dkl', 'dkl_1', 'dkl_2', 'dkl_3', 'index', 'res_inner', 'res_outer']
parameters = {key: 0 for key in df_results.columns.tolist() if key not in not_param}

# df_results.drop(df_results['folds'] == 10, inplace=True)
df_results.drop(df_results.index[:1], inplace=True)
print df_results


df_results['folds'] = 3

print 'Mode of clf params: (NaN in res_inner, res_outer)'
print parameters
# print df_results.mode()

print_mean_std(df_results['res_inner'], 'inner')
print_mean_std(df_results['res_outer'], 'outer')

# print df_results.groupby(['folds']).median()
# print df_results.groupby(['folds']).mad()

plt.figure()
sns.boxplot(data=df_results[['res_inner', 'res_outer']], notch=True)
sns.swarmplot(data=df_results[['res_inner', 'res_outer']], color=".25")
sns.plt.ylim([0.5, 1])

# plt.figure()
# sns.distplot(df_results['res_inner'], color='r', label='inner')
# sns.distplot(df_results['res_outer'], color='b', label='outer')

nrcv = df_results['folds'].unique()

plt.figure()
for i, nr in enumerate(nrcv):

    print_mean_std(df_results[df_results['folds'] == nr]['res_inner'], 'inner-{0}'.format(nr))
    print_mean_std(df_results[df_results['folds'] == nr]['res_outer'], 'outer-{0}'.format(nr))

    plt.subplot(len(nrcv), 1, i+1)
    sns.distplot(df_results[df_results['folds'] == nr]['res_inner'], color='r', label='inner')
    sns.distplot(df_results[df_results['folds'] == nr]['res_outer'], color='b', label='outer')
    sns.plt.title('folds: {0}'.format(nr))
    sns.plt.xlim([0.4, 1])

# s = 'clf__C'
# s = 'clf__max_depth'
# s = 'fs__k'
# s = 'n_estimators'
# s = 'max_depth'
# s = 'colsample_bytree'
# param = ['2_clf__C']

# for s in parameters:
#
#     plt.figure()
#     sns.countplot(x=s, data=df_results)
#     sns.plt.title(s)

    # plt.figure()
    # sns.boxplot(x=s, y='res_inner', data=df_results)
    #
    # plt.figure()
    # sns.boxplot(x=s, y='res_outer', data=df_results)

# plt.figure()
# sns.boxplot(data=df_results, y=['res_inner', 'res_outer'], notch=True, hue='folds')
# # sns.swarmplot(data=df_results[['res_inner', 'res_outer']], color=".25")
# sns.plt.ylim([0.5, 1])

plt.show()
