import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import sys


# sname = '20161019_101151_all_p_results_DLCV_INNER_mix_rank.res'
sname = '20161026_patient_3_results_SLCV_KNN_rem_cov_false_[3]k_stat_spectral_sp_entropy.res'
# sname = '20161020_patient_1_results_SLCV_KNN_JMI_[3]k_sp_entropy.res'

WRITE_CSV = False


def print_mean_std(df, s):
    m, se = df.mean(), df.std()
    print '{0}: '.format(s),
    print '{0:2.2}+-{1:2.2} ({2:2.2}-{3:2.2})'.format(m, se, m - se, m + se)

df_results = pd.read_pickle(sname)

# print df_results

b_all_patients = True if 'result_1' in df_results.columns.tolist() else False

not_param = ['repetition', 'folds', 'result', 'result_1', 'result_2', 'result_3',
             'dkl', 'dkl_1', 'dkl_2', 'dkl_3', 'index']

parameters = {key: 0 for key in df_results.columns.tolist() if key not in not_param}
# nrcv = df_results['folds'].unique()
# print nrcv
print parameters

# with pd.option_context('display.max_rows', 999, 'display.max_columns', 12):
#     print df_results.groupby(parameters.keys()).median()

# df_results.groupby(parameters.keys()).median().sort_values(by='result', ascending=False).to_csv('temp.csv')
if not b_all_patients:
    result = df_results.groupby(parameters.keys(), as_index=False).agg({'result': ['median', 'std']})
else:
    if 'dkl' in df_results.columns.tolist():
        result = df_results.groupby(
            parameters.keys(), as_index=False).agg({'result': ['median', 'std'], 'result_1': ['median'],
                                                    'result_2': ['median'], 'result_3': ['median']})
        # , 'dkl': ['median', 'std'],'dkl_1': ['median'], 'dkl_2': ['median'], 'dkl_3': ['median']})
    else:
        result = df_results.groupby(
            parameters.keys(), as_index=False).agg({'result': ['median', 'std'],'result_1': ['median'],
                                                    'result_2': ['median'], 'result_3': ['median']})

# df_results.to_csv('temp.csv')

names = result.columns.tolist()
names.sort(key=lambda tup: tup[0])

if WRITE_CSV:
    result[names].to_csv(sname[:-4] + '.csv')
result = result.sort_values(by=[('result', 'median')], ascending=False)

if WRITE_CSV:
    result[names].to_csv(sname[:-4] + '_sorted.csv')
print result
# result.to_csv('temp.csv')
# sns.pointplot(x='index', y=('result', 'median'), data=result)
#
# sns.pointplot(data=result.reset_index(), x='index', y=('result', 'median'), ci=('result', 'std'))
# df.sort([('Group1', 'C')], ascending=False)
# print result['result','mean']

# plt.figure()
# plt.subplot(211)
# plt.plot(result['result', 'median'], marker='.', linestyle='None')
# # plt.ylim(0.6, 0.8)
# plt.subplot(212)
# plt.errorbar(range(1, len(result)+1), result['result', 'median'], yerr=result['result', 'std'],
#              marker='.', linewidth=2, linestyle='None')
# # plt.ylim(0.5, 0.9)

# plt.show()
# import sys
# sys.exit(0)

# s = '1_fs__k' if '1_fs__k' in parameters else 'jmi__k_feat'
# s = 'jmi__k_feat'
s = 'clf__n_neighbors'
# s = 'nb__alpha'
# s = 'clf3__gr__group'
# s = 'clf1__fs__k'
# s = 'fs__k'
# s = 'clf3__fs__k'
# s = 'clf3__xgb__n_estimators'

for p in parameters.keys():
    plt.figure()
    # sns.pointplot(x=p, hue='max_depth', y='result', data=df_results[df_results['folds'] == 2], estimator=np.median)
    sns.pointplot(x=p, hue=s, y='result', data=df_results, estimator=np.median)
    plt.ylim(0.6, 0.8)


if '1_clf__C' in parameters:
    plt.figure()
    plt.subplot(311)
    sns.pointplot(x='1_clf__C', y='result_1', data=df_results, estimator=np.median, color='r')
    plt.legend('patient 1')
    plt.grid()
    plt.subplot(312)
    sns.pointplot(x='2_clf__C', y='result_2', data=df_results, estimator=np.median, color='g')
    plt.legend('patient 2')
    plt.grid()
    plt.subplot(313)
    sns.pointplot(x='3_clf__C', y='result_3', data=df_results, estimator=np.median, color='b')
    plt.legend('patient 3')
    plt.grid()

# plt.figure()
# sns.pointplot(x='1_fs__k', y='result_1', data=df_results, estimator=np.median, color='r', n_boot=100)
# sns.pointplot(x='2_fs__k', y='result_2', data=df_results, estimator=np.median, color='g', n_boot=100)
# sns.pointplot(x='3_fs__k', y='result_3', data=df_results, estimator=np.median, color='b', n_boot=100)
#
# plt.figure()
# sns.pointplot(x='1_fs__k', y='dkl_1', data=df_results, estimator=np.median, color='r', n_boot=100)
# sns.pointplot(x='2_fs__k', y='dkl_2', data=df_results, estimator=np.median, color='g', n_boot=100)
# sns.pointplot(x='3_fs__k', y='dkl_3', data=df_results, estimator=np.median, color='b', n_boot=100)

print_mean_std(df_results['result'], 'results-all')

# plt.figure()
# for i, nr in enumerate(nrcv):
#     print_mean_std(df_results[df_results['folds'] == nr]['result'], 'results-{0:3}'.format(nr))
#     plt.subplot(len(nrcv), 1, i+1)
#     sns.distplot(df_results[df_results['folds'] == nr]['result'], color='r', label='inner')
#     sns.plt.title('folds: {0}'.format(nr))
#     sns.plt.xlim([0, 1])

# df_results.drop(df_results['n_estimators'] > 10, inplace=True)
# print df_results

# plt.figure()
# df_results.boxplot(column="dkl", by=parameters.keys(), notch=True)
# dummy, labels = plt.xticks()
# # plt.ylim(0.5, 1)
# plt.setp(labels, rotation=90)

# plt.figure()
# plt.plot(result[('result', 'median')].get_values(), result[('dkl', 'median')].get_values(), marker='.', linestyle='None')

# print df_results.groupby('1_fs__k').median()

sns.plt.show()