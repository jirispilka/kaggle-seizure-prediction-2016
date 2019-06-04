import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import sys

sname = '20161007_103144_all_p_res_SLCV_LR_[5]k.res_dkl.csv'

df_results = pd.read_csv(sname, index_col=0)

bjoin = True

if bjoin:
    df_results.drop(['result', 'result_1', 'result_2', 'result_3', 'result.1'], axis=1, inplace=True)
    # print df_results
    sname_join = '20161007_104135_all_p_res_SLCV_LR_[10]k.csv'
    dft = pd.read_csv(sname_join, index_col=0)
    dft.dropna(inplace=True)
    # print dft
    df_results = df_results.join(dft, rsuffix='_join')
    # print df_results

df_results.replace([np.inf, -np.inf], np.nan, inplace=True)
df_results.dropna(axis=0, inplace=True)
# print df_results
result_sort = df_results.sort_values(by=['result'], ascending=False)
result_sort.to_csv('temp.csv')

not_param = ['repetition', 'folds', 'result', 'result_1', 'result_2', 'result_3', 'result.1,',
             'dkl1', 'dkl2', 'dkl3', 'dkl_all', 'index']

parameters = {key: 0 for key in df_results.columns.tolist() if key not in not_param}
print parameters

# sns.pointplot(x='1_clf__C', y='result_1', data=df_results, estimator=np.median, color='r')
# sns.pointplot(x='2_clf__C', y='result_2', data=df_results, estimator=np.median, color='g')
# sns.pointplot(x='3_clf__C', y='result_3', data=df_results, estimator=np.median, color='b')

sns.pointplot(x='1_clf__C', y='dkl1', data=df_results, estimator=np.median, color='r')
sns.pointplot(x='2_clf__C', y='dkl2', data=df_results, estimator=np.median, color='g')
sns.pointplot(x='3_clf__C', y='dkl3', data=df_results, estimator=np.median, color='b')
# sns.plt.figure()
# sns.pairplot(x_vars='result', y_vars='dkl_all', data=df_results)

plt.plot(df_results['result'].get_values(), df_results['dkl_all'].get_values(), marker='.', linestyle='None')

plt.show()
