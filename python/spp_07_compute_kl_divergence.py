import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.grid_search import ParameterGrid, _CVScoreTuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import f_classif, SelectKBest
import numpy as np
import sys
import time

from utils import PreprocessPipeline, drop_data_quality_thr, drop_nan_single, remove_features_by_name
from spp_00_load_data import load_features, load_removed_features

# sall = ['stat', 'spectral', 'sp_entropy', 'mfj', 'corr']
# feat_select = [sall, sall, sall]

feat_select = [['stat'], ['stat'], ['stat']]

REMOVE_COVARIATE_SHIFT = True

pp = PreprocessPipeline(remove_outliers=True, standardize=True)

d_data_train = dict()
d_data_test = dict()
for i in range(0, 3):

    nsubject = i + 1

    XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr = load_features('train', nsubject, feat_select[i])
    XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts = load_features('test', nsubject, feat_select[i])

    pp.fit(XTRAIN, XTEST, drop_nan=True)

    print 'Subject: ', nsubject
    print 'Original dataset'
    print XTRAIN.shape
    print ytrain.shape

    XTRAIN, ytrain, plabels_tr = drop_data_quality_thr(XTRAIN, ytrain, plabels_tr, data_q_tr, 10)
    XTRAIN = pp.transform(XTRAIN)

    XTEST = drop_nan_single(XTEST)
    XTEST = pp.transform(XTEST)

    print 'Transformed and removed NaNs'
    print XTRAIN.shape
    print ytrain.shape
    print plabels_tr.shape

    if REMOVE_COVARIATE_SHIFT:
        l_feat_remove = load_removed_features(nsubject, feat_select[i])
        XTRAIN, aFeatNames_tr, ind_remove = remove_features_by_name(XTRAIN, aFeatNames_tr, l_feat_remove)
        # XTEST, aFeatNames_ts, ind_remove = remove_features_by_name(XTEST, aFeatNames_ts, l_feat_remove)

        print 'Removed features with covariate shift: '
        print 'TRAIN :', XTRAIN.shape
        # print '        XTEST :', XTEST.shape

    T = np.hstack((XTRAIN, ytrain[:, np.newaxis], plabels_tr[:, np.newaxis]))
    names = aFeatNames_tr
    names.append('ytrain')
    names.append('plabels')
    df = pd.DataFrame(data=T, columns=names)

    df_ts = pd.DataFrame(data=XTEST, columns=aFeatNames_ts)
    d_data_train[nsubject] = df
    d_data_test[nsubject] = df_ts

sname = '20161010_124113_all_p_res_SLCV_LR_[5]k.csv'

df_results = pd.read_csv(sname, index_col=0)
df_results.dropna(axis=0, inplace=True)

# print df_results

''' LR '''
# clflr = LogisticRegression(class_weight='balanced', penalty='l1', n_jobs=1)
# clf = Pipeline([
#     ('clf', clflr)
# ])

clf = Pipeline([('fs', SelectKBest(f_classif)), ('clf', GaussianNB())])

not_param = ['repetition', 'folds', 'result', 'result_1', 'result_2', 'result_3', 'result.1', 'index']
parameters = {key: 0 for key in df_results.columns.tolist() if key not in not_param}
print parameters

df_results['dkl1'] = np.nan
df_results['dkl2'] = np.nan
df_results['dkl3'] = np.nan
df_results['dkl_all'] = np.nan

predicted_yhat_tr = dict()  # store predicted yhat
predicted_yhat_ts = dict()

nr = df_results['result'].count()

for index, row in df_results.iterrows():

    for p in parameters:
        val = row[p]
        parameters[p] = val

    sys.stdout.write('Processed params: {0}%, {1}\r'.format((100 * int(index) / nr), int(index)))
    # sys.stdout.write("Processed params: %d%% \r" % (iparam))
    sys.stdout.flush()

    print parameters

    dkl = np.zeros((3, 1))
    for (key, df), (key_ts, df_ts) in zip(d_data_train.iteritems(), d_data_test.iteritems()):
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df_ts, pd.DataFrame)
        # print 'ifold: {0}, patient: {1}'.format(i, key)

        ''' select parameters for this patient only'''
        param_for_patient = dict()
        for name, val in parameters.iteritems():
            if '{0}_'.format(key) in name[0:2]:
                param_for_patient[name[2:]] = val

        s = '_'.join('{0}_{1}'.format(k, v) for k, v in param_for_patient.items())
        clf_param_key = str(key) + '_'  + s

        if clf_param_key not in predicted_yhat_tr:

            afeatnames = [col for col in df.columns if col not in ['ytrain', 'plabels']]
            XTRAIN = df[afeatnames].get_values()
            ytrain = df['ytrain'].get_values()

            # df_ts.dropna(axis=0, inplace=True)
            XTEST = df_ts[afeatnames].get_values()
            # ind = np.any(np.isnan(XTEST), axis=1)
            # XTEST[ind] = 0

            # print param_for_patient

            clf.set_params(**param_for_patient)
            ''' prediction '''
            clf.fit(XTRAIN, ytrain)
            yhat_tr = clf.predict_proba(XTRAIN)
            yhat_ts = clf.predict_proba(XTEST)
            # yhat_ts[ind, :] = 0

            predicted_yhat_tr[clf_param_key] = yhat_tr
            predicted_yhat_ts[clf_param_key] = yhat_ts

        else:
            yhat_tr = predicted_yhat_tr[clf_param_key]
            yhat_ts = predicted_yhat_ts[clf_param_key]

        # plt.figure()
        # plt.hist(yhat_tr[:, 1], 20, histtype='step', color='r', linewidth=2, label='train')
        # plt.hist(yhat_ts[:, 1], 20, histtype='step', color='b', linewidth=2, label='test')
        # plt.title('Patient {0}'.format(key))
        # plt.xlim(0, 1)
        # plt.grid()
        # plt.legend()
        # # plt.show()

        from scipy.stats.kde import gaussian_kde
        from scipy.stats import entropy

        # Estimating the pdf and plotting
        pdf_tr = gaussian_kde(yhat_tr[:, 1])
        pdf_ts = gaussian_kde(yhat_ts[:, 1])
        x = np.linspace(0, 1, 100)

        en = entropy(pdf_tr(x), pdf_ts(x))
        # print key-1
        # print dkl.shape
        dkl[key-1] = en

        # plt.figure()
        # plt.plot(x, pdf_tr(x), color='r', linewidth=2, label='train')
        # plt.plot(x, pdf_ts(x), color='b', linewidth=2, label='test')
        # # plt.hist(d1_np,normed=1,color="cyan",alpha=.8)
        # # plt.plot(x,norm.pdf(x,mu,stdv),label="parametric distribution",color="red")
        # plt.legend()
        # plt.grid()
        # # plt.show()

        # sys.exit()

    # print dkl.T, dkl.mean()
    print row['result'], dkl.mean(), dkl.T

    df_results.loc[index, 'dkl1'] = dkl[0]
    df_results.loc[index, 'dkl2'] = dkl[1]
    df_results.loc[index, 'dkl3'] = dkl[2]
    df_results.loc[index, 'dkl_all'] = dkl.mean()


df_results.to_csv(sname[0:-4] + '_dkl.csv')