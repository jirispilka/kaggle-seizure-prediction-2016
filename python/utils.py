
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.feature_selection.base import SelectorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.externals.joblib import Parallel, delayed
from sklearn.externals import six
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.utils import check_array, deprecated, shuffle
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES
from sklearn.metrics import roc_auc_score
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils.validation import indexable, _num_samples
import pandas as pd
import re
import numpy as np
from scipy.stats import rankdata, mannwhitneyu
from scipy.stats import hmean, gmean
import random
from sklearn.preprocessing import StandardScaler

from spp_00_load_data import load_features, load_removed_features

d_map_feat_names_int = {'stat': 0, 'spectral': 1, 'sp_entropy': 2, 'mfj': 3, 'corr': 4, 'wav_entropy': 5}


def drop_nan(X, y, p):
    mask = np.logical_or(np.any(np.isnan(X), axis=1), np.any(np.isinf(X), axis=1))
    X = X[~mask]
    y = y[~mask].ravel()
    p = p[~mask]
    return X, y, p


def drop_nan_single(X):
    mask = np.logical_or(np.any(np.isnan(X), axis=1), np.any(np.isinf(X), axis=1))
    X = X[~mask]
    return X


def drop_data_quality_thr(X, y, p, q, thr=10):
    ind = np.logical_or(q < thr, np.any(np.isnan(X), axis=1))
    X = X[~ind]
    y = y[~ind].ravel()
    p = p[~ind]
    return X, y, p, ind


def load_features_and_preprocess(nsubject, feat_select, settings, verbose=True):

    qthr = settings.qthr
    remove_covariates = settings.remove_covariate_shift
    remove_outliers = settings.remove_outliers
    standardize = settings.standardize
    drop_nan = settings.drop_nan

    # XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr = load_features('train', nsubject, feat_select)
    # XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts = load_features('test', nsubject, feat_select)

    data_tr = load_features('train', nsubject, feat_select)
    data_ts = load_features('test', nsubject, feat_select)

    XTRAIN, ytrain, plabels_tr, data_q_tr = data_tr['X'], data_tr['y'], data_tr['plabels'], data_tr['data_q']
    XTEST, ytest, plabels_ts = data_ts['X'], data_ts['y'], data_ts['plabels']

    aFeatNames_tr = data_tr['aFeatNames']
    aFeatNames_ts = data_ts['aFeatNames']

    # data['X'] = X
    # data['y'] = y
    # data['aFeatNames'] = afeatnames
    # data['aFiles'] = aFiles
    # data['plabels'] = plabels
    # data['plabels_10min'] = p_labels_10min
    # data['data_q'] = data_q

    if verbose:
        print '############ Subject: ', nsubject, ' ########### '
        print ' -- Features: ', '_'.join(feat_select)

    pp = PreprocessPipeline(remove_outliers=remove_outliers, standardize=standardize)
    pp.fit(XTRAIN, XTEST)

    if verbose:
        print ' -- Original dataset'
        print 'TRAIN:', XTRAIN.shape
        print 'ytrain', ytrain.shape

    if drop_nan:
        XTRAIN, ytrain, plabels_tr, ind_nan_tr = drop_data_quality_thr(XTRAIN, ytrain, plabels_tr, data_q_tr, qthr)
        ind_nan_tr = ind_nan_tr[~ind_nan_tr]

    else:
        ind_nan_tr = np.any(np.isnan(XTRAIN), axis=1)

    ind_nan_ts = np.any(np.isnan(XTEST), axis=1)
    XTEST[ind_nan_ts] = 0
    ytest = ytest.ravel()
    plabels_tr = plabels_tr.ravel()
    plabels_ts = plabels_ts.ravel()
    # XTEST, ytest, plabels_ts, ind_nan_ts = drop_data_quality_thr(XTEST, ytest, plabels_ts, data_q_ts, qthr)

    if verbose:
        print ' -- Removed data quality with treshold: ', qthr
        print 'TRAIN :', XTRAIN.shape
        print 'ytrain:', ytrain.shape
        print 'XTEST :', XTEST.shape
        print 'ytest:', ytest.shape

    XTRAIN = pp.transform(XTRAIN)
    XTEST = pp.transform(XTEST)

    if remove_covariates:
        l_feat_remove = load_removed_features(nsubject, feat_select)
        # l_feat_remove_all = load_removed_features(nsubject, ['stat_spectral_sp_entropy_mfj_corr'])
        # l_feat_remove += l_feat_remove_all
        XTRAIN, aFeatNames_tr, ind_remove = remove_features_by_name(XTRAIN, aFeatNames_tr, l_feat_remove)
        XTEST, aFeatNames_ts, ind_remove = remove_features_by_name(XTEST, aFeatNames_ts, l_feat_remove)

        if verbose:
            print '-- Removed features with covariate shift: '
            print 'TRAIN :', XTRAIN.shape
            print 'XTEST :', XTEST.shape

    data_tr['X'] = XTRAIN
    data_tr['y'] = ytrain
    data_tr['aFeatNames'] = aFeatNames_tr
    data_tr['plabels'] = plabels_tr
    data_tr['ind_nan'] = ind_nan_tr

    data_ts['X'] = XTEST
    data_ts['y'] = ytest
    data_ts['plabels'] = plabels_ts
    data_ts['aFeatNames'] = aFeatNames_ts
    data_ts['ind_nan'] = ind_nan_ts
    # data_tr = [XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr]
    # data_ts = [XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts]
    return data_tr, data_ts


def ismember(a, b):
    """Find values in an array a that are present in array b
    The return variable is of size a

    :param a array
    :param b array
    :return res array of size a
    """
    res = np.zeros((len(a)), dtype=np.bool)
    for i, val in enumerate(a):
        if val in b:
            res[i] = 1

    return res


def create_unique_key_param_patient(patient, param):
    param_for_patient = get_params_for_patient(patient, param)
    s = '_'.join('{0}_{1}'.format(k, v) for k, v in param_for_patient.items())
    clf_param_key = str(patient) + '_' + s
    return clf_param_key


def get_params_for_patient(nsubject, params):
    param_for_patient = dict()
    for name, val in params.iteritems():
        if '{0}_'.format(nsubject) in name[0:2]:
            param_for_patient[name[2:]] = val

    return param_for_patient


def get_params_for_specific_clf(param_for_patient, params_clf):

    param_for_clf = dict()
    for name, val in param_for_patient.iteritems():
        # print name, val
        if name in params_clf:
            param_for_clf[name] = val

    return param_for_clf


def remove_features_by_name(X, aFeatNames, features_to_remove):

    ind_to_remove = np.zeros((1, len(aFeatNames)), dtype=np.int)

    aFeatNamesNew = list()
    for i, s in enumerate(aFeatNames):
        if s in features_to_remove:
            ind_to_remove[0, i] = 1
        else:
            aFeatNamesNew.append(s)

    ind_to_remove = ind_to_remove.ravel()
    ind_to_remove = np.where(ind_to_remove > 0)[0]
    X = np.delete(X, ind_to_remove, axis=1)
    return X, aFeatNamesNew, ind_to_remove


def select_features_by_name(X, aFeatNames, features_to_select):

    ind_to_select = np.zeros((1, len(aFeatNames)), dtype=np.int)

    aFeatNamesNew = list()
    for i, s in enumerate(aFeatNames):
        if s in features_to_select:
            ind_to_select[0, i] = 1
            aFeatNamesNew.append(s)

    ind_to_select = ind_to_select.ravel()
    ind = np.where(ind_to_select > 0)[0]
    X = X[:, ind]

    return X, aFeatNamesNew, ind_to_select


def select_feature_group(X, aFeatNames, feature_group, verbose=False):

    # if type(feature_group) == list:
    #     feature_group = feature_group.split('__')
    # elif type(feature_group) == int:
    #     feature_group = [key for key, val in d_map_feat_names_int.iteritems() if val in feature_group]
    feature_group = feature_group.split('__')

    pattern = ''
    if 'stat' in feature_group:
        pattern += '|mean|std|skewness|kurtosis|hjort'
    if 'spectral' in feature_group:
        pattern += '|e_delta|e_theta|e_alpha|e_beta|e_low_gamma|e_high_gamma|energy_tot|spectrum_slope|energy_'
    if 'sp_entropy' in feature_group:
        pattern += '|_H_sh|_H_sh_knn|_H_sh_ann'
    if 'mfj' in feature_group:
        pattern += '|-H27|-c1|-c2|-c3|-c4|-H_j_'
    if 'corr' in feature_group:
        pattern += '|corr_|eigenvalue_'
    if 'wav_entropy' in feature_group:
        pattern += '|EnAxN|EnDxN'

    if pattern == '':
        raise Exception('Select feature group: Unknown group')

    # print pattern
    pattern = pattern.replace('||', '|')
    if pattern[0] == '|':
        pattern = pattern[1:]

    # print pattern
    p = re.compile(pattern, re.IGNORECASE)

    feat_to_select = filter(lambda s: p.search(s), aFeatNames)

    # check found features
    if verbose:
        for s in aFeatNames:
            if s in feat_to_select:
                print 'SELECT', s
            else:
                print s

    X, aFeatNames, ind_to_select = select_features_by_name(X, aFeatNames, feat_to_select)
    return X, aFeatNames, ind_to_select


def get_cv_groups_folds_for_all_p(d_data_train, k):
    """
    Create k-fold group cross-validation
    Return list of itrn, itst indicies
    Make sure that labels are randomized
    """

    acv = dict()
    for key, df in d_data_train.iteritems():
        assert isinstance(df, pd.DataFrame)
        skf = GroupKFold(n_splits=k)
        y = df['ytrain'].get_values().astype(int)
        p = df['plabels'].get_values().astype(int)
        p = rename_groups_random(p)
        # df['plabels'] = p

        itr, its = list(), list()
        for train, test in skf.split(y.copy(), y, p):
            itr.append(train)
            its.append(test)

        cv = zip(itr, its)
        acv[key] = cv

    return acv


def repeated_slcv_for_p_save_computation(estimator, n_splits, nr_repeat, df, param,
                                         key, y_hat_valid_precomputed, auc_valid_precomputed):

    # print 'running ', key
    param_for_patient = get_params_for_patient(key, param)
    clf_param_key = create_unique_key_param_patient(key, param)

    if clf_param_key not in y_hat_valid_precomputed:
        out = repeated_slcv_for_patient(estimator=estimator, n_splits=n_splits, nr_repeat=nr_repeat, df=df,
                                        param_for_patient=param_for_patient)

        auc_avg_r = out[0]
        yhat_valid = out[1]
    else:
        yhat_valid = y_hat_valid_precomputed[clf_param_key]
        auc_avg_r = auc_valid_precomputed[clf_param_key]

    return auc_avg_r, yhat_valid


def repeated_slcv_for_patient(estimator, n_splits, nr_repeat, df, param_for_patient):
    """
    :param estimator
    :param n_splits
    :param nr_repeat
    :param df pandas data frame with data
    :param param_for_patient (extract params for this specific patient (given by the key)
    """

    assert isinstance(df, pd.DataFrame)
    afeatnames = [col for col in df.columns if col not in ['ytrain', 'plabels']]
    XTRAIN = df[afeatnames].get_values()
    ytrain = df['ytrain'].get_values()
    p = df['plabels'].get_values()

    skf = GroupKFold(n_splits=n_splits)

    yhat_valid_rk = np.zeros((XTRAIN.shape[0], nr_repeat))
    # yhat_valid_ranked_rk = np.zeros((XTRAIN.shape[0], Rinner))
    auc_cv = np.zeros((n_splits, nr_repeat))

    # repeat inner cross-validation and save predicted labels
    for r_cv in range(0, nr_repeat):
        p = rename_groups_random(p)

        for ifold, (itrn, itst) in enumerate(skf.split(XTRAIN, ytrain, p)):
            Xtr, ytr = XTRAIN[itrn, :], ytrain[itrn]
            Xts, yts = XTRAIN[itst, :], ytrain[itst]

            #  select parameters for clf
            # print estimator.get_params()
            param_for_clf = get_params_for_specific_clf(param_for_patient, estimator.get_params())
            # print param_for_clf
            estimator.set_params(**param_for_clf)

            # prediction
            estimator.fit(Xtr, ytr)
            yhat = estimator.predict_proba(Xts)
            yhat = yhat[:, 1]

            yhat_valid_rk[itst, r_cv] = yhat
            # yhat_valid_ranked_rk[itst, r_cv] = probability_calibration(yhat)

            auc = compute_roc_auc_score_label_safe(yts, yhat)
            auc_cv[ifold, r_cv] = auc

    # if verbose:
    #     print auc_cv.shape
    #     print 'patient {0}: auc_cv: '.format(key), auc_cv
    #     print auc_cv.mean(axis=1)

    # y_hat_valid_precomputed[clf_param_key] = yhat_valid_rk
    # d_yhat_valid_ranked_to_save_computation[clf_param_key] = yhat_valid_ranked_rk
    # print yhat_valid_rk.shape

    auc_mean = auc_cv.mean(axis=1)
    # print auc_mean
    # auc_hat_valid_precomputed = auc_single_cv

    return auc_mean, yhat_valid_rk


def compute_auc_cv_for_all_p(df_data_train, d_yhat_predicted, k, hyperparam, probability_calib_method):
    """
    :param df_data_train pandas data frame with training data (X, y, plabels)
    :param d_yhat_predicted predicted labels during cross validation
    :param k number cross-validation folds
    :param hyperparam hyperparameters (used to get predicted labels from d_yhat_predicted)
    :param probability_calib_method
    """

    auc_cv = np.zeros((k, 1))
    CV_all_p = get_cv_groups_folds_for_all_p(df_data_train, k)

    for ifold in range(0, k):

        yhat_valid_all_p = 0
        ytrue_all_p = 0

        # get the predicted labels from each patient and add them to one vector
        for key, df in df_data_train.iteritems():

            ytr = df['ytrain'].get_values()
            clf_param_key = create_unique_key_param_patient(key, hyperparam)
            y_hat = d_yhat_predicted[clf_param_key]

            cv = CV_all_p[key]
            itst = cv[ifold][1]

            y_hat = y_hat[itst, :]

            # print y_hat.shape
            for i in range(0, y_hat.shape[1]):
                y_hat[:, i] = probability_calibration(y_hat[:, i], probability_calib_method)

            y_hat = y_hat.mean(axis=1)

            y_hat = probability_calibration(y_hat, probability_calib_method)
            y_hat = y_hat.ravel()

            yhat_valid_all_p = y_hat if key == 1 else np.hstack((yhat_valid_all_p, y_hat))
            ytrue_all_p = ytr[itst] if key == 1 else np.hstack((ytrue_all_p, ytr[itst]))

        auc = compute_roc_auc_score_label_safe(ytrue_all_p, yhat_valid_all_p)
        auc_cv[ifold] = auc

    auc_all_p = auc_cv.mean(axis=1)
    return auc_all_p


def rename_groups_random(p):
    """Rename group names.
    This is because KFoldGroup starts splits based on group names and splits are always the same.
    To make random splits, this function renames groups.
    e.g.
    [1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4 5 5 5 5]
    [5 5 5 5 4 4 4 4 3 3 3 3 1 1 1 1 2 2 2 2]
    """

    p_unique = np.unique(p, return_index=False)
    p_unique_shuffled = shuffle(p_unique)

    # print p_unique
    # print p_unique_shuffled

    p_renamed = p.copy()

    for i in range(0, len(p_unique)):
        ind = np.where(p == p_unique[i])[0]
        p_renamed[ind] = p_unique_shuffled[i]

    assert max(p) == max(p_renamed)
    assert min(p) == min(p_renamed)

    return p_renamed


def insert_pathol_to_normal_random_keep_order(X, y, p):
    """
    Insert pathological records inside normal ones such that order of normal and pathological is kept the same.
    e.g normal [1, 2, 3, 4, 5]
    pathol [6, 7]
    results [1, 2, 6, 3, 7, 4, 5] - pathol are inserted at random but hte order is the same

    :param X:
    :param y:
    :param p:
    :return:
    """

    # if we want to shuffle pathological records
    # p[y == 1] = rename_groups_random(p[y == 1])

    p_0 = p[y == 0]
    p_1 = p[y == 1]

    val_0 = min(p_0) - 1
    val_1 = min(p_1) - 1

    nr_p = len(np.unique(p))

    p_sanity_check = list()
    ind_for_resampling = list()

    place_to_insert_pathological = random.sample(xrange(min(p_0), max(p_0)), len(np.unique(p_1)))
    # place_to_insert_pathological = range(min(p_0), max(p_0), len(np.unique(p_1)))
    # print place_to_insert_pathological

    for i in range(1, nr_p + 1):

        if i not in place_to_insert_pathological:
            val_0 += 1
            val = val_0
        else:
            val_1 += 1
            val = val_1

        ind = np.where(p == val)[0]
        for j in ind:
            p_sanity_check.append(p[j])
            ind_for_resampling.append(j)
        # p_sanity_check.append(p[ind])
        # ind_for_resampling.append(ind)
        # print i, val, ind, p[ind]

    p_sanity_check = np.asarray(p_sanity_check).ravel()
    ind_for_resampling = np.asarray(ind_for_resampling).ravel()

    # print p_sanity_check.ravel()
    # print ind_for_resampling.ravel()

    # this should be the same
    assert sum(p_sanity_check != p[ind_for_resampling]) == 0

    p = p[ind_for_resampling].copy()
    y = y[ind_for_resampling].copy()
    X = X[ind_for_resampling, :].copy()
    return X, y, p, ind_for_resampling


def compute_roc_auc_score_label_safe(ytrue, ypred):

    if sum(ytrue == 1) == 0 or sum(ytrue == 0) == 0:
        return 0.5
    else:
        return roc_auc_score(ytrue, ypred)


def StratifiedKFoldPLabels(y, plabels, k=3):
    """Stratified K-Folds P-labels cross validation iterator.

    Provides train/test indices to split data in train test sets.
    The data split is stratified and the same values of plabels are not in train/test set

    :param y array of class labels
    :param plabels array of labels for patients (or segments)
    :param k number of cross validation folds
    """

    assert(len(y) == len(plabels))

    p, ind = np.unique(plabels, return_index=True)
    y_unique_p = y[ind]

    # print plabels
    # print p, ind
    # print 'y_unique:', y_unique_p, y_unique_p.shape

    # cv = StratifiedKFold(y_unique_p, n_folds=k, shuffle=True)
    # cv = KFold(n=len(y_unique_p), n_folds=k, shuffle=True)

    # import warnings
    # warnings._show_warning('Using ordinary KFOLD', Warning, 'utils.py', 1)

    skf = StratifiedKFold(n_splits=k, shuffle=True)
    # skf = KFold(n_splits=k, shuffle=True)

    itr, its = list(), list()
    for train, test in skf.split(y_unique_p.copy(), y_unique_p):
        # print("%s %s" % (train, test))
        # print len(train), len(test)
        itr.append(train)
        its.append(test)

    cv = zip(itr, its)

    k_est_in = [len(tr) / float(len(ts)) for tr, ts in cv]

    # for tr, ts in cv:
    #     print tr, ts
    #     print plabels, p[tr]
    #     print ismember(plabels, p[tr])

    # make sure we are not mixing the labels
    itrn = [np.where(ismember(plabels, p[tr]))[0] for tr, ts in cv]
    itst = [np.where(ismember(plabels, p[ts]))[0] for tr, ts in cv]

    # print itrn
    # print itst

    cv = zip(itrn, itst)

    # check the division based on labels
    if sum([sum(ismember(plabels[tr], plabels[ts])) for tr, ts in cv]) > 0:
        raise Exception('The same labels are in the training set and in the test set!')

    k_est = [len(tr) / float(len(ts)) for tr, ts in cv]

    if not np.isclose(np.average(k_est_in), np.average(k_est), atol=1e-01):
        print 'mean in:', np.average(k_est_in)
        print 'mean out:', np.average(k_est)
        raise Exception('Something wrong with data split!')

    return cv


def _handle_zeros_in_scale(scale, copy=True):
    ''' Makes sure that whenever scale is zero, we handle it correctly.

    This happens in most scalers when we have constant features.'''

    # if we are fitting on 1D arrays, scale might be a scalar
    if np.isscalar(scale):
        if scale == .0:
            scale = 1.
        return scale
    elif isinstance(scale, np.ndarray):
        if copy:
            # New array to avoid side-effects
            scale = scale.copy()
        scale[scale == 0.0] = 1.0
        return scale


class TimeSeriesSplitGroupSafe(_BaseKFold):
    """Time Series cross-validator

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.

    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 1.

    Examples
    --------
    # >>> from sklearn.model_selection import TimeSeriesSplit
    # >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    # >>> y = np.array([1, 2, 3, 4])
    # >>> tscv = TimeSeriesSplit(n_splits=3)
    # >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE
    # TimeSeriesSplit(n_splits=3)
    # >>> for train_index, test_index in tscv.split(X):
    # ...    print("TRAIN:", train_index, "TEST:", test_index)
    # ...    X_train, X_test = X[train_index], X[test_index]
    # ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]

    Notes
    -----
    The training set has size ``i * n_samples // (n_splits + 1)
    + n_samples % (n_splits + 1)`` in the ``i``th split,
    with a test set of size ``n_samples//(n_splits + 1)``,
    where ``n_samples`` is the number of samples.
    """

    def __init__(self, n_splits=3):
        super(TimeSeriesSplitGroupSafe, self).__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """

        if groups is not None:
            # find all indices that are at the beginning of a group
            groups_unique = np.unique(groups)
            possible_test_start = [np.where(i == groups)[0][0] for i in np.nditer(groups_unique)]
            possible_test_start = np.asarray(possible_test_start)

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        indices = np.arange(n_samples)
        test_size = (n_samples // n_folds)
        test_starts = range(test_size + n_samples % n_folds,
                            n_samples, test_size)

        if groups is not None:
            # find all possible starts that are closest to predefined test_starts
            test_starts = [possible_test_start[np.abs(possible_test_start - i).argmin()]
                           for i in test_starts]

        for test_start in test_starts:
            yield (indices[:test_start],
                   indices[test_start:test_start + test_size])


class PreprocessPipeline:
    """
    Standarize and outliers

    """

    def __init__(self, remove_outliers=True, standardize=True):

        self.ow = OutliersWinsorization()
        self.sc = StandardScalerDdof()
        self.remove_outliers = remove_outliers
        self.standardize = standardize

    def fit(self, Xtrain, Xtest=None, drop_nan=True):

        if Xtest is None:
            XALL = Xtrain.copy()
        else:
            XALL = np.vstack((Xtrain, Xtest)).copy()

        if drop_nan:
            XALL = drop_nan_single(XALL)

        XALL = self.ow.fit_transform(XALL)
        self.sc.fit(XALL)

    def transform(self, X):
        X = X.copy()

        if self.remove_outliers is True:
            X = self.ow.transform(X)

        if self.standardize is True:
            X = self.sc.transform(X)

        return X

# def preprocess_pipeline(Xtr, ytr, Xts, feat_names, plabels, data_q, verbose=True):
#
#     # NESMIM dat do pipeline!!!
#     # jinak se StandardScaler fituje na data s outliers
#
#     if verbose is True:
#         print 'Original dataset'
#         print 'TRAIN:', Xtr.shape
#         print 'TEST :', Xts.shape
#         print 'ytrain', ytr.shape
#
#     # Xtr, feat_names, ind_remove = preprocess_feat_nan_zeros(Xtr, feat_names)
#     # Xts = Xts[:, ~ind_remove]
#
#     if verbose is True:
#         print '\nRemoved features with NaNs and zeros'
#         print 'TRAIN:', Xtr.shape
#         print 'TEST :', Xtr.shape
#
#     if verbose is True:
#         print 'XALL: ', XALL.shape
#
#     thr = 10
#     Xtr, ytr, plabels = drop_data_quality_thr(Xtr, ytr, plabels, data_q, thr)
#     # Xtr, ytr, plabels = drop_nan(Xtr, ytr, plabels)
#
#     XALL = ow.fit_transform(XALL)
#     Xtr = ow.transform(Xtr)
#     Xts = sc.transform(Xts)
#
#     sc.fit(XALL)
#     Xtr = sc.transform(Xtr)
#     Xts = sc.transform(Xts)
#
#     if verbose is True:
#         print '\nTransformed and removed data quality with treshold: ', thr
#         print 'TRAIN :', Xtr.shape
#         print 'ytrain:', ytr.shape
#         print 'plabels', plabels.shape
#
#     return Xtr, ytr, Xts, feat_names, plabels, data_q


class StandardScalerDdof(BaseEstimator, TransformerMixin):
    """
    This is modified StandardScaler to provide an unbiased estimator of variance.

    Standardize features by removing the mean and scaling to unit variance

    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    with_mean : boolean, True by default
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : boolean, True by default
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    copy : boolean, optional, default True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    Attributes
    ----------
    scale_ : ndarray, shape (n_features,)
        Per feature relative scaling of the data.

        .. versionadded:: 0.17
           *scale_* is recommended instead of deprecated *std_*.

    mean_ : array of floats with shape [n_features]
        The mean value for each feature in the training set.

    var_ : array of floats with shape [n_features]
        The variance for each feature in the training set. Used to compute
        `scale_`

    n_samples_seen_ : int
        The number of samples processed by the estimator. Will be reset on
        new calls to fit, but increments across ``partial_fit`` calls.

    See also
    --------
    :func:`sklearn.preprocessing.scale` to perform centering and
    scaling without using the ``Transformer`` object oriented API

    :class:`sklearn.decomposition.RandomizedPCA` with `whiten=True`
    to further remove the linear correlation across features.
    """

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy

        self.mean_ = None
        self.var_ = None
        self.scale_ = None
        self.n_samples_seen_ = None

    @property
    @deprecated("Attribute ``std_`` will be removed in 0.19. Use ``scale_`` instead")
    def std_(self):
        return self.scale_

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.

        __init__ parameters are not touched.
        """

        # Checking one attribute is enough, becase they are all set together
        # in partial_fit
        if hasattr(self, 'scale_'):
            del self.scale_
            del self.mean_
            del self.var_

    def fit(self, X, y=None):
        """Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y: Passthrough for ``Pipeline`` compatibility.
        """

        # Reset internal state before fitting
        self._reset()
        # return self.partial_fit(X, y)
        self.mean_ = X.mean(axis=0)
        self.var_ = X.var(axis=0, ddof=1)

        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None

        return self

    def transform(self, X, y=None, copy=None):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr', copy=copy,
                        ensure_2d=False, warn_on_dtype=True,
                        estimator=self, dtype=FLOAT_DTYPES)

        if self.with_mean:
            X -= self.mean_
        if self.with_std:
            X /= self.scale_

        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy

        X = np.asarray(X)
        if copy:
            X = X.copy()
        if self.with_std:
            X *= self.scale_
        if self.with_mean:
            X += self.mean_

        return X


def _parallel_fit_estimator(estimator, X, y):
    """Private function used to fit an estimator within a job."""
    estimator.fit(X, y)
    return estimator


class OutliersWinsorization(BaseEstimator, TransformerMixin):
    """
    Features are are  preprocessed for outliers (Winsorization in the interval [Qi - d*IQR, Qi + d*IQR
    where Qi is the i-th quartile and IQR = Q3 - Q1)

    Typical values for d are 1.5 and 3
    """

    def __init__(self, d=3, test_var_1502=0):
        self.d = d
        self.n = 0
        self.lo = np.array((1, 1))
        self.hi = np.array((1, 1))
        self.test_var_1502 = test_var_1502
        # self.q25 = 0
        # self.q75 = 0

    def fit(self, X, y=None):
        """ Find low/high intervals for outliers detection"""

        # self.cnt += 1
        # print 'OW'
        # print self.test_var_1502, self.cnt

        self.n = X.shape[1]
        self.lo = np.zeros((self.n, 1))
        self.hi = np.zeros((self.n, 1))

        # self.q25 = np.zeros((self.n, 1))
        # self.q75 = np.zeros((self.n, 1))

        for i in range(0, X.shape[1]):
            q25 = np.percentile(X[:, i], 25, interpolation='linear')
            q75 = np.percentile(X[:, i], 75, interpolation='linear')
            r = q75 - q25
            self.lo[i] = q25 - self.d * r
            self.hi[i] = q75 + self.d * r

            # self.q25[i] = q25
            # self.q75[i] = q75

        return self

    def transform(self, X, y=None):
        """Winsorization of outliers"""

        assert X.shape[1] == len(self.lo)
        assert X.shape[1] == len(self.hi)

        Xnew = X.copy()
        for i in range(0, Xnew.shape[1]):
            ind = Xnew[:, i] < self.lo[i]
            Xnew[ind, i] = self.lo[i]

            ind = Xnew[:, i] > self.hi[i]
            Xnew[ind, i] = self.hi[i]

        return Xnew


class VotingClassifierRank(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Soft Voting/Majority Rule classifier for unfitted estimators.

    """

    def __init__(self, estimators, voting='rank', n_jobs=1):
        self.estimators = estimators
        self.named_estimators = dict(estimators)
        self.voting = voting
        self.n_jobs = n_jobs

        self.le_ = LabelEncoder()
        self.classes_ = 0
        self.estimators_ = []

    def fit(self, X, y, sample_weight=None):
        """ Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('rank', 'median_centered', 'soft', 'none'):
            raise ValueError("Voting must be 'soft', 'median_centered', or 'rank'; got (voting=%r)"
                             % self.voting)

        if self.estimators is None or len(self.estimators) == 0:
            raise AttributeError('Invalid `estimators` attribute, `estimators`'
                                 ' should be a list of (string, estimator)'
                                 ' tuples')

        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_
        self.estimators_ = []

        transformed_y = self.le_.transform(y)

        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_parallel_fit_estimator)(clone(clf), X, transformed_y)
            for _, clf in self.estimators)

        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """

        check_is_fitted(self, 'estimators_')

        ''' not implemented '''

        # if self.voting == 'soft':
        #     maj = np.argmax(self.predict_proba(X), axis=1)
        #
        # elif self.voting == 'median_centered':
        #     pass
        # else:  # 'rank' voting
        #     predictions = self._predict(X)
        #     maj = np.apply_along_axis(lambda x:
        #                               np.argmax(np.bincount(x)),
        #                               axis=1,
        #                               arr=predictions.astype('int'))
        #
        # maj = self.le_.inverse_transform(maj)

        return self._predict_proba

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.estimators_])

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        if self.voting == 'hard':
            raise AttributeError("predict_proba is not available when"
                                 " voting=%r" % self.voting)

        check_is_fitted(self, 'estimators_')

        if self.voting == 'soft' or self.voting is 'none':
            avg = np.average(self._collect_probas(X), axis=0)

        else:  # rank or median_centered
            y_pred = self._collect_probas(X)
            for i in range(0, y_pred.shape[0]):
                y_pred[i, :, 0] = probability_calibration(y_pred[i, :, 0], self.voting)
                y_pred[i, :, 1] = 1 - y_pred[i, :, 0]

            avg = y_pred.mean(axis=0)
            avg[:, 1] = 1 - avg[:, 0]

        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilities calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_samples, n_classifiers]
            Class labels predicted by each classifier.
        """
        check_is_fitted(self, 'estimators_')
        if self.voting == 'soft':
            return self._collect_probas(X)
        else:
            return self._predict(X)

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support"""
        if not deep:
            return super(VotingClassifierRank, self).get_params(deep=False)
        else:
            out = super(VotingClassifierRank, self).get_params(deep=False)
            out.update(self.named_estimators.copy())
            for name, step in six.iteritems(self.named_estimators):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.estimators_]).T


class FeatureSelectGroup(BaseEstimator, SelectorMixin):
    """
    Select feature group based on name
    """

    def __init__(self, feature_names=None, group=None, verbose=False):

        self.feature_names = feature_names
        self.group = group
        self.selected_mask = None
        self.verbose = verbose

    def _get_support_mask(self):
        return self.selected_mask > 0

    def set_feature_names(self, feat_names):
        # print feat_names
        self.feature_names = feat_names

    def fit(self, X, y):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like, shape (n_samples,)
            The target values (integers that correspond to classes in
            classification, real numbers in regression).

        Returns
        -------
        self : object
            Returns self.
        """
        if self.feature_names is None:
            raise Exception('Feature names must be defined!!!')

        _, _, self.selected_mask = select_feature_group(X.copy(), self.feature_names, self.group, self.verbose)

        return self


def probability_calibration(y, method='rank'):

    if method == 'rank':
        return rankdata_normalized(y)
    elif method == 'none' or method is None:
        return y
    elif method == 'median_centered':

        y -= np.median(y)
        y /= 2
        y += 0.5

        y[y < 0] = 0
        y[y > 1] = 1

        return y

    else:
        raise Exception('Improper probability calibration algorithm set')


def rankdata_normalized(y):
    y = rankdata(y)
    y /= float(len(y))
    return y


def auc_score_mannwhitneyu(ytrue, x):

    # n1 = sum(ytrue == 1)
    # n2 = sum(ytrue == 0)
    # u, prob = mannwhitneyu(ytrue, x)
    #
    # auc = u / (n1 * n2)
    # # auc = auc if auc >= 0.5 else 1 - auc
    # return auc

    n1 = sum(ytrue == 1)
    n2 = sum(ytrue == 0)
    rank_value = rankdata(x)

    rank_sum = sum(rank_value[ytrue == 1])
    u_value = rank_sum - (n1 * (n1 + 1)) / 2
    auc = u_value / (n1 * n2)
    # if auc < 0.50:
    #     auc = 1.0 - auc
    return auc


def probability_aggregate(ytrue, ypredicted, plabels, method='average'):

    up = np.unique(plabels)
    y_agg = np.zeros(up.shape)
    y_true_agg = np.zeros(up.shape)

    if method == 'hmean':
        method_ = hmean
    elif method == 'average':
        method_ = np.average
    elif method == 'gmean':
        method_ = gmean
    elif method == 'median':
        method_ = np.median
    else:
        raise Exception('Not implemented')

    for i in range(0, len(up)):

        # only one for given group!
        ytmp = ytrue[plabels == up[i]]
        assert len(np.unique(ytmp)) == 1
        y_true_agg[i] = ytmp[0]
        # print ypredicted[plabels == up[i]]
        y_agg[i] = method_(ypredicted[plabels == up[i]])

    return y_true_agg, y_agg


def dropBadFiles(df):
    print df.shape

    s1 = ['1_137_0.mat', '1_138_0.mat', '1_139_0.mat', '1_140_0.mat', '1_141_0.mat', '1_142_0.mat', '1_166_0.mat', '1_167_0.mat', '1_168_0.mat', '1_169_0.mat', '1_170_0.mat',
          '1_171_0.mat', '1_266_0.mat', '1_267_0.mat', '1_268_0.mat', '1_269_0.mat', '1_270_0.mat', '1_271_0.mat', '1_303_0.mat', '1_304_0.mat', '1_305_0.mat', '1_306_0.mat',
          '1_307_0.mat', '1_308_0.mat', '1_397_0.mat', '1_398_0.mat', '1_399_0.mat', '1_400_0.mat', '1_401_0.mat', '1_402_0.mat', '1_412_0.mat', '1_413_0.mat', '1_414_0.mat',
          '1_415_0.mat', '1_416_0.mat', '1_417_0.mat', '1_481_0.mat', '1_482_0.mat', '1_483_0.mat', '1_484_0.mat', '1_485_0.mat', '1_486_0.mat', '1_520_0.mat', '1_521_0.mat',
          '1_522_0.mat', '1_523_0.mat', '1_524_0.mat', '1_525_0.mat', '1_585_0.mat', '1_586_0.mat', '1_587_0.mat', '1_588_0.mat', '1_589_0.mat', '1_590_0.mat', '1_621_0.mat',
          '1_622_0.mat', '1_623_0.mat', '1_624_0.mat', '1_625_0.mat', '1_626_0.mat', '1_703_0.mat', '1_704_0.mat', '1_705_0.mat', '1_706_0.mat', '1_707_0.mat', '1_708_0.mat',
          '1_784_0.mat', '1_785_0.mat', '1_786_0.mat', '1_787_0.mat', '1_788_0.mat', '1_789_0.mat', '1_855_0.mat', '1_856_0.mat', '1_857_0.mat', '1_858_0.mat', '1_859_0.mat',
          '1_860_0.mat', '1_985_0.mat', '1_986_0.mat', '1_987_0.mat', '1_988_0.mat', '1_989_0.mat', '1_990_0.mat', '1_1015_0.mat', '1_1016_0.mat', '1_1017_0.mat', '1_1018_0.mat',
          '1_1019_0.mat', '1_1020_0.mat', '1_1099_0.mat', '1_1100_0.mat', '1_1101_0.mat', '1_1102_0.mat', '1_1103_0.mat', '1_1104_0.mat', '1_1129_0.mat', '1_1130_0.mat', '1_1131_0.mat',
          '1_1133_0.mat', '1_1134_0.mat']

    s2 = ['2_69_0.mat', '2_70_0.mat', '2_71_0.mat', '2_72_0.mat', '2_399_0.mat', '2_400_0.mat', '2_401_0.mat', '2_402_0.mat', '2_439_0.mat', '2_440_0.mat', '2_441_0.mat',
          '2_442_0.mat', '2_443_0.mat', '2_444_0.mat', '2_452_0.mat', '2_453_0.mat', '2_454_0.mat', '2_455_0.mat', '2_456_0.mat', '2_531_0.mat', '2_532_0.mat', '2_533_0.mat',
          '2_534_0.mat', '2_763_0.mat', '2_764_0.mat', '2_765_0.mat', '2_766_0.mat', '2_767_0.mat', '2_768_0.mat', '2_1427_0.mat', '2_1428_0.mat', '2_1429_0.mat', '2_1430_0.mat',
          '2_1431_0.mat', '2_1432_0.mat', '2_1603_0.mat', '2_1604_0.mat', '2_1605_0.mat', '2_1763_0.mat', '2_1764_0.mat', '2_1765_0.mat', '2_1766_0.mat', '2_1767_0.mat', '2_2119_0.mat',
          '2_2120_0.mat', '2_2121_0.mat', '2_2122_0.mat', '2_2123_0.mat', '2_2124_0.mat']

    s3 = ['3_799_0.mat', '3_901_0.mat', '3_902_0.mat', '3_903_0.mat', '3_904_0.mat', '3_905_0.mat', '3_906_0.mat',
          '3_1105_0.mat', '3_1106_0.mat', '3_1107_0.mat', '3_1108_0.mat']

    for item in s1:
        df = df.drop(df[df.file == item].index)

    for item in s2:
        df = df.drop(df[df.file == item].index)

    for item in s3:
        df = df.drop(df[df.file == item].index)

    print 'Final shape:' + str(df.shape)

    return df