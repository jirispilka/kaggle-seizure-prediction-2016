import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

from utils import load_features_and_preprocess
from spp_ut_settings import Settings
from spp_00_load_data import write_removed_features
import sys

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

if len(sys.argv) > 1:
    nsubject = int(sys.argv[1])
else:
    nsubject = 3

feat_select = ['stat']
# feat_select = ['spectral']
# feat_select = ['sp_entropy']
# feat_select = ['mfj']
# feat_select = ['corr']
# feat_select = ['wav_entropy']
# feat_select = ['stat', 'spectral', 'sp_entropy', 'mfj', 'corr']

settings = Settings()
print settings

settings.remove_covariate_shift = False

# d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select, settings=settings)
# XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
#                                                                               d_tr[4], d_tr[5], d_tr[6]
# XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
#                                                                          d_ts[4], d_ts[5], d_ts[6]

d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select, settings=settings)

XTRAIN, ytrain, aFeatNames_tr, plabels_tr = d_tr['X'], d_tr['y'], d_tr['aFeatNames'], d_tr['plabels']

# urceni covariates na trenovaci mnozine
n_half = int(XTRAIN.shape[0]/2)
XTEST = XTRAIN[0:n_half, :]
ytest = ytrain[0:n_half]

XTRAIN = XTRAIN[n_half:-1, :]
ytrain = ytrain[n_half:-1]

print '\n RUNNING DIVISION OF THE TRAIN SET !!! \n\n'
print 'TRAIN :', XTRAIN.shape
print 'ytrain:', ytrain.shape
print 'XTEST :', XTEST.shape
print 'ytest:', ytest.shape

# for p in cv:
#     print p
# print len(cv)
# sys.exit()

''' Logistic regression '''
# w = 'balanced'
# clf = LogisticRegression(class_weight=w, penalty='l1', n_jobs=1)
# parameters = {'C': np.hstack((np.arange(0.0095, 0.02, 0.0001), np.arange(0.02, 0.601, 0.005)))}
# parameters = {'C': [0.005, 0.0075, 0.01]}
# parameters = {'C': [0.005]}

clf = Pipeline([
    # ('rfe', RFE(estimator=LogisticRegression(class_weight='balanced', penalty='l1', C=0.01), n_features_to_select=2,
    ('rfe', RFE(estimator=LogisticRegression(class_weight='balanced', penalty='l1', C=0.001), n_features_to_select=2,
                step=0.1)),
    ('clf', LogisticRegression(class_weight='balanced', penalty='l1', n_jobs=1))
])

# parameters = {'clf__C': [0.005, 0.0075, 0.01]}
parameters = {'clf__C': [0.001, 0.01]}

K = 5
R = 1  # repeat cross-validation

auc_limit = 0.55
auc_hat = 1
step_remove = 1

# TODO
# TODO
# TODO
# TODO
print 'TRY TO SCORE ACCURACY'

X = np.vstack((XTRAIN, XTEST))
y = np.vstack((np.zeros((len(ytrain), 1)), np.ones((len(ytest), 1))))
print X.shape
print y.shape
y = y.ravel()

removed_features = list()

while auc_hat > auc_limit:

    feat_importance = np.zeros((1, len(aFeatNames_tr)))

    for r in range(0, R):

        cv = StratifiedKFold(n_splits=K, shuffle=True)
        # X, y = shuffle(X, y)

        print 'FOLDS:', K, 'REPEAT: ', r,
        grid_search = GridSearchCV(clf, parameters, verbose=0, n_jobs=1, cv=cv, scoring='roc_auc')
        grid_search.fit(X, y)

        ft = grid_search.best_estimator_.steps[0][1].ranking_
        # print ft
        # print grid_search.best_estimator_.steps[0][1].n_features_

        # ft = np.abs(grid_search.best_estimator_.coef_)
        # print feat_importance.shape, ft.shape
        feat_importance[0, :] += ft.ravel()

        auc_hat = grid_search.best_score_

        print("Best score: %0.3f" % grid_search.best_score_)
        # print("Best parameters set:")
        # best_parameters = grid_search.best_estimator_.get_params()
        # for param_name in sorted(parameters.keys()):
        #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
        #
        # print 'Grid scores'
        # for score in grid_search.grid_scores_:
        #     print score

    if auc_hat > auc_limit:
        feat_importance = feat_importance.ravel()
        feat_importance /= float(R)
        # print feat_importance
        ind = np.argsort(feat_importance)
        ind = ind.ravel()
        # ind = ind[::-1].ravel()

        # print '\n\n #### Sorted by importance'
        # for i in range(0, len(ind)):
        #     print '{0:30}: {1}'.format(aFeatNames_tr[ind[i]], feat_importance[ind[i]])

        X = np.delete(X, ind[0:step_remove], axis=1)
        for j in range(0, step_remove):
            print 'REMOVING: ', aFeatNames_tr[ind[j]]
            removed_features.append(aFeatNames_tr[ind[j]])
            # del aFeatNames[ind[j]]

        aFeatNames_tr = [s for i, s in enumerate(aFeatNames_tr) if i not in ind[0:step_remove]]
        print X.shape

# write_removed_features(nsubject, feat_select, removed_features)
