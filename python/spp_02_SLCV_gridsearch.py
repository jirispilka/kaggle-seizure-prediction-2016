import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, l1_min_c
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures

# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from utils import rename_groups_random, load_features_and_preprocess, VotingClassifierRank, FeatureSelectGroup
from spp_ut_feat_selection import JMIFeatureSelector
from spp_ut_settings import Settings
from discretization import MDLP
import sys
import itertools

import lasagne
from lasagne import layers
from lasagne.updates import adam
from nolearn.lasagne import NeuralNet

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

if len(sys.argv) > 1:
    nsubject = int(sys.argv[1])
else:
    nsubject = 3


# def run_batch(feat_select, nsubject):

# nsubject = 1

# feat_select = ['stat']
# feat_select = ['spectral']
feat_select = ['sp_entropy']
# feat_select = ['mfj']
# feat_select = ['corr']
# feat_select = ['wav_entropy']
# feat_select = ['stat', 'spectral']
# feat_select = ['stat', 'spectral', 'sp_entropy']
# feat_select = ['sp_entropy', 'spectral']
# feat_select = ['spectral']
# feat_select = ['sp_entropy', 'corr']
# feat_select = ['wav_entropy']
# feat_select = ['sp_entropy','corr']
# feat_select = ['stat', 'spectral', 'sp_entropy', 'wav_entropy', 'mfj', 'corr']

settings = Settings()
print settings

K = [settings.kfoldCV]
R = settings.repeatCV

# settings.remove_covariate_shift = False

# XTRAIN, ytrain, aFeatNames, aFiles_tr, plabels, data_q = load_features('train', nsubject, feat_select)
# XTEST, ytest, aFeatNames_ts, dummy4, dummy5, dummy3 = load_features('test', nsubject, feat_select)

d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select, settings=settings)
XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                              d_tr[4], d_tr[5], d_tr[6]
XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                         d_ts[4], d_ts[5], d_ts[6]

# '''RANDOM OVERSAMPLING'''
# from imblearn.over_sampling import SMOTE
# from imblearn.over_sampling import RandomOverSampler
#
# ro = RandomOverSampler()
# XTRAIN, ytrain, plabels_tr = ro.fit_sample(XTRAIN, ytrain, plabels_tr)
# print 'oversampling', XTRAIN.shape
#
# ''' FEATURE SELECTION '''
# n_select_feat = 100
# jmi = JMIFeatureSelector(k_feat=n_select_feat)
# jmi.fit(XTRAIN, ytrain)
# XTRAIN = jmi.fit_transform(XTRAIN, ytrain)
#
# aFeatSeleced = list()
# for i in jmi.selected_indicies_:
#     aFeatSeleced.append(aFeatNames_tr[int(i)])
#
# print '\nSELECTED FEATURES\n'
# for i in range(0, n_select_feat):
#     print '{0:6} {1}'.format(jmi.selected_indicies_[i], aFeatSeleced[i])
#
# aFeatNames_tr = aFeatSeleced
# print 'TRAIN :', XTRAIN.shape
# print 'ytrain:', ytrain.shape

# clf_fs = SelectFromModel(estimator=LogisticRegression(class_weight='balanced', penalty='l1', C=0.5))
# XTRAIN = clf_fs.fit_transform(XTRAIN, ytrain)
# print 'After FEATURE SELECTION:', XTRAIN.shape

''' XGB '''
# fratio = sum(ytrain == 0) / float(sum(ytrain == 1))
# print 'ratio', fratio
# clf = xgb.XGBClassifier(max_depth=1, n_estimators=60, min_child_weight=1, gamma=0,
#                         learning_rate=0.1, colsample_bytree=0.15, subsample=1,
#                         reg_lambda=1, reg_alpha=1, nthread=15,
#                         objective='binary:logistic', seed=2016, silent=1,
#                         scale_pos_weight=1)
#
# parameters = {
#     # 'max_depth': [1, 2, 3, 4, 5],
#     # 'max_depth': range(1, 6, 1),
#     # 'n_estimators': range(5, 31, 5) + [40, 50, 60, 70, 100, 150],
#     # 'n_estimators': range(5, 27, 5),
#     # 'min_child_weight': range(1, 6, 1),
#     # 'gamma': [0,.01, .02, .03, 1 , 2, 50, 100],
#     # 'colsample_bytree': np.arange(0.05, 0.2, 0.01),
#     # 'colsample_bytree': np.hstack((np.arange(0.01, 0.151, 0.01), np.arange(0.06, 0.21, 0.02))),
#     # 'colsample_bytree': np.arange(0.02, 0.51, 0.02),
#     # 'subsample': np.arange(0.1, 1.01, 0.1),
#     # 'learning_rate': [0.01, 0.1, 0.2],
#     # 'reg_alpha': [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1, 50, 100],
#     # 'reg_lambda': [1e-4, 0.1, 1, 10, 50, 100]
#     # 'max_delta_step': np.arange(0.2, 2, 0.2)
#     'scale_pos_weight': np.arange(1, 15, 1)
# }

'''RF'''
# clf = RandomForestClassifier(class_weight='balanced', random_state=2016)
# parameters = {
#     'max_depth': [1, 2],
#     # 'max_leaf_nodes': [int(x) for x in np.exp2(range(1, 8))],
#     'n_estimators': range(5, 51, 5)
#     # 'n_estimators': range(5, 21, 5) + range(20, 101, 20) + [150]
#     # 'min_samples_split': range(2, 20, 2)
# }

''' SVM '''
# # # # cs = l1_min_c(XTRAIN, ytrain, loss='log') * np.logspace(0, 2)
# clf = SVC(kernel='poly', probability=True, class_weight='balanced', degree=3)
# # # clf = LinearSVC(class_weight='balanced', loss='hinge')
# parameters = {
#     # 'penalty': ['l1', 'l2'],
#     # 'C': np.exp2(range(-12, 3, 2)),
#     'C':  0.0001 * np.logspace(0, 3, 10),
#     # 'C': [0.1, 1, 10, 100, 1000],
#     # 'gamma': np.exp2(range(-20, -2, 4)),
#     # 'degree': [2, 3, 4],
#     # 'C': cs
#     # 'jmi__k_feat': [5, 10, 15, 20],
# }

''' Logistic regression '''
w = 'balanced'
clflr = LogisticRegression(class_weight=w, penalty='l2', n_jobs=1)
clf = Pipeline([
    # ('jmi', JMIFeatureSelector(k_feat=50)),
    ('pca', PCA(n_components=2)),
    ('clf', clflr)
])

# print np.logspace(0, 2.5, 12)
# cs = 0.005 * np.logspace(0, 2.5, 12)
# cs = [0.006, 0.008, 0.010, 0.012, 0.015, 0.02, 0.04, 0.06, 0.12, 0.2, 1, 2]
# cs = [0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.014, 0.016, 0.2, 0.3, 0.5, 1]
# cs = [0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.014, 0.016, 0.2, 0.3, 0.5, 1, 2, 5]
# cs = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.02, 0.04, 0.06, 0.12, 0.2, 1]
# cs = np.arange(0.00001, 0.00201, 0.00005)

parameters = {
    'clf__penalty': ['l1', 'l2'],
    'clf__C': [1, 10],
    # 'clf__C': np.exp2(range(-7, 4))  # stat features
    # 'clf__C': np.hstack((np.arange(0.01, 0.031, 0.001))),  # P1
    # 'clf__C': np.hstack((np.arange(0.005, 0.031, 0.001))),  # stat P1
    # 'clf__C': np.hstack((np.arange(0.006, 0.051, 0.002))),  # stat P2, P3
    # 'clf__C': np.hstack((np.arange(0.01, 0.501, 0.02))),  # spectral P1
    'pca__n_components': range(2, 20, 8),
}

''' SGD Classifier '''
# cs = l1_min_c(XTRAIN, ytrain, loss='log') * np.logspace(0, 2)
# clf = SGDClassifier(class_weight='balanced', penalty='l1')
# parameters = {
#     'penalty': ['l1', 'l2', 'elasticnet'],
#     # 'alpha': np.exp2(range(-8, 1))  # stat features
#     'alpha': cs
# }

''' Naive Bayes '''
# p = None
# clfnb = GaussianNB(priors=p)
# clf = Pipeline([
#     # ('jmi', JMIFeatureSelector(k_feat=20)),
#     # ('pca', PCA(n_components=2)),
#     # ('pf', PolynomialFeatures()),
#     ('fs', SelectKBest(f_classif)),
#     ('clf', clfnb)
# ])
# # parameters = {'fs__k': range(2, 11, 2), 'jmi__k_feat': [10, 20]}
# # parameters = {'fs__k': range(2, min(40, XTRAIN.shape[1]), 2) + range(min(40, XTRAIN.shape[1]), min(100, XTRAIN.shape[1]), 10)}
# parameters = {'fs__k': range(2, 20, 2)}
# # parameters = {'fs__k': range(2, 20, 2), 'pf__degree': [1, 2, 3]}
# # parameters = {'pca__n_components': range(1, 20, 1)}
# # parameters = {'jmi__k_feat': range(5, 40, 5)}
# # poly = PolynomialFeatures(2)


''' Multinomial NB '''

# # mdlp = MDLP(shuffle=False)
# # print XTRAIN.shape
# # XTRAIN = mdlp.fit_transform(XTRAIN, ytrain)
#
# # print np.unique(XTRAIN)
# # sys.exit()
#
# for i in range(0, XTRAIN.shape[1]):
#     nr, bins = np.histogram(XTRAIN[:, i], bins=2)
#     XTRAIN[:, i] = np.digitize(XTRAIN[:, i], bins=bins)
#     # print np.unique(XTRAIN[:, i])
#
# # print XTRAIN
#
# clf = MultinomialNB()
# # clf = BernoulliNB()
# # parameters = {'alpha': np.exp2(range(-12, 12, 4))}
#
# # clf = Pipeline([
# #     # ('mdlp', MDLP(shuffle=False)),
# #     ('nb', BernoulliNB())
# # ])
# # # np.exp2(range(-12, 12, 4)
# parameters = {
#               'alpha': np.exp2(range(-10, 12, 2)),
#               # 'nb__binarize': np.arange(-2.00, 2.01, 0.25),
#               }
#
# # parameters = {'nb__alpha': [0.062500, 1]}

''' KNN '''
# # sm = SMOTE(kind='regular')
# # # sm.fit(XTRAIN, ytrain)
# # # # XTRAIN, ytrain = sm.transform(XTRAIN, ytrain)
# # print XTRAIN.shape
#
# knn = KNeighborsClassifier()
# clf = Pipeline([
#     # ('jmi', JMIFeatureSelector()),
#     # ('sm', SMOTE(kind='regular')),
#     ('clf', knn)
# ])
# # parameters = {'jmi__k_feat': [10, 20, min(XTRAIN.shape[1], 50), min(XTRAIN.shape[1], 100)],
#
# #               'clf__n_neighbors': range(20, 200, 20)}
#
# # parameters = {'clf__n_neighbors': range(1, 10, 1)}
# # parameters = {'clf__n_neighbors': range(20, 201, 20)}
# parameters = {'clf__n_neighbors': range(20, 200, 20) + range(200, 401, 50) + [500]}
# # parameters = {'clf__n_neighbors': range(200, 501, 100)}
# # parameters = {'clf__n_neighbors': range(150, 500, 50)}

# K=[20]
''' Neural network '''
# layers0 = [('input', layers.InputLayer),
#     ('hidden', layers.DenseLayer),
#     ('output', layers.DenseLayer),
#  ]
#
# clf = NeuralNet(
#     layers=layers0,
#     # layer parameters:
#     input_shape=(None, XTRAIN.shape[1]),
#     hidden_num_units=200,  # number of units in 'hidden' layer
#     output_nonlinearity=lasagne.nonlinearities.softmax,
#     output_num_units=2,  # 10 target values for the digits 0, 1, 2, ..., 9
#
#     # optimization method:
#     update=adam,
#     update_learning_rate=0.0004,
#     objective_l2=0.001,
#
#     # max_epochs=20,
#     verbose=0,
# )
#
# parameters = {
#     'max_epochs': [10, 20, 30, 40, 50]
# }

'''----------------------------- VOTING P1 ---------------------------'''
# clf1 = Pipeline([
#     ('gr', FeatureSelectGroup(feature_names=aFeatNames_tr)),
#     ('fs', SelectKBest(f_classif)),
#     ('nb', GaussianNB())
# ])
#
# clf2 = Pipeline([
#     ('gr', FeatureSelectGroup(feature_names=aFeatNames_tr)),
#     ('lr', LogisticRegression(class_weight='balanced', penalty='l1', n_jobs=1))
# ])
#
# fratio = sum(ytrain == 0) / float(sum(ytrain == 1))
# print 'fratio: ', fratio
# xgb = xgb.XGBClassifier(max_depth=2, n_estimators=10, min_child_weight=1, gamma=0, learning_rate=0.1,
#                         colsample_bytree=0.3, subsample=0.9, objective='binary:logistic', seed=2016,
#                         scale_pos_weight=fratio)
#
# clf3 = Pipeline([
#     ('gr', FeatureSelectGroup(feature_names=aFeatNames_tr)),
#     ('xgb', xgb)
# ])
#
# # clf3 = Pipeline([
# #     ('gr', FeatureSelectGroup(feature_names=aFeatNames_tr)),
# #     ('knn', KNeighborsClassifier())
# # ])
#
# # est = [('clf3', clf3)]
# # est = [('clf1', clf1), ('clf2', clf2)]
# est = [('clf1', clf1), ('clf2', clf2), ('clf3', clf3)]
# clf = VotingClassifierRank(estimators=est, voting='rank')
#
# # cs = [0.002, 0.004, 0.006, 0.008, 0.010, 0.016, 0.02, 0.06, 0.12, 0.2]
# parameters = {
#               'clf1__fs__k': [10],
#               'clf1__gr__group':  ['spectral'],
#               'clf2__lr__C':  [0.016],
#               'clf2__gr__group': ['sp_entropy'],
#               'clf3__xgb__max_depth': [2],
#               'clf3__xgb__n_estimators': [10],
#               'clf3__gr__group': ['spectral'],
#               # # 'clf3__lr__C': cs,
#               # # 'clf3__xgb__colsample_bytree': np.arange(0.1, 1, 0.1),
#               # # 'clf3__xgb__subsample': np.arange(0.1, 1, 0.1),
#               # # 'clf3__xgb__max_depth': [1, 2],
#               # # 'clf3__xgb__n_estimators': range(5, 36, 5),
#               # # 'clf3__gr__group': ['stat', 'spectral', 'sp_entropy'],
#               # # 'clf3__knn__n_neighbors': range(20, 201, 20),
#               }

'''----------------------------- VOTING P2 ---------------------------'''
# clf1 = Pipeline([
#     ('gr', FeatureSelectGroup(feature_names=aFeatNames_tr)),
#     ('fs', SelectKBest(f_classif)),
#     ('nb', GaussianNB())
# ])
#
# clf2 = Pipeline([
#     ('gr', FeatureSelectGroup(feature_names=aFeatNames_tr)),
#     ('fs', SelectKBest(f_classif)),
#     ('nb', GaussianNB())
# ])
#
# clf3 = Pipeline([
#     ('gr', FeatureSelectGroup(feature_names=aFeatNames_tr)),
#     ('lr', LogisticRegression(class_weight='balanced', penalty='l1', n_jobs=1))
# ])
#
# fratio = sum(ytrain == 0) / float(sum(ytrain == 1))
# xgb = xgb.XGBClassifier(max_depth=1, n_estimators=20, min_child_weight=1, gamma=0, learning_rate=0.1,
#                         colsample_bytree=0.3, subsample=0.9, objective='binary:logistic', seed=2016, scale_pos_weight=1)
#
# clf4 = Pipeline([
#     ('gr', FeatureSelectGroup(feature_names=aFeatNames_tr)),
#     ('xgb', xgb)
# ])
#
# # est = [('clf1', clf1), ('clf2', clf2), ('clf3', clf3)]
# est = [('clf1', clf1), ('clf2', clf2), ('clf3', clf3), ('clf4', clf4)]
# clf = VotingClassifierRank(estimators=est, voting='rank')
#
# # parameters = {'clf1__fs__k': range(2, 31, 4),
# #               'clf1__gr__group': ['sp_entropy'],
# #               'clf2__fs__k':  range(2, 31, 4),
# #               'clf2__gr__group': ['spectral'],
# #               # 'clf3__fs__k': range(2, 11, 2),
# #               # 'clf3__gr__group': ['stat', 'spectral', 'sp_entropy', 'mfj', 'corr', 'wav_entropy'],
# #               }
#
# # cs = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.01, 0.012, 0.016, 0.02, 0.06]
# # cs = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.02, 0.04, 0.06, 0.12, 0.2, 1]
# parameters = {
#               'clf1__fs__k': [10],
#               'clf1__gr__group':  ['stat'],
#               'clf2__fs__k':  [10],
#               'clf2__gr__group': ['sp_entropy'],
#               'clf3__lr__C': [0.006],
#               'clf3__gr__group': ['spectral'],
#               'clf4__xgb__n_estimators':  [90],
#               'clf4__xgb__max_depth': [1],
#               'clf4__gr__group': ['sp_entropy'],
#               # # 'clf3__xgb__colsample_bytree': np.arange(0.1, 1, 0.1),
#               # # 'clf3__xgb__max_depth': [1, 2],
#               # # 'clf3__xgb__n_estimators': range(5, 36, 5),
#               # # 'clf2__gr__group': ['stat', 'spectral', 'sp_entropy'],
#               }

'''--------------------- VOTING P3 ---------------------------'''
# # clf1 = Pipeline([
# #     ('gr', FeatureSelectGroup(feature_names=aFeatNames_tr)),
# #     ('fs', SelectKBest(f_classif)),
# #     ('nb', GaussianNB())
# # ])
#
#
# clf1 = Pipeline([
#     ('gr', FeatureSelectGroup(feature_names=aFeatNames_tr, verbose=False)),
#     ('lr', LogisticRegression(class_weight='balanced', penalty='l1', n_jobs=1))
# ])
#
# clf2 = Pipeline([
#     ('gr', FeatureSelectGroup(feature_names=aFeatNames_tr)),
#     ('fs', SelectKBest(f_classif)),
#     ('nb', GaussianNB())
# ])
#
# clf3 = Pipeline([
#     ('gr', FeatureSelectGroup(feature_names=aFeatNames_tr)),
#     ('lr', LogisticRegression(class_weight='balanced', penalty='l1', n_jobs=1))
# ])
#
# clf4 = Pipeline([
#     ('gr', FeatureSelectGroup(feature_names=aFeatNames_tr)),
#     ('lr', LogisticRegression(class_weight='balanced', penalty='l1', n_jobs=1))
# ])
#
# # est = [('clf4', clf4)]
# # est = [('clf1', clf1), ('clf2', clf2), ('clf3', clf3)]
# est = [('clf1', clf1), ('clf2', clf2), ('clf3', clf3), ('clf4', clf4)]
# # est = [('clf1', clf1), ('clf2', clf2), ('clf3', clf3), ('clf4', clf4), ('clf5', clf5)]
# clf = VotingClassifierRank(estimators=est, voting='rank')
#
# # cs = [0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.012, 0.016, 0.02, 0.06]
# # cs = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.02, 0.04, 0.06, 0.12, 0.2, 1]
# parameters = {'clf1__lr__C': [0.014],
#               'clf1__gr__group':  ['stat'],
#               'clf2__fs__k': [28],
#               'clf2__gr__group': ['stat'],
#               'clf3__lr__C': [1],
#               'clf3__gr__group': ['stat'],
#               'clf4__lr__C': [.5],
#               'clf4__gr__group': ['spectral'],
#               # # 'clf5__lr__C': cs,
#               # # 'clf5__gr__group': ['stat', 'spectral', 'sp_entropy'],
#               # # 'clf5__xgb__n_estimators':  range(20, 121, 40),
#               # # 'clf5__xgb__max_depth': range(1, 5, 1),
#               # # 'clf5__gr__group': ['stat', 'spectral', 'sp_entropy'],
#               # # 'clf3__xgb__colsample_bytree': np.arange(0.1, 1, 0.1),
#               # # 'clf3__xgb__max_depth': [1, 2],
#               # # 'clf3__xgb__n_estimators': range(5, 36, 5),
#               # # 'clf2__gr__group': ['stat', 'spectral', 'sp_entropy'],
#               }

#
# print "Parameters:"
# for param_name in sorted(parameters.keys()):
#     print param_name, parameters[param_name]


nr = np.sum([R*k for k in K])
nr *= np.prod([len(d) for d in parameters.itervalues()])
index = range(0, nr - 1)

l = parameters.keys()
# l.append('repetition')
# l.append('folds')
l.append('result')
l.append('index')
# df_results = pd.DataFrame(columns=l, index=index, dtype=float)
df_results = pd.DataFrame(columns=l, dtype=float)
cnt = -1

for k in K:
    for r in range(0, R):

        # X, y, p = shuffle(XTRAIN, ytrain, plabels_tr)
        X, y, p = XTRAIN, ytrain, plabels_tr
        p = rename_groups_random(p)
        skf = GroupKFold(n_splits=k)

        # X, y, p = XTRAIN, ytrain, plabels_tr
        # # X, y = XTRAIN, ytrain
        # skf = KFold(n_splits=k, shuffle=True)

        print 'FOLDS:', k, 'REPEAT: ', r,
        grid_search = GridSearchCV(clf, parameters, verbose=1, n_jobs=3, cv=skf, scoring='roc_auc')
        # grid_search.fit(X, y)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            grid_search.fit(X, y, p)

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        cv_res = grid_search.cv_results_
        for param, mean_s, std_s in zip(cv_res['params'], cv_res['mean_test_score'],
                                        cv_res['std_test_score']):

            print('mean: {0:2.5f}, std: {1:2.5f}, params: {2}'.format(mean_s, std_s, param))
            cnt += 1
            d = param.copy()
            d['result'] = mean_s
            d['index'] = cnt
            df_results.loc[cnt] = d

        # print df_results
        # # sys.exit()
        #
        # for score in grid_search.grid_scores_:
        #     print score
        #     # print score[0], score[1], score[2]
        #     d = score[0].copy()
        #     res = score[2]
        #     # d['repetition'] = r
        #     # d['folds'] = k
        #
        #     for rr in res:
        #         cnt += 1
        #         d['result'] = rr
        #         d['index'] = cnt
        #         df_results.loc[cnt] = d

                # df_results.insert(cnt, 'result', rr)
                # print df_results.loc[cnt]

        # for score in grid_search.grid_scores_:
        #     print score[2].mean(), score[2]

# if type(df_results['result'].loc[0]) == str:
#     df_results.drop(df_results.index[:1], inplace=True)
    # df_results.reset_index(inplace=True)
    # df_results.drop('index', inplace=True)

# print df_results.describe()
# print df_results.info()
# print df_results

# df_results['clf1__gr__group'] = df_results['clf1__gr__group'].map(d_map_feat_names_int).astype(int)
# df_results['clf2__gr__group'] = df_results['clf2__gr__group'].map(d_map_feat_names_int).astype(int)
# print df_results[parameters.keys()]

# print df_results['result'].groupby(parameters.keys()).median()
print df_results.groupby(parameters.keys(), as_index=False).agg({'result': ['median', 'std']})
# print df_results.groupby(['clf2__lr__C', 'clf1__lr__C', 'clf1__gr__group', 'clf2__gr__group']).median()

import time
stime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
sname = "{0}_patient_{1}_results_SLCV_KNN_{2}k_{3}.res".format(stime, nsubject, K, '_'.join(feat_select))
# sname = "{0}_patient_{1}_results_SLCV_LR_{2}k.res".format(stime, nsubject, K)
df_results.to_pickle(sname)


# a_subject = [1, 2, 3]
# # a_feat_select = ['stat', 'spectral', 'sp_entropy', 'mfj', 'corr', 'wav_entropy']
# # a_subject = [2, 3]
# a_feat_select = ['stat', 'spectral', 'sp_entropy']
# s_iter = list(itertools.product(a_feat_select, a_subject))
# print s_iter
# for s in s_iter:
#     run_batch([str(s[0])], s[1])
