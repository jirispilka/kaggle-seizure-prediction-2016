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

from utils import rename_groups_random, load_features_and_preprocess, VotingClassifierRank, FeatureSelectGroup, d_map_feat_names_int
from spp_ut_settings import Settings
import sys

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

if len(sys.argv) > 1:
    nsubject = int(sys.argv[1])
else:
    nsubject = 1


# feat_select = ['stat']
# feat_select = ['spectral']
# feat_select = ['sp_entropy']
# feat_select = ['mfj']
# feat_select = ['corr']
# feat_select = ['wav_entropy']
# feat_select = ['stat', 'spectral']
feat_select = ['stat', 'spectral', 'sp_entropy']
# feat_select = ['sp_entropy', 'spectral']
# feat_select = ['spectral']
# feat_select = ['sp_entropy', 'corr']
# feat_select = ['wav_entropy']
# feat_select = ['sp_entropy','corr']
# feat_select = ['stat', 'spectral', 'sp_entropy', 'wav_entropy', 'mfj', 'corr']

settings = Settings()
# print settings

K = [settings.kfoldCV]
R = settings.repeatCV

d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select, settings=settings)
XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                              d_tr[4], d_tr[5], d_tr[6]
XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                         d_ts[4], d_ts[5], d_ts[6]


''' Naive Bayes '''
# clfnb = GaussianNB()
# clf = Pipeline([
#     ('fs', SelectKBest(f_classif)),
#     ('clf', clfnb)
# ])
# parameters = {'fs__k': range(2, 20, 2)}

from vowpalwabbit.sklearn_vw import VWClassifier
from sklearn.preprocessing import LabelEncoder
XTRAIN = XTRAIN.astype(np.float32)
ytrain = ytrain.astype(np.int8)
# le_ = LabelEncoder()
# le_.fit([0, 1])
ytrain[ytrain == 0] = -1
clf = VWClassifier(bfgs=False, quiet=True)

parameters = {'bfgs': [False]}

# clf.fit(XTRAIN, ytrain)

# print model.score(XTRAIN, ytrain)
# print model._predict_proba_lr(XTRAIN)
# print model.decision_function(XTRAIN)
# model.score(X_test, y_test))

# sys.exit()


print "Parameters:"
for param_name in sorted(parameters.keys()):
    print param_name, parameters[param_name]


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

        X, y, p = shuffle(XTRAIN, ytrain, plabels_tr)
        p = rename_groups_random(p)
        skf = GroupKFold(n_splits=k)

        # X, y, p = XTRAIN, ytrain, plabels_tr
        # # X, y = XTRAIN, ytrain
        # skf = KFold(n_splits=k, shuffle=True)

        print 'FOLDS:', k, 'REPEAT: ', r,
        grid_search = GridSearchCV(clf, parameters, verbose=1, n_jobs=1, cv=skf, scoring='roc_auc')
        # grid_search.fit(X, y)
        grid_search.fit(X, y, p)

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))

        import warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        print 'Grid scores'
        # print grid_search.cv_results_

        for score in grid_search.grid_scores_:
            print score
            d = score[0].copy()
            res = score[2]
            # d['repetition'] = r
            # d['folds'] = k

            for rr in res:
                cnt += 1
                d['result'] = rr
                d['index'] = cnt
                df_results.loc[cnt] = d

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

# import time
# stime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
# sname = "{0}_patient_{1}_results_SLCV_Voting_{2}k_{3}.res".format(stime, nsubject, K, '_'.join(feat_select))
# # sname = "{0}_patient_{1}_results_SLCV_LR_{2}k.res".format(stime, nsubject, K)
# df_results.to_pickle(sname)


# a_subject = [1, 2, 3]
# # a_feat_select = ['stat', 'spectral', 'sp_entropy', 'mfj', 'corr', 'wav_entropy']
# # a_subject = [2, 3]
# a_feat_select = ['stat', 'spectral', 'sp_entropy']
# s_iter = list(itertools.product(a_feat_select, a_subject))
# print s_iter
# for s in s_iter:
#     run_batch([str(s[0])], s[1])
