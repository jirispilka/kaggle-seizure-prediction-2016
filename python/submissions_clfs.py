from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
import xgboost as xgb

from utils import FeatureSelectGroup, VotingClassifierRank, get_params_for_specific_clf, get_params_for_patient

from spp_ut_settings import Settings
settings = Settings()

prob_calib_alg = settings.prob_calib_alg

# v93
clf_nb = Pipeline([
    ('gr', FeatureSelectGroup()),
    ('fs', SelectKBest(f_classif)),
    ('nb', GaussianNB())
])

clf_lr = Pipeline([
    ('gr', FeatureSelectGroup()),
    ('lr', LogisticRegression(class_weight='balanced', penalty='l1', n_jobs=1))
])

clf_xgb = Pipeline([
    ('gr', FeatureSelectGroup()),
    ('xgb', xgb.XGBClassifier(gamma=0, colsample_bytree=0.3, subsample=0.9, seed=2016))
])

# P1
est = [('clfa', clone(clf_nb)), ('clfb', clone(clf_lr)), ('clfc', clone(clf_xgb))]
clf1 = VotingClassifierRank(estimators=est, voting=prob_calib_alg)

# P2
est = [('clfa', clone(clf_nb)), ('clfb', clone(clf_nb)), ('clfc', clone(clf_lr)), ('clfd', clone(clf_xgb))]
clf2 = VotingClassifierRank(estimators=est, voting=prob_calib_alg)

# P3
est = [('clfa', clone(clf_lr)), ('clfb', clone(clf_nb)), ('clfc', clone(clf_lr)), ('clfd', clone(clf_lr))]
clf3 = VotingClassifierRank(estimators=est, voting=prob_calib_alg)

a_clf_v93 = dict()
a_clf_v93[1] = clf1
a_clf_v93[2] = clf2
a_clf_v93[3] = clf3

parameters = {'1_clfa__fs__k': [10],
              '1_clfa__gr__group': ['spectral'],
              '1_clfb__lr__C': [0.016],
              '1_clfb__gr__group': ['sp_entropy'],
              '1_clfc__xgb__max_depth': [2],
              '1_clfc__xgb__n_estimators': [10],
              '1_clfc__xgb__scale_pos_weight': [7.9],
              '1_clfc__gr__group': ['spectral'],
              '2_clfa__fs__k': [10],
              '2_clfa__gr__group': ['stat'],
              '2_clfb__fs__k': [10],
              '2_clfb__gr__group': ['sp_entropy'],
              '2_clfc__lr__C': [0.006],
              '2_clfc__gr__group': ['spectral'],
              '2_clfd__xgb__n_estimators': [1],
              '2_clfd__xgb__max_depth': [90],
              '2_clfd__xgb__scale_pos_weight': [14.76],
              '2_clfd__gr__group': ['sp_entropy'],
              '3_clfa__lr__C': [0.014],
              '3_clfa__gr__group': ['stat'],
              '3_clfb__fs__k': [28],
              '3_clfb__gr__group': ['stat'],
              '3_clfc__lr__C': [1],
              '3_clfc__gr__group': ['stat'],
              '3_clfd__lr__C': [.5],
              '3_clfd__gr__group': ['spectral'],
              }

param_grid_v93 = ParameterGrid(parameters)
for key, pipe in a_clf_v93.iteritems():
    param_for_patient = get_params_for_patient(key, param_grid_v93[0])
    param_for_clf = get_params_for_specific_clf(param_for_patient, pipe.get_params())
    # print param_for_clf
    pipe.set_params(**param_for_clf)
