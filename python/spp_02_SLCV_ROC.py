
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, l1_min_c
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.cross_validation import StratifiedKFold, LeavePLabelOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb

from matplotlib.pyplot import *
from utils import get_from_all_10_min, StratifiedKFoldPLabels
from python.utils_learning import OutliersWinsorization

nsubject = 1

filename_tr = 'features/sp2016_feat_train_{0}_stat_20160914'.format(nsubject)
XTRAIN, ytrain, aFeatNames, aFiles_tr, plabels = get_from_all_10_min(filename_tr)

# clf = xgb.XGBClassifier()
# parameters = {
#     'max_depth': range(2, 31, 5),
#     'gamma': [0, 2],
#     # 'n_estimators': [10, 25, 50, 100],
#     'learning_rate': [0.01, 0.1, 0.2],
#     # 'subsample': [0.5, 1]
#     # 'learning_rate': [0.01, 0.05]
# }

# clf = DecisionTreeClassifier(class_weight='balanced', max_depth=1)
clf = LogisticRegression(class_weight='balanced', penalty='l1', C=.25)

print 'Subject: ', nsubject
print 'Original dataset'
print XTRAIN.shape
print ytrain.shape
print list(aFeatNames)

mask = np.any(np.isnan(XTRAIN), axis=1)
XTRAIN = XTRAIN[~mask]
ytrain = ytrain[~mask].ravel()
plabels = plabels[~mask]

print 'Removed NaNs'
print XTRAIN.shape
print ytrain.shape
print plabels.shape

# outlier removal
ow = OutliersWinsorization()
XTRAIN = ow.fit_transform(XTRAIN)

sc = StandardScaler()
XTRAIN = sc.fit_transform(XTRAIN)

# K = [5, 10, 20, 50]
K = [15]
R = 3  # repeat cross-validation

cnt = -1

for k in K:

    list_roc_auc = list()
    for r in range(0, R):
        # cnt_fold = -1
        # cv = cross_validation.StratifiedKFold(y, n_folds=k_feat)

        X, y, plabels = shuffle(XTRAIN, ytrain, plabels)
        cv = StratifiedKFoldPLabels(y, plabels=plabels, k=k)

        print 'FOLDS:', k, 'REPEAT: ', r

        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)
        all_tpr = []

        for i, (itrn, itst) in enumerate(cv):

            clf.fit(X[itrn, :], y[itrn])
            yhat = clf.predict_proba(X[itst, :])
            # print yhat
            fpr, tpr, thresholds = metrics.roc_curve(y[itst], yhat[:, 1], pos_label=1)

            # yhat = clf.predict(Xt[itst, :])
            # print metrics.confusion_matrix(y[itst], yhat)

            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = metrics.auc(fpr, tpr)
            list_roc_auc.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

        print 'mean AUC: {0}'.format(np.mean(list_roc_auc))

        # mean_tpr /= len(cv)
        # mean_tpr[-1] = 1.0
        # mean_auc = metrics.auc(mean_fpr, mean_tpr)
        # plt.plot(mean_fpr, mean_tpr, 'k_feat--',
        #          label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
        # sys.exit(0)

