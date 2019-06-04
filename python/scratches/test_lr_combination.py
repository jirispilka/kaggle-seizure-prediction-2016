import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

npatol = 150
n1 = 1200
n2 = 2200
n3 = 2200

x1 = np.random.normal(0.17, 0.01, n1)
x2 = np.random.lognormal(0.05, .05, n2) - .8
x3 = np.random.exponential(0.05, n3)

xp1 = np.random.normal(0.25, 0.1, npatol)
xp2 = np.random.lognormal(0.05, .1, npatol) - .7
xp3 = np.random.exponential(0.1, npatol) + 0.1

x1 = np.hstack((xp1, x1))
x2 = np.hstack((xp2, x2))
x3 = np.hstack((xp3, x3))

y1 = np.zeros((len(x1), 1)).ravel()
y2 = np.zeros((len(x2), 1)).ravel()
y3 = np.zeros((len(x3), 1)).ravel()
y1[0:151] = 1
y2[0:151] = 1
y3[0:151] = 1

n1 = x1.shape[0]
n2 = x2.shape[0]
n3 = x3.shape[0]

print x1.shape, y1.shape

T1 = np.hstack((x1[:, np.newaxis], y1[:, np.newaxis]))
T2 = np.hstack((x2[:, np.newaxis], y2[:, np.newaxis]))
T3 = np.hstack((x3[:, np.newaxis], y3[:, np.newaxis]))
df1 = pd.DataFrame(data=T1, columns=['X', 'y'])
df2 = pd.DataFrame(data=T2, columns=['X', 'y'])
df3 = pd.DataFrame(data=T3, columns=['X', 'y'])

dall = dict()
dall[1] = df1
dall[2] = df2
dall[3] = df3

acolors = ['r', 'g', 'b']

y_hat_x = 0
y_all_p = 0

R = 1
N = n1+n2+n3
Y = np.zeros((N, R))
ind = np.zeros((N, 1))
ind[0:n1] = 1
ind[n1: n1+n2] = 2
ind[n1+n2: n1+n2+n3] = 3
ind = ind.ravel()

for r in range(0, R):

    for key, df in dall.iteritems():

        nsubject = key

        assert isinstance(df, pd.DataFrame)
        x = df['X'].get_values()
        y = df['y'].get_values()

        noise = np.random.normal(0, 0.001, len(y))
        x = x + abs(noise)
        x[x < 0] = 0
        x[x > 1] = 1

        plt.figure(10)
        plt.hist(x, 50, range=[0, 1], histtype='step', color=acolors[key-1], linewidth=2, label='{0}'.format(nsubject))
        plt.grid()
        plt.legend()

        plt.figure(11)
        plt.subplot(3, 1, nsubject)
        plt.hist(x[y == 1], 50, range=[0, 1], histtype='step', color='r', linewidth=2, label='1')
        plt.hist(x[y == 0], 50, range=[0, 1], histtype='step', color='b', linewidth=2, label='1')
        plt.title('patient: {0}'.format(nsubject))
        plt.grid()

        fraction_of_positives, mean_predicted_value = calibration_curve(y, x, n_bins=10)
        xtmp = np.arange(0, 1, 0.01)

        plt.figure(100+nsubject)
        plt.plot(xtmp, xtmp, 'k--')
        plt.plot(mean_predicted_value, fraction_of_positives, color='b', marker='x')
        plt.grid()
        plt.legend()

        y_hat_x = x if nsubject == 1 else np.hstack((y_hat_x, x))
        y_all_p = y if nsubject == 1 else np.hstack((y_all_p, y))

        Y[ind == nsubject, r] = x.ravel()

auc = roc_auc_score(y_all_p, y_hat_x)

''' ROC curve '''
plt.figure(12)
fpr, tpr, dummy = roc_curve(y_all_p, y_hat_x, pos_label=1)
plt.plot(fpr, tpr, 'r', lw=1, label='auc all = %0.5f' % (auc))
plt.title('ROC for all patients')
# plt.grid()
# plt.legend()

plt.figure(13)
plt.hist(y_hat_x, 50, range=[0, 1], histtype='step', color='k', linewidth=2, label='all')
plt.hist(y_hat_x[y_all_p == 0], 50, range=[0, 1], histtype='step', color='b', linewidth=2, label='all-0')
plt.hist(y_hat_x[y_all_p == 1], 50, range=[0, 1], histtype='step', color='r', linewidth=2, label='all-1')
plt.grid()
plt.legend()


''' LOGISTIC REGRESSION '''
# plt.figure(100)
# plt.plot(Y[y_all_p == 0, 0], Y[y_all_p == 0, 1], color='b', marker='.', linestyle='')
# plt.plot(Y[y_all_p == 1, 0], Y[y_all_p == 1, 1], color='r', marker='.', linestyle='')
# plt.grid()

clf = LogisticRegression(class_weight='balanced', penalty='l1')
parameters = {'C': [1e-3, 1, 100]}

grid_search = GridSearchCV(clf, parameters, verbose=1, n_jobs=1, cv=3, scoring='roc_auc')

y_hat_x = y_hat_x[:, np.newaxis]

grid_search.fit(Y, y_all_p)
print grid_search.best_estimator_.coef_

y_prob = grid_search.predict_proba(Y)
y_prob = y_prob[:, 1]

plt.figure(14)
plt.hist(y_prob, 50, range=[0, 1], histtype='step', color='k', linewidth=2, label='all')
plt.hist(y_prob[y_all_p == 0], 50, range=[0, 1], histtype='step', color='b', linewidth=2, label='all-0')
plt.hist(y_prob[y_all_p == 1], 50, range=[0, 1], histtype='step', color='r', linewidth=2, label='all-1')
plt.grid()
plt.legend()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))

for score in grid_search.grid_scores_:
    print score

plt.figure(12)
auc = roc_auc_score(y_all_p, y_prob)
fpr, tpr, dummy = roc_curve(y_all_p, y_prob, pos_label=1)
plt.plot(fpr, tpr, 'k', lw=1, label='auc all lr = %0.5f' % (auc))
plt.legend()
plt.grid()


fraction_of_positives, mean_predicted_value = calibration_curve(y_all_p, y_hat_x, n_bins=10)
fraction_of_positives2, mean_predicted_value2 = calibration_curve(y_all_p, y_prob, n_bins=10)
xtmp = np.arange(0, 1, 0.01)

plt.figure(1500)
plt.plot(xtmp, xtmp, 'k--')
plt.plot(mean_predicted_value, fraction_of_positives, color='b', label='original', marker='x')
plt.plot(mean_predicted_value2, fraction_of_positives2, color='k', label='meta lr', marker='o')
plt.grid()
plt.legend()
# ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
#          label="%s" % (name,))
#
# ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
#          histtype="step", lw=2)

plt.show()