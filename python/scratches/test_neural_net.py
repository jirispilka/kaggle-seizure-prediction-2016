from utils import load_features_and_preprocess, compute_roc_auc_score_label_safe
from spp_ut_settings import Settings
from sklearn.metrics import roc_auc_score, roc_curve
import sys

from sklearn.utils import shuffle
from sklearn.feature_selection import SelectFromModel

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import pandas as pd

import lasagne
from lasagne import layers
from lasagne.updates import nesterov_momentum

from lasagne.layers import DenseLayer, Conv1DLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer, MaxPool1DLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adam
from lasagne.layers import get_all_params

from nolearn.lasagne import NeuralNet
from nolearn.lasagne import TrainSplit
from nolearn.lasagne import objective
from nolearn.lasagne import PrintLayerInfo

from nolearn.lasagne.visualize import draw_to_notebook
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from nolearn.lasagne.visualize import plot_conv_activity
from nolearn.lasagne.visualize import plot_occlusion
from nolearn.lasagne.visualize import plot_saliency

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)

if len(sys.argv) > 1:
    nsubject = int(sys.argv[1])
else:
    nsubject = 3


# feat_select = ['stat']
# feat_select = ['spectral']
feat_select = ['sp_entropy']
# feat_select = ['stat', 'spectral', 'sp_entropy']

settings = Settings()
print settings

K = [settings.kfoldCV]
R = settings.repeatCV

d_tr, d_ts = load_features_and_preprocess(nsubject, feat_select, settings=settings)
XTRAIN, ytrain, aFeatNames_tr, aFiles_tr, plabels_tr, data_q_tr, ind_nan_tr = d_tr[0], d_tr[1], d_tr[2], d_tr[3], \
                                                                              d_tr[4], d_tr[5], d_tr[6]
XTEST, ytest, aFeatNames_ts, aFiles_ts, plabels_ts, data_q_ts, ind_nan_ts = d_ts[0], d_ts[1], d_ts[2], d_ts[3], \
                                                                         d_ts[4], d_ts[5], d_ts[6]



X = XTRAIN
y = ytrain.astype(np.int32)

# X = X.reshape(
#     -1,  # number of samples, -1 makes it so that this number is determined automatically
#     1,  # 1 color channel, since images are only black and white
#     40,  # second image dimension (horizontal)
# )
#
# print X.shape, y.shape
#
# layers0 = [
#     # layer dealing with the input data
#     (InputLayer, {'shape': (None, X.shape[1], X.shape[2])}),
#
#     # first stage of our convolutional layers
#     (Conv1DLayer, {'num_filters': 32, 'filter_size': 5}),
#     # (Conv1DLayer, {'num_filters': 16, 'filter_size': 3}),
#     # (Conv1DLayer, {'num_filters': 16, 'filter_size': 2}),
#     (MaxPool1DLayer, {'pool_size': 2}),
#
#     # second stage of our convolutional layers
#     # (Conv1DLayer, {'num_filters': 16, 'filter_size': 5}),
#     # (Conv1DLayer, {'num_filters': 16, 'filter_size': 2}),
#     # (Conv1DLayer, {'num_filters': 16, 'filter_size': 2}),
#     # (MaxPool1DLayer, {'pool_size': 2}),
#
#     # two dense layers with dropout
#     (DenseLayer, {'num_units': 32}),
#     (DropoutLayer, {}),
#     (DenseLayer, {'num_units': 32}),
#
#     # the output layer
#     (DenseLayer, {'num_units': 2, 'nonlinearity': softmax}),
# ]
#
# net = NeuralNet(
#     layers=layers0,
#     max_epochs=100,
#
#     update=adam,
#     update_learning_rate=0.001,
#
#     objective_l2=0.005,
#
#     train_split=TrainSplit(eval_size=0.2),
#     verbose=1,
# )

layers0 = [('input', layers.InputLayer),
    ('hidden', layers.DenseLayer),
    ('output', layers.DenseLayer),
 ]

net = NeuralNet(
    layers=layers0,
    # layer parameters:
    # input_shape=(None, 28 * 28),
    input_shape=(None, X.shape[1]),
    hidden_num_units=200,  # number of units in 'hidden' layer
    output_nonlinearity=lasagne.nonlinearities.softmax,
    output_num_units=2,  # 10 target values for the digits 0, 1, 2, ..., 9
    custom_scores=[('auc0', lambda y_true, y_proba: compute_roc_auc_score_label_safe(y_true, y_proba[:, 0])),
                  ('auc1', lambda y_true, y_proba: compute_roc_auc_score_label_safe(y_true, y_proba[:, 1]))],

    # optimization method:
    update=adam,
    update_learning_rate=0.0004,
    # update_momentum=0.9,
    objective_l2=0.001,

    max_epochs=1000,
    train_split=TrainSplit(eval_size=0.3),
    verbose=1,
)

# net.initialize()
# layer_info = PrintLayerInfo()
# layer_info(net)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

net.fit(X_train, y_train)
plot_loss(net)

y_hat = net.predict_proba(X_test)

auc = compute_roc_auc_score_label_safe(y_test, y_hat[:, 1])

plt.figure(10)
fpr, tpr, dummy = roc_curve(y_test, y_hat[:, 1], pos_label=1)
plt.plot(fpr, tpr, color='k', lw=1, label='ROC feat %0.5f' % auc)
plt.grid()
plt.legend(loc=4)


plt.show()

