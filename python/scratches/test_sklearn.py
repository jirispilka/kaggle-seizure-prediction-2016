
from discretization import MDLP
from sklearn.datasets import load_iris

from sklearn.ensemble import partial_dependence
from mlxtend.classifier import stacking_classification

iris = load_iris()
X = iris.data
y = iris.target
mdlp = MDLP(shuffle=False)
conv_X = mdlp.fit_transform(X, y)
# print conv_X