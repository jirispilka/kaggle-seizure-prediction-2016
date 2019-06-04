from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from feast import JMI
from utils import ismember


class JMIFeatureSelector(BaseEstimator, SelectorMixin):
    """
    JMI - Joint Mutual Information - Yang and Moody (1999)
    Feature selection
    """
    def __init__(self, k_feat=3):
        self.k_feat = k_feat
        self.selected_indicies_ = list()
        self.selected_mask = None

    def _get_support_mask(self):
        # JMIFeatureSelector can directly call on transform.

        # ind = range(0, len(self.selected_indicies_))
        # mask = ismember(self.selected_indicies_, ind)
        # mask = self.selected_indicies_
        # scores = _get_feature_importances(estimator)
        # return scores >= self.threshold_
        return self.selected_mask > 0

    def fit(self, X, y):
        """Fit the JMIFeatureSelector meta-transformer.
        Get indicies of selected features

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
        ind = JMI(X, y, self.k_feat)
        self.selected_indicies_ = [int(i) for i in ind]
        # print self.selected_indicies_

        ind = range(0, X.shape[1])
        self.selected_mask = ismember(ind, self.selected_indicies_)

        return self
