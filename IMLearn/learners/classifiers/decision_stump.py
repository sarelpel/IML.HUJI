from __future__ import annotations
from typing import Tuple, NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from IMLearn.metrics import misclassification_error
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        best_thr = np.inf
        best_feature_ind = np.inf
        best_loss = np.inf
        best_sign = 0
        for j in range(X.shape[1]):
            for sign in [1, -1]:
                thr, loss = self._find_threshold(X[:, j], y, sign)
                if loss < best_loss:
                    best_loss = loss
                    best_thr, best_sign, best_feature_ind = thr, sign, j

        self.sign_ = best_sign
        self.threshold_ = best_thr
        self.j_ = best_feature_ind

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        col = X[:, self.j_]
        y_pred = np.ndarray(X.shape[0])
        y_pred[col >= self.threshold_] = self.sign_
        y_pred[col < self.threshold_] = -self.sign_
        return y_pred

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """

        best_loss = 1
        thr = np.NINF
        for sample in values:
            y_pred = (values >= sample).astype(int)
            y_pred[y_pred == 1] = sign
            y_pred[y_pred == 0] = -sign

            loss = self.decision_stump_error(labels, y_pred, np.abs(labels))
            if loss < best_loss:
                best_loss = loss
                thr = sample
        if thr == np.min(values):
            thr = np.NINF
        return thr, best_loss

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        y_pred = self._predict(X)
        return self.decision_stump_error(y, y_pred, np.abs(y))


    def decision_stump_error(self, y_true: np.ndarray, y_pred: np.ndarray, d: np.ndarray) -> float:
        y_error = np.sign(y_true) != np.sign(y_pred)
        return np.sum(d @ y_error)
