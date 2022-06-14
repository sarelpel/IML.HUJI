from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from IMLearn.metrics.loss_functions import mean_square_error



class RidgeRegression(BaseEstimator):
    """
    Ridge Regression Estimator

    Solving Ridge Regression optimization problem
    """

    def __init__(self, lam: float, include_intercept: bool = True) -> RidgeRegression:
        """
        Initialize a ridge regression model

        Parameters
        ----------
        lam: float
            Regularization parameter to be used when fitting a model

        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """


        """
        Initialize a ridge regression model
        :param lam: scalar value of regularization parameter
        """
        super().__init__()
        self.coefs_ = None
        self.include_intercept_ = include_intercept
        self.lam_ = lam

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Ridge regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        n_samples, n_features = X.shape # TODO column stack and why order?
        id_lambda = self.lam_ * np.identity(n_features)
        if self.include_intercept_:
            X = np.insert(X, 0, [np.ones(n_samples)], axis=1)
            id_lambda = np.insert(id_lambda, 0, [np.zeros(id_lambda.shape[1])], axis=0)
            id_lambda = np.insert(id_lambda, 0, [np.zeros(id_lambda.shape[0])], axis=1)
        self.coefs_ = np.linalg.inv((X.T @ X) + id_lambda) @ X.T @ y
        self.fitted_ = True



        # if self.include_intercept_:
        #     X = np.column_stack((np.ones(n_samples), X))
        # u, s, vh = np.linalg.svd(X, full_matrices=False)
        # s_ = [i / (i ** 2 + self.lam_) for i in s]
        # self.coefs_ = (u @ (s_ * np.identity(len(s))) @ vh).T @ y
        # self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if not self.fitted_:
            raise ValueError()
        if self.include_intercept_:
            n_samples, n_features = X.shape
            X = np.column_stack((np.ones(n_samples), X))
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        y_pred = self._predict(X)
        return mean_square_error(y, y_pred)
