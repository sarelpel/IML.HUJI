from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """

        m = len(y)
        features = X.shape[1]
        self.classes_, class_count = np.unique(y, return_counts=True)
        self.pi_ = class_count/ m

        mu_matrix = np.zeros((len(self.classes_), features))
        var_matrix = np.zeros((len(self.classes_), features))

        for ind, c in enumerate(self.classes_):
            mu_matrix[ind] = np.mean(X[y == c], axis=0)
            var_matrix[ind] = np.var(X[y == c], axis=0)

        self.mu_ = mu_matrix
        self.vars_ = var_matrix

        self.fitted_ = True


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
        like_mat = self.likelihood(X)
        y_pred = np.argmax(like_mat, axis=1)
        return self.classes_[y_pred]


    def _log_pdf(self, X, class_ind):
        """
        Calculate the log(pdf) of X under specific label.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its log pdf under specific class.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            The pdf for each sample under specific class.
        """
        class_mu = self.mu_[class_ind]
        class_var = self.vars_[class_ind]

        return np.log(1/np.sqrt(2*np.pi*class_var)) + (-((X - class_mu)**2)/(2*class_var))


    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        m = X.shape[0]
        n_classes = len(self.classes_)
        like_mat = np.empty((n_classes, m))

        for class_ind in range(n_classes):
            like_mat[class_ind] = np.sum(self._log_pdf(X, class_ind), axis=1) + np.log(self.pi_[class_ind])

        return like_mat.T


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
        from ...metrics import misclassification_error
        y_pred = self._predict(X)
        return misclassification_error(y, y_pred)
