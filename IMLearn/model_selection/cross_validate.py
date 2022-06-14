from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    # data = X.copy()
    # y_data = y.copy()
    # val_size = X.shape[0] / cv
    # train1, validation_set, train2 = data[ : i * val_size], data[i * val_size : (i+1) * val_size],\
    #                                  data[(i+1) * val_size : cv * val_size]
    # y1, y_val, y2 = data[ : i * val_size], data[i * val_size : (i+1) * val_size],\
    #                                  data[(i+1) * val_size : cv * val_size]
    #
    # train_set = np.concatenate(train1, train2)

    val_score_set = []
    train_score_set = []
    for i in range(cv):

        x_range = np.arange(X.shape[0])
        remainder = np.remainder(x_range, cv)
        train_set, val_set = X[remainder != i], X[remainder == i]
        train_y, val_y = y[remainder != i], y[remainder == i]

        h_i = estimator.fit(train_set, train_y)
        val_score = scoring(val_y, h_i.predict(val_set))
        val_score_set.append(val_score)
        train_score = scoring(train_y, h_i.predict(train_set))
        train_score_set.append(train_score)

    return np.array(train_score_set).mean(), np.array(val_score_set).mean()


