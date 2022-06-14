from typing import Tuple
import numpy as np
import pandas as pd


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    tagert = y.name
    union_set = pd.concat([X,y], axis=1)
    mix_union = union_set.sample(frac=1)
    n_samples = mix_union.shape[0]
    train_size = int(train_proportion * n_samples)
    union_train = mix_union[:train_size]
    union_test = mix_union[train_size:]
    test_y = union_test[tagert]
    test_x = union_test.drop([tagert], axis=1, inplace=False)
    train_y = union_train[tagert]
    train_x = union_train.drop([tagert], axis=1, inplace=False)

    if len(train_x.columns) == 1:
        train_x = train_x[train_x.columns[0]]
        test_x = test_x[test_x.columns[0]]

    return (train_x, train_y, test_x, test_y)


    # tagert_col = y.name
    # df = pd.DataFrame(X)
    # df[tagert_col] = y
    # train_x = df.sample(frac=train_proportion)
    # train_y = train_x[tagert_col]
    # test_x = df[~df.index.isin(train_x.index)]
    # test_y = test_x[tagert_col]
    # train_x.drop([tagert_col], axis=1, inplace=True)
    # test_x.drop([tagert_col], axis=1, inplace=True)
    # if len(train_x.columns) == 1:
    #     train_x = train_x[train_x.columns[0]]
    #     test_x = test_x[test_x.columns[0]]
    # return train_x, train_y, test_x, test_y

def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
