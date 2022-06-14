from __future__ import annotations
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

import pandas as pd
import sklearn.linear_model
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    # samples_x = np.random.uniform(-1.2, 2, n_samples)
    samples_x = np.linspace(-1.2, 2, n_samples)
    f = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    mu, sigma = 0, noise
    eps = np.random.normal(mu, sigma, n_samples)  # noise
    true_y = f(samples_x)
    samples_y = true_y + eps
    samples_x = pd.DataFrame(samples_x)
    samples_y = pd.Series(samples_y)
    samples_y.name = 'y'
    # samples_y.columns.name = 'y'
    train_x, train_y, test_x, test_y = split_train_test(samples_x, samples_y, 2/3)
    train_x, train_y, test_x, test_y = train_x.to_numpy(), train_y.to_numpy(), test_x.to_numpy(), test_y.to_numpy()
    samples_x, samples_y = samples_x.to_numpy().flatten(), samples_y.to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(name= "True Model", x=samples_x, y=true_y, mode='markers'))
    fig.add_trace(go.Scatter(name= "Train Samples", x=train_x, y=train_y, mode='markers'))
    fig.add_trace(go.Scatter(name= "Test Samples", x=test_x, y=test_y, mode='markers'))
    fig.update_layout(title_text= "Generated Data, With and Without Noise")
    fig.show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    val_score_set = []
    train_score_set = []
    for k in range(11):
        h_i = PolynomialFitting(k)
        train_score, val_score = cross_validate(h_i, train_x, train_y, mean_square_error, cv=5)
        val_score_set.append(val_score)
        train_score_set.append(train_score)
    k_arange = np.arange(11)
    best_k = np.argmin(val_score_set)

    fig1 = go.Figure(data=[go.Bar(x=k_arange, y=val_score_set, name="Average Validation Error"),
                           go.Bar(x= k_arange, y=train_score_set, name="Average Training Error")])
    fig1.update_layout(title_text= "Average Validation and Training Error Per Polynomial Degree",
                       xaxis_title="Polynomial Degree", yaxis_title="MSE")
    fig1.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_k_polynomial = PolynomialFitting(best_k)
    best_k_polynomial.fit(train_x, train_y)
    best_k_error = best_k_polynomial.loss(test_x, test_y)  ## TODO use loss?
    print(f"Best Polynomial Degree is: {best_k}, "
          f"With Test Error {best_k_error: .2f} ")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_x, train_y = X[:n_samples], y[:n_samples]
    test_x, test_y = X[n_samples:], y[n_samples:]



    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lasso_range = np.linspace(0.001, 2.5, n_evaluations)
    ridge_range = np.linspace(0.001, 2.5, n_evaluations)

    ridge_train_error = []
    ridge_val_error = []
    lasso_train_error = []
    lasso_val_error = []

    for lam in lasso_range:
        l_t_error, l_v_error = cross_validate(Lasso(alpha=lam), train_x, train_y, mean_square_error, cv=5) #  TODO intercept? why not estimator?

        lasso_train_error.append(l_t_error)
        lasso_val_error.append(l_v_error)

    for lam in ridge_range:
        r_t_error, r_v_error = cross_validate(RidgeRegression(lam), train_x, train_y, mean_square_error, cv=5)
        ridge_train_error.append(r_t_error)
        ridge_val_error.append(r_v_error)


    fig2 = go.Figure() # Lasso Plot
    fig2.add_trace(go.Scatter(name="Train Error", x=lasso_range, y=lasso_train_error, mode='markers+lines'))
    fig2.add_trace(go.Scatter(name="Validation Error", x=lasso_range, y=lasso_val_error, mode='markers+lines'))
    fig2.update_layout(title_text="Train and Validation Errors as a Function of Lasso Parameter Value")
    fig2.show()

    fig3 = go.Figure()  # Ridge Plot
    fig3.add_trace(go.Scatter(name="Train Error", x=ridge_range, y=ridge_train_error, mode='markers+lines'))
    fig3.add_trace(go.Scatter(name="Validation Error", x=ridge_range, y=ridge_val_error, mode='markers+lines'))
    fig3.update_layout(title_text="Train and Validation Errors as a Function of Ridge Parameter Value")
    fig3.show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_lasso_error = lasso_range[np.argmin(lasso_val_error)]
    print(f"Best Validation Error for the Lasso Achieved by the regularization Parameter {best_lasso_error}")
    best_ridge_error = ridge_range[np.argmin(ridge_val_error)]
    print(f"Best Validation Error for the Ridge Achieved by the regularization Parameter {best_ridge_error}")


    ridge = RidgeRegression(best_ridge_error).fit(train_x, train_y)
    print(f"Test Error For Ridge: {ridge.loss(test_x, test_y)}")
    lasso = Lasso(best_lasso_error).fit(train_x, train_y)
    print(f"Test Error For Lasso: {mean_square_error(test_y, lasso.predict(test_x))}")
    least_sq = LinearRegression().fit(train_x, train_y)
    print(f"Test Error For Least Squares: {least_sq.loss(test_x, test_y)}")



if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
