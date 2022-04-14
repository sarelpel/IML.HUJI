import datetime
import os.path

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import kaleido

pio.templates.default = "simple_white"


def data_restrictions(df: pd.DataFrame):
    df.dropna(inplace=True)
    curr_age = 2022
    df.drop(df.index[df['floors'] <= 0], inplace=True)
    df.drop(df.index[df['bathrooms'] < 0], inplace=True)
    df.drop(df.index[df['bedrooms'] <= 0], inplace=True)
    df.drop(df.index[df['sqft_living'] <= 0], inplace=True)
    df.drop(df.index[df['yr_built'] > curr_age], inplace=True)
    df.drop(df.index[df['yr_renovated'] > curr_age], inplace=True)

    df = pd.get_dummies(df, columns=['zipcode'])
    df['built_or_renovated_yr'] = df[['yr_built', 'yr_renovated']].max(axis=1)

    return df



def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = data_restrictions(df)
    y = df['price']
    df.drop(['id', 'date', 'price', 'yr_built', 'yr_renovated', 'lat', 'long'], axis=1, inplace=True)

    return (df, y)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for title in X.columns:
        feature = X[title]
        p_corr = np.cov(feature, y) [0][1]/ (np.std(feature) * np.std(y))

        fig = go.Figure([go.Scatter(x=feature, y=y, mode='markers')],
                        layout=go.Layout(title=f"{title} As Function Of The House's Price. Pearson Correlation={p_corr}",
                                         xaxis_title=f"Feature: {title}",
                                         yaxis_title="Price Of The House"))
        fig.show()
        fig.write_image(os.path.join(output_path, str(title) + ".png"))



if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    (df, y) = load_data(r'C:\Users\Asus\IML.HUJI\datasets\house_prices.csv')

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(df, y, "C:/Users/Asus/IML.HUJI/exercises/figures")

    # Question 3 - Split samples into training- and testing sets.
    (train_x, train_y, test_x, test_y) = split_train_test(df, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    precentage_vec = np.linspace(10, 100, 91)
    total_mean = []
    total_std = []
    for per in precentage_vec:
        inner_loss = []
        for i in range(10):
            (new_train_x, new_train_y, new_test_x, new_test_y) = split_train_test(train_x, train_y, float(per/100))
            linear_reg = LinearRegression()
            linear_reg.fit(new_train_x.to_numpy(), new_train_y.to_numpy())
            inner_loss.append(linear_reg.loss(test_x.to_numpy(), test_y.to_numpy()))
        inner_loss = np.array(inner_loss)
        total_mean.append(inner_loss.mean())
        total_std.append(inner_loss.std())

    total_mean = np.array(total_mean)
    total_std = np.array(total_std)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=precentage_vec, y=total_mean, mode="markers+lines", name="Mean Loss",
                             line=dict(dash="dash"), marker=dict(color="blue", opacity=.7)))
    fig.add_trace(go.Scatter(x=precentage_vec, y=total_mean - 2*total_std, fill=None, mode="lines",
                             line=dict(color="lightgrey"), showlegend=False))
    fig.add_trace(go.Scatter(x=precentage_vec, y=total_mean + 2*total_std, fill='tonexty', mode="lines",
                             line=dict(color="lightgrey"), showlegend=False))
    fig.update_layout(title=f" Connection Between average loss And training size With Error Ribbon Of Size (+,-) 2*std",
                        xaxis_title="Training Size In Precentages",
                        yaxis_title="Average Loss")
    fig.show()
