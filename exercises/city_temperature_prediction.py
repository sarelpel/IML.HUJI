from pygments.lexers import go

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df.dropna(inplace=True)
    df = df[(df.Temp > -15) & (df.Temp < 50)]
    df = df[df.Year > 0]
    df = df[(df.Month > 0) & (df.Month <= 12)]
    df = df[(df.Day > 0) & (df.Day <= 31)]

    df['DayOfYear'] = df['Date'].dt.dayofyear

    return df




if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data(r'C:\Users\Asus\IML.HUJI\datasets\City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    i_df = df[df.Country == 'Israel']

    fig = px.scatter(x = i_df['DayOfYear'], y = i_df['Temp'], color = i_df['Year'].astype(str))
    fig.update_layout(title=f"Average Daily Temperature As A Function Of The `DayOfYear`.",
                        xaxis_title="Day Of Year (each year gets different color)",
                        yaxis_title="Average Temperature",
                      legend_title="Year")
    fig.show()


    std_month = i_df.groupby(['Month']).Temp.agg(std= 'std')
    std_month_bar = px.bar(std_month,  y="std", title="Standard Deviation for the Different Months")
    std_month_bar.show()

    # Question 3 - Exploring differences between countries
    country_month_set = df.groupby(['Country', 'Month']).Temp
    country_month_set = country_month_set.agg(mean = "mean", std = "std").reset_index()
    country_month_set.columns = ["Country", "Month", "mean", "std"]
    fig_C_M = px.line(country_month_set, x="Month", y= "mean", color= country_month_set["Country"].astype(str),
                      title= "temperature avarage and std grouped by country and month",
                      error_y= "std")
    fig_C_M.update_layout(legend_title = "Country").show()



    # Question 4 - Fitting model for different values of `k`
    (train_x, train_y, test_x, test_y) = split_train_test(i_df['DayOfYear'], i_df['Temp'], 0.75)
    loss_arr = []
    for k in range(1, 11):
        polynomial_fit = PolynomialFitting(k)
        polynomial_fit.fit(train_x.to_numpy(), train_y.to_numpy())
        l= polynomial_fit.loss(test_x.to_numpy(), test_y.to_numpy())
        l = round(l, 2)
        print(l)
        loss_arr.append(l)
    loss_arr = np.array(loss_arr)
    loss_per_deg = px.bar(x= range(1, 11), y= loss_arr,  title="Loss Value per Polynomial Degree",
                              text_auto= True)
    loss_per_deg.update_layout(xaxis_title="Polynomial Degree",
                                yaxis_title="MSE")
    loss_per_deg.show()


    # Question 5 - Evaluating fitted model on different countries
    best_deg = 5
    best_polynomial = PolynomialFitting(best_deg)
    best_polynomial.fit(i_df['DayOfYear'], i_df['Temp'])
    countries_arr = []
    country_loss_arr = []
    for country in df['Country'].unique():
        if country != "Israel":
            o_df = df[df.Country == country]
            c_loss = best_polynomial.loss(o_df.DayOfYear, o_df.Temp)
            countries_arr.append(country)
            country_loss_arr.append(c_loss)
    country_loss_bar = px.bar(x = countries_arr, y = country_loss_arr,
                              title= "Israel's best model's error over each of the other countries")
    country_loss_bar.update_layout(xaxis_title="Country",
                                yaxis_title="MSE")
    country_loss_bar.show()
