from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    X = np.random.normal(mu, sigma, size=1000)
    uni_gaussian1 = UnivariateGaussian()
    uni_gaussian1.fit(X)
    # print ("(" + uni_gaussian1.mu_, uni_gaussian1.var_ + ")")
    print(f"({uni_gaussian1.mu_}, {uni_gaussian1.var_})")


    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    estimated_mean = []
    uni_gaussian2 = UnivariateGaussian()
    for m in ms:
        uni_gaussian2.fit(X[:m])
        estimated_mean.append(abs(uni_gaussian2.mu_ - mu))

    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu - \mu$')],
              layout=go.Layout(title=r"$\text{ Absolute Distance Between The Estimated- And True Value Of The"
                                     r" Expectation,"r" As A Function Of The Sample Size}$",
                               xaxis_title="Number Of Samples",
                               yaxis_title="Distance From True Expectation")).show()


    # Question 3 - Plotting Empirical PDF of fitted model
    density_arr = uni_gaussian1.pdf(X)

    go.Figure([
               go.Scatter(x=X, y=density_arr, mode='markers', line=dict(width=4, color="rgb(204,68,83)"),
                          name=r'$N(\mu, \frac{\sigma^2}{m1})$')],
              layout=go.Layout(barmode='overlay',
                               title="Estinated Point Density Function",
                               xaxis_title="Sample Value",
                               yaxis_title="Density")).show()

def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    X = np.random.multivariate_normal(mu, cov, size=1000)
    multi_gaussian = MultivariateGaussian()
    multi_gaussian.fit(X)

    print("mu:\n", multi_gaussian.mu_)
    print("cov:\n", multi_gaussian.cov_)

    # Question 5 - Likelihood evaluation
    f = np.linspace(-10, 10, 200)

    mu_combs = np.array(np.meshgrid(f, 0, f, 0)).T.reshape(-1, 4)
    log_like_func = lambda a: MultivariateGaussian.log_likelihood(a, cov, X)
    log_likes = np.apply_along_axis(log_like_func, 1, mu_combs)



    fig = go.Figure()
    fig.add_trace(go.Heatmap(x = f, y = f, z = log_likes.reshape(200 , 200).T, colorbar=dict(title="Log Likelihood")))
    fig.update_layout(title=r"$\text{Log Likelihood Of Samples With Mean [f1, 0, f3, 0]"
                                     r"As Function Of f1 And f3}$",
                               xaxis_title="value of f3",
                               yaxis_title="value of f1")
    fig.show()


    # Question 6
    max_mu = mu_combs[np.argmax(log_likes)]
    print("f1: %.3f" % max_mu[0])
    print("f3: %.3f" % max_mu[2])




if __name__ == '__main__':

    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
