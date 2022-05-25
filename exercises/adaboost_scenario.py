import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics import accuracy

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    def wl():
        return DecisionStump()

    adaboost = AdaBoost(wl, n_learners)
    adaboost.fit(train_X, train_y)
    iterations = np.linspace(1, n_learners, n_learners)
    train_loss = []
    test_loss = []
    for i in np.arange(1, n_learners + 1):
        train_loss.append(adaboost.partial_loss(train_X, train_y, i))
        test_loss.append(adaboost.partial_loss(test_X, test_y, i))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=iterations, y= train_loss))
    fig.add_trace(go.Scatter(x=iterations, y= test_loss))
    fig.update_layout(title_text="Train And Test Loss As A Function Of The Number Of Fitted Learners")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]

    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    symbols = np.array(["circle", "x"])

    fig1 = make_subplots(rows=2, cols=2, subplot_titles=[f"{t} Weak Learners" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        f = lambda x: adaboost.partial_predict(x, t)
        fig1.add_traces([decision_surface(f, lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                                               colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig1.update_layout(title=rf"$\textbf{{Decision boundaries for different amounts of weak learners}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig1.show()



    # Question 3: Decision surface of best performing ensemble
    best_t = np.argmin(test_loss) + 1
    p = lambda X: adaboost.partial_predict(X, best_t)
    best_pred = adaboost.partial_predict(test_X, best_t)
    fig2 = go.Figure()
    fig2.add_traces([decision_surface(p, lims[0], lims[1], showscale=False),
                    go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(color=test_y.astype(int), symbol=symbols[test_y.astype(int)],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])
    ensemble_accuracy = accuracy(test_y, best_pred)
    fig2.update_layout(title=f"Decision Boundaries for Best Ensemble Of Size {best_t},"
                             f" With Accuracy: {ensemble_accuracy} \n  - Noise Factor {noise}.",
                      margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig2.show()

    # Question 4: Decision surface with weighted samples
    normalized_D = (adaboost.D_ / np.max(adaboost.D_)) * 5
    fig3 = go.Figure()
    fig3.add_traces([decision_surface(adaboost.predict, lims[0], lims[1], showscale=False),
                    go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                               marker=dict(size=normalized_D, color=train_y.astype(int),
                                           symbol=symbols[train_y.astype(int)],
                                           colorscale=[custom[0], custom[-1]],
                                           line=dict(color="black", width=1)))])
    fig3.update_layout(title=f"Train Set Samples According To Their Weights in the  Last Distribution\n "
                             f"- Noise Factor {noise}.",
                      margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    fig3.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0.4)
