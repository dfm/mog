#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.patches import Ellipse

import mog


def demo():
    # Make some random data in 2D.
    np.random.seed(150)
    means = np.array([[2.1, 4.5],
                      [2.0, 2.7],
                      [3.5, 5.6]])
    covariances = [np.array([[0.20, 0.10], [0.10, 0.60]]),
                   np.array([[0.35, 0.22], [0.22, 0.15]]),
                   np.array([[0.06, 0.05], [0.05, 1.30]])]
    amplitudes = [5, 1, 2]
    factor = 100

    data = np.zeros((1, 2))
    for i in range(len(means)):
        data = np.concatenate([data,
            np.random.multivariate_normal(means[i], covariances[i],
                                                size=factor * amplitudes[i])])
    data = data[1:, :]

    # Set up the mixture model.
    mixture = mog.MixtureModel(3, data)

    # Fit the mixture model.
    mixture.run_kmeans()
    mixture.run_em()

    # Plot the samples color coded by fit mixture component.
    pl.scatter(data[:, 0], data[:, 1], marker="o",
            c=[tuple(mixture.rs[i, :]) for i in range(data.shape[0])],
            s=8., edgecolor="none")

    # Plot the covariance ellipses.
    for k in range(mixture.K):
        x, y = mixture.means[k][0], mixture.means[k][1]
        U, S, V = np.linalg.svd(mixture.cov[k])
        theta = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        ellipsePlot = Ellipse(xy=[x, y], width=2 * np.sqrt(S[0]),
            height=2 * np.sqrt(S[1]), angle=theta,
            facecolor="none", edgecolor="k", lw=2)
        ax = pl.gca()
        ax.add_patch(ellipsePlot)

    pl.savefig("demo.png")


if __name__ == "__main__":
    demo()
