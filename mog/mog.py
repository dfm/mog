#!/usr/bin/env python
"""
Gaussian mixture models

"""

from __future__ import division, print_function


__all__ = ['MixtureModel']


import numpy as np
import numpy.ma as ma

from . import _algorithms


class MixtureModel(object):
    """
    Gaussian mixture model for samples.

    ``P`` data points in ``D`` dimensions with ``K`` clusters.

    :param K:
        The number of Gaussians to include in the mixture.

    :param data:
        A ``P x D`` ``numpy.ndarray`` of the samples.

    """
    def __init__(self, K, data, init_grid=False):
        self.K = K
        self._data = np.atleast_2d(data)
        self._lu = None

        self.kmeans_rs = np.zeros(self._data.shape[0], dtype=int)

        # Randomly choose ``K`` components to be the initial means.
        inds = np.random.randint(data.shape[0], size=self.K)
        self.means = data[inds, :]

        # Initialize the covariances as the data covariance.
        self.cov = np.array([np.cov(data, rowvar=0)] * self.K)

        # Randomly assign the amplitudes.
        self.amps = np.random.rand(K)
        self.amps /= np.sum(self.amps)

    def run_kmeans(self, maxiter=200, tol=1e-4, verbose=True):
        """
        Run the K-means algorithm using the C extension.

        :param maxiter:
            The maximum number of iterations to try.

        :param tol:
            The tolerance on the relative change in the loss function that
            controls convergence.

        :param verbose:
            Print all the messages?

        """
        iterations = _algorithms.kmeans(self._data, self.means,
                self.kmeans_rs, tol, maxiter)

        if verbose:
            if iterations < maxiter:
                print("K-means converged after {0} iterations."
                        .format(iterations))
            else:
                print("K-means *didn't* converge after {0} iterations."
                        .format(iterations))

    def run_em(self, maxiter=400, tol=1e-4, verbose=True, regularization=0.0):
        """
        Run the EM algorithm.

        :param maxiter:
            The maximum number of iterations to try.

        :param tol:
            The tolerance on the relative change in the loss function that
            controls convergence.

        :param regularization:
            Add this value on the diagonal of the covariances to avoid
            singular matrices.

        :param verbose:
            Print all the messages?

        """
        self.means = self.means.T

        L = None
        for i in xrange(maxiter):
            newL = self._expectation()
            if i == 0 and verbose:
                print("Initial NLL =", -newL)

            self._maximization(regularization)
            if L is None:
                L = newL
            else:
                dL = np.abs((newL - L) / L)
                if i > 5 and dL < tol:
                    break
                L = newL

        if i < maxiter - 1:
            if verbose:
                print("EM converged after {0} iterations".format(i))
                print("Final NLL = {0}".format(-newL))
        else:
            print("Warning: EM didn't converge after {0} iterations"
                    .format(i))

        self.means = self.means.T

    def _log_multi_gauss(self, k, X):
        # X.shape == (P,D)
        # self.means.shape == (D,K)
        # self.cov[k].shape == (D,D)
        sgn, logdet = np.linalg.slogdet(self.cov[k])
        if sgn <= 0:
            return -np.inf * np.ones(X.shape[0])

        # X1.shape == (P,D)
        X1 = X - self.means[None, :, k]

        # X2.shape == (P,D)
        X2 = np.linalg.solve(self.cov[k], X1.T).T

        p = -0.5 * np.sum(X1 * X2, axis=1)

        return -0.5 * np.log((2 * np.pi) ** (X.shape[1])) - 0.5 * logdet + p

    def _expectation(self):
        # self.rs.shape == (P,K)
        L, self.rs = self._calc_prob(self._data)
        return np.sum(L, axis=0)

    def _maximization(self, regularization):
        # Nk.shape == (K,)
        Nk = np.sum(self.rs, axis=0)
        Nk = ma.masked_array(Nk, mask=Nk <= 0)
        # self.means.shape == (D,K)
        self.means = ma.masked_array(np.sum(self.rs[:, None, :] \
                * self._data[:, :, None], axis=0))
        self.means /= Nk[None, :]
        self.cov = []
        for k in range(self.K):
            # D.shape == (P,D)
            D = self._data - self.means[None, :, k]
            self.cov.append(np.dot(D.T, self.rs[:, k, None] * D) / Nk[k]
                    + regularization * np.eye(self.means.shape[0]))
        self.amps = Nk / self._data.shape[0]

    def _calc_prob(self, x):
        x = np.atleast_2d(x)

        logrs = []
        for k in range(self.K):
            logrs += [np.log(self.amps[k]) + self._log_multi_gauss(k, x)]
        logrs = np.concatenate(logrs).reshape((-1, self.K), order='F')

        # here lies some ghetto log-sum-exp...
        # nothing like a little bit of overflow to make your day better!
        a = np.max(logrs, axis=1)
        L = a + np.log(np.sum(np.exp(logrs - a[:, None]), axis=1))
        logrs -= L[:, None]
        return L, np.exp(logrs)

    def lnprob(self, x):
        return self._calc_prob(x)[0]
