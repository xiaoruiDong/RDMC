#!/usr/bin/env python3
#-*- coding: utf-8 -*-

"""
A module used to fit curves or surfaces. Potentially useful in characterize PES.
"""

import numpy as np


class FourierSeries1D(object):
    """
    FourierSeries1D helps to fit the potential energy surface to a 1D
    Fourier Series, and evaluate the values using the fitted curve.
    """

    # Control the max number of terms used in fitting
    max_num_term = None

    def __init__(self,
                 max_num_term: int = None,):
        """
        Initialize the Fourier Series 1D fitting class

        Args:
            max_num_term (int, optional): The max number of cosine terms used in fitting. Defaults to None.
        """
        self.max_num_term = max_num_term or self.max_num_term

    def fit(self,
            X: np.ndarray,
            y: np.ndarray):
        """
        Fit the 1D Fourier series.

        Args:
            X (np.ndarray): _description_
            y (np.ndarray): 1D array. It is assumed to be a periodic function of `x` with
                            a periodicity of :math:`2 \pi`.
        """
        negative_barrier = True
        # numterms is actually half the number of terms. It is called numterms
        # because it is the number of terms of either the cosine or sine fit
        self.num_terms = 6
        if self.max_num_term:
            maxterms = min(np.floor(X.shape[0] / 3.0), self.num_terms)
        else:
            maxterms = np.floor(X.shape[0] / 3.0)

        while negative_barrier and self.num_terms <= maxterms:
            # Fit Fourier series potential
            # A: [1, cos(phi), ..., cos(M * phi), sin(phi), ..., sin(M * phi)]
            A = self._preprocess_x(X)
            # Last row correspond to dy/dX(0), equivalent to forces stationary point at angle = 0
            dydX_0 = np.zeros((1, A.shape[1]))
            dydX_0[0, self.num_terms:] = 1
            A = np.vstack([A, dydX_0])
            b = np.concatenate([y, np.array([0.])])

            # Least square linear regression
            coef, _, _, _ = np.linalg.lstsq(A, b)

            # This checks if there are any negative values in the Fourier fit.
            negative_barrier = False
            V0 = 0.0
            self.coef_ = np.array([coef[1:self.num_terms],
                                   coef[self.num_terms:2 * self.num_terms - 1]])
            for k in range(self.coef_.shape[1]):
                V0 -= self.coef_[0, k] * (k + 1) * (k + 1)
            if V0 < 0:
                negative_barrier = True
                print(f"Fourier fit for hindered rotor gave a negative barrier when fit with {2 * self.num_terms} terms, "
                      f"retrying with {2 * self.num_terms + 4} terms...")
                self.num_terms = self.num_terms + 2
            if V0 < 0:
                print(f"Fourier fit for hindered rotor gave a negative barrier on final try with "
                      f"{self.num_terms * 2} terms")

    def _preprocess_x(self,
                      X: np.ndarray,):
        """
        Preprocess the X variables. Convert X to [1, cos(X), ..., cos(M * X), sin(X), ..., sin(M * X)],
        with a size of (len(X), 2 * numterms + 1).

        Args:
            X (np.ndarray): A 1D array.
        """
        n_rows = X.shape[0]
        A = np.zeros((n_rows, 2 * self.num_terms - 1), np.float64)
        # Set first column of A to 1, corresponding to the bias term
        A[:, 0] = 1
        for m in range(1, self.num_terms):
            A[:, m] = np.cos(m * X)
            A[:, self.num_terms + m - 1] = np.sin(m * X)
        return A

    def predict(self,
                X: np.ndarray,):
        """
        Predict the value using Fourier Series. This function forces y=0 at the origin.

        Args:
            X (np.ndarray): The X variable that should be a 1D array.

        Returns:
            _type_: _description_
        """
        A = self._preprocess_x(X)
        return np.dot(A, np.hstack([np.array([-np.sum(self.coef_[0, :])]), self.coef_[0, :], self.coef_[1, :]]))
