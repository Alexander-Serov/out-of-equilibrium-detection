"""
Calculate likelihoods that will be used for model comparison.
Those are mainly likelihoods for periodogram components.
"""

import copy
import functools
import logging
import time
from datetime import datetime
from operator import itemgetter

import numdifftools as nd
import numpy as np
import scipy
from filelock import FileLock
from matplotlib import pyplot as plt
from numpy import cos, exp, log, pi, sin, sqrt
from numpy.linalg import eig, inv
from scipy import integrate
from scipy.fftpack import fft
from scipy.integrate import dblquad, nquad
from scipy.optimize import minimize, root, root_scalar
from scipy.special import erf, gamma, gammainc, gammaincc, gammaln, logsumexp

from constants_main import color_sequence
# from plot import plot_periodogram
from support import (delete_data, hash_from_dictionary, load_data,
                     load_MLE_guess, save_data, save_MLE_guess,
                     save_number_of_close_values, stopwatch)
# import pymc3 as pm
# from pymc3.math import matrix_inverse, dot
# import theano
# from theano import tensor
# import theano.tensor as tt
import timeit
from stopwatch import stopwatch
import warnings

ln_neg_infty = - 1000 * log(10)
plot = False
max_tries = 100
SAME_MINIMA_STOP_CRITERION = 4  # How many similar minima should be found for the procedure to stop
MINIMA_CLOSE_ATOL = 0.1

# Prior parameters
max_expected_D = 5
max_expected_n = 100
max_expected_n12 = 1000
max_expected_L0 = 10  # unused?


# tries = 3  # 3

# used to evaluate how many points after optimization are similar on the average
# prior_sampling_statistics_file = 'statistics.dat'


class J_class:
    """
    A class that allows one to perform calculations with Fourier images of the correlation function.
    """

    def __init__(self, j, k, Jfunc, scale=1):
        self.scale = scale
        self.k = k
        self.Jfunc = Jfunc
        if isinstance(j, int):
            self.js = [j]
        elif isinstance(j, list):
            self.js = j
        else:
            raise TypeError(f'Cannot initialize J_class object with j={j}')

    def __str__(self):
        return f'J_class({self.js},{self.k}, {self.scale})'

    def __repr__(self):
        return self.__str__()

    def __mul__(self, other):

        if isinstance(other, self.__class__):
            if self.k != other.k or self.Jfunc != other.Jfunc:
                raise RuntimeError(
                    f'Unable to calculate correlation of integrals with different k: [{self.k}, {other.k}]')

            # Convert to a float
            new_scale = self.scale * other.scale
            new_js = self.js + other.js  # list concat
            # new_Jfunc = self.Jfun
            if len(new_js) > 2:
                raise RuntimeError(
                    f'The behvaior for a correlation of more than 2 integrals is not defined. Encountered: {len(new_js)} integrals')

            out = new_scale * self.Jfunc(new_js[0], new_js[1], self.k)

        elif other == 0:
            # print('n')
            out = 0
        else:
            # keep class object
            # print('o', other)
            out = J_class(self.js, self.k, self.Jfunc, self.scale * other)
        return out


def __rmul__(self, other):
    return self.__mul__(other)


def __pow__(self, pow):
    if pow != 2:
        raise ValueError(f'Power operator not defined for pow = {pow}')
    else:
        return self * self


def __add__(self, other):
    if other == 0:
        return self
    elif isinstance(other, self.__class__):
        # new_js = [self.js, other.js]
        # ks = [self.k, other.k]
        # Jfuncs = [self.Jfunc, other.Jfunc]
        # scales = [self.scale, other.scale]
        return J_combination([self, other])
    else:
        raise RuntimeError(f'Addition not defined for the argument {other}')


def __radd__(self, other):
    return self.__add__(other)


class J_combination:
    """A linear combination of J classes"""

    def __init__(self, Js):
        self.Js = Js
        # self.js = list(js)
        # self.ks = list(ks)
        # self.Jfuncs = list(Jfuncs)
        # self.scales = list(scales)

    def __str__(self):
        return f'J_combination({self.Js})'

    def __repr__(self):
        return self.__str__()

    # def len(self):
    #     return len(self.Js)

    def __add__(self, other):
        if other == 0:
            return self
        else:
            raise RuntimeError(f'Addition not defined for the argument {other}')

    def __radd__(self, other):
        return self.__add__(other)

    def __pow__(self, pow):
        if pow == 2:
            n = len(self.Js)
            out = [self.Js[i] * self.Js[j] for i, j in np.ndindex(n, n)]
            out = np.sum(out)
        else:
            raise ValueError(f'Power operator not defined for pow = {pow}')

        return out

    def __mul__(self, other):
        n = len(self.Js)
        return np.sum([J * other for J in self.Js])

    def __rmul__(self, other):
        return self.__mul__(other)


class matrix:
    """A class that can handle matrix multiplication of arrays with J integrals"""

    def __init__(self, array):
        self.array = array

    def __str__(self):
        return self.array.__str__()

    def __repr__(self):
        return self.__str__()

    def __matmul__(self, other):
        s1 = self.array.shape
        s2 = other.array.shape
        s = [s1[0], s2[1]]
        # print('in1', self.array)
        # print('in2', other.array)

        out = np.full(s, np.nan, dtype=object)
        for i, j in np.ndindex(s[0], s[1]):
            # print('p1', self.array[i, :] * other.array[:, j])
            out[i, j] = (self.array[i, :] * other.array[:, j]).sum()
            # print('p2',  (self.array[i, :] * other.array[:, j]).sum(), out[i, j])
            # print('p3', out)

        # print('type', type(out))
        # print('prod', out)
        return matrix(out)

    def __pow__(self, pow):
        """Term-wise power"""
        new_array = self.array ** pow
        return matrix(new_array)


@functools.lru_cache(maxsize=128)  # Cache the return values
def get_sigma2_matrix_func(D1=1, D2=3, n1=1, n2=1, n12=1, M=999, dt=0.3, alpha=0, **kwargs):
    """
    This function constructs the sigma^2 matrix, which is the matrix of the coefficients of the chi-squared functions that then define the distribution of periodogram components:
    $ |\ksi_k^\alpha|^2 \simeq \sum_i \sigma_{i \alpha}^2 \chi_2^2 $
    Here
    $ \ksi_n \equiv d R_n - \mu_n $

    Here alpha corresponds to the analyzed coordinates: {x1, y1, x2, y2}, and k is supposedly the wave-number.
    """
    atol = 1e-10

    # Check values
    if np.any(np.array([D1, D2, n1, n2, n12]) < 0):
        return np.nan

    A = np.array([[-n1 - n12, 0, n12, 0],
                  [0, -n1, 0, 0],
                  [n12, 0, -n2 - n12, 0],
                  [0, 0, 0, -n2]])

    b = np.diag(np.sqrt([2 * D1, 2 * D1, 2 * D2, 2 * D2]))

    # print('A', A)
    # print(n1, n2, n12)
    try:
        lambdas, U = np.linalg.eig(A)
    except Exception as e:
        print(
            f'Unable to calcualte eigenvalues of the A matrix. Check out the n12 value ({n12}). The matrix: ',
            A)
        raise (e)
    # print('lambs', lambdas)
    Um1 = inv(U)

    # The decorator allows one to accelerate execution by saving previously calculated values
    @functools.lru_cache(maxsize=None)
    def J(i, j, k):
        """Return the Fourier image of the average correlation between the stochastic integrals"""
        i -= 1
        j -= 1
        if i == j:
            lamb = lambdas[i]
            if np.isclose(lamb, 0, atol):
                # A special limit for lambda close to 0, because otherwise the limit is not calculated correctly
                J = M * dt ** 3 / 2
            else:
                c1 = exp(lamb * dt)
                J = -(((M * dt ** 2) * ((1 - c1 ** 2) * (1 - cos((2 * pi * k) / M)))) / (
                        (2 * lamb) * (1 + c1 ** 2 - 2 * c1 * cos((2 * pi * k) / M))))

        else:
            lamb1, lamb2 = lambdas[[i, j]]
            J = -(((M * dt ** 2) * ((1 - exp(lamb1 * dt) * exp(lamb2 * dt)) * (
                    1 + exp(lamb1 * dt) * exp(lamb2 * dt) - (
                    exp(lamb1 * dt) + exp(lamb2 * dt)) * cos((2 * pi * k) / M)) * (
                                            1 - cos((2 * pi * k) / M)))) /
                  ((lamb1 + lamb2) * ((1 + exp(lamb1 * dt) ** 2 - 2 * exp(lamb1 * dt) * cos(
                      (2 * pi * k) / M)) * (1 + exp(lamb2 * dt) ** 2 - 2 * exp(lamb2 * dt) * cos(
                      (2 * pi * k) / M)))))
        return J

    def sigma2s_row(alpha, k):
        """Return sigma2 values that correspond to the component alpha and frequency k.
        The result is divided by (M*dt) to directly give the power spectrum
        """
        diag_J = np.diag([J_class(i, k, J) for i in range(1, 5)])
        mat = matrix(U) @ matrix(diag_J) @ matrix(Um1)

        new_mat = [((mat @ matrix(b[:, i, np.newaxis])) ** 2).array for i in range(4)]
        sigma2s = np.concatenate(new_mat, axis=1)

        return sigma2s[alpha, :] / (M * dt)

    def sigma2s_full(k):
        """Return sigma2 values that correspond to the component alpha and frequency k.
        The result is divided by (M*dt) to directly give the power spectrum
        """
        diag_J = np.diag([J_class(i, k, J) for i in range(1, 5)])
        mat = matrix(U) @ matrix(diag_J) @ matrix(Um1)

        new_mat = [((mat @ matrix(b[:, i, np.newaxis])) ** 2).array for i in range(4)]
        sigma2s = np.concatenate(new_mat, axis=1)

        return sigma2s / (M * dt)

    # print(alpha)
    if alpha is not None:
        return sigma2s_row
    else:
        return sigma2s_full


def estimate_sigma2_matrix(fit_params, ks_fit, true_parameters):
    """
    Evaluate the sigma2 matrix for the true and fit parameters.
    The sigma2 matrix defines the coefficients of the linear combination of chi2 in the distributions of the periodogram components.
    Need a function because it is repeated for different estimates.

    true_parameters must containt M and dt
    """
    k = ks_fit[-1]

    M, dt = [true_parameters[key] for key in 'M dt'.split()]

    s2_mat_func = get_sigma2_matrix_func(alpha=None, **true_parameters)
    s2_mat_true = s2_mat_func(k=k)

    s2_mat_func = get_sigma2_matrix_func(alpha=None, **fit_params)
    s2_mat_fit = s2_mat_func(k=k)

    print('True sigma2 matrix: ', s2_mat_true)
    print('Fit sigma2 matrix: ', s2_mat_fit)
    print('Fit-true sigma2 matrix: ', s2_mat_fit - s2_mat_true)


def likelihood_2_particles_x_link_one_point(z, k=1, D1=1, D2=3, n1=1, n2=1, n12=1, M=999, dt=0.3,
                                            alpha=0):
    """
    Calculate likelihood of one power spectrum observation z for the frequency k.
    Does not work for k = 0.

    Input:
    gamma --- viscosity, kg/s,
    z --- the value of the power spectrum component,
    n1, n2 , n12 --- variables starting with n are spring constants normalizedby gamma, i.e. n1 = k1 /gamma

    Definitions:
    R --- 4D vector of locations vs. time, {x1, y1, x2, y2} x N,
    dR --- 4D vector of displacements, shorter by 1 point,
    mu = <dR>,
    ksi = dR - mu --- stochastic part of the displacement,

    Main equations:
    <ksi> = 0,
    |ksi_{alpha, k}|^2 ~ \sum_i \sigma2_{alpha, i} \chi_2^2 --- the power specturm of the alpha component {x1, y1, x2, y2} of the dR vector is a sum over the contribution of 4 different stochastic processes. The 2nd order of the chi-squared function comes from the fact that the power spectrum of the real and imaginary part of dR is the same and uncorrelated,

    Important:
    - The calculations below will be wrong if one of lambda values is 0.

    """

    if np.any(np.array((D1, D2, n1, n2, n12)) < 0):
        return ln_neg_infty

    # Defined the pdf function for a sum of chi-squared
    def ln_pdf_chi_squared_sum(z, sigma2s):
        """
        Calculate the pdf for a sum of chi-squared of order 2 through residues of the characteristic function

        Formula:
        f(z) = 1/2 * \sum_j \e^{-z/2 \sigma^2_j} / (\prod_{n \neq j} (sigma2^_j - \sigma^2_n))
        """
        atol = 1e-18

        # if np.all(sigma2s < 1e-10):
        #     raise RuntimeError(
        #         f'The input contains only 0 sigma2 values. \n Input parameters: k={k}, D1={D1}, D2={D2}, n1={n1}, n2={n2}, n12={n12}')

        # print('s2: ', sigma2s)
        sigma2s_nonzero = np.array([sigma2 for sigma2 in sigma2s if abs(sigma2) > 0])

        # Check if we have same sigmas
        if len(sigma2s_nonzero) > 1 and np.any(np.abs(np.diff(np.sort(sigma2s_nonzero))) == 0):
            str = "Encountered exactly same sigmas. The current likelihood calculation through poles of order 1 may fail. Sigma2 values: " + \
                  repr(sigma2s_nonzero)
            logging.warning(str)
            # print(str)

        if len(sigma2s_nonzero) == 0:
            # print('sigma2s: ', sigma2s)
            logging.warn(
                'All input sigma2 are effectively zero. Returning 0 probability. Input lg sigmas: ' +
                np.array_str(sigma2s)
                + f'.\n Other parameters: k={k}, D1={D1}, D2={D2}, n1={n1}, n2={n2}, n12={n12}')
            return -np.inf
        # print(sigma2s_nonzero.dtype)

        n = len(sigma2s_nonzero)
        full_set = set(range(n))
        # print('n', n)

        ln_terms = np.full(n, np.nan, dtype=np.float64)
        signs = np.full_like(ln_terms, np.nan)

        # terms = np.full(n, np.nan, dtype=np.float)

        for j in range(n):
            s2j = sigma2s_nonzero[j]
            other_inds = list(full_set - set([j]))
            ln_terms[j] = -z / 2 / s2j + (n - 2) * np.log(s2j)
            ln_terms[j] -= np.log(np.abs(s2j - sigma2s_nonzero[other_inds])).sum()
            signs[j] = np.sign(np.prod(s2j - sigma2s_nonzero[other_inds]))
            # print('h', j, sigma2s_nonzero[j] -
            #       sigma2s_nonzero[other_inds], -z / 2 / sigma2s_nonzero[j])
            # print('b ', exp(-z / 2 / sigma2s_nonzero[j]),
            #       (sigma2s_nonzero[j] - sigma2s_nonzero[other_inds]).prod())

        # val = terms.sum() / 2
        # print('g', ln_terms)
        # print(ln_terms, signs)
        ln_val = logsumexp(ln_terms, b=signs) - np.log(2)

        return ln_val

    # Calculate the likelihood of the spectrum
    # sigma2s = sigma2_matrix[alpha, [0, 2]]
    sigma2s = get_sigma2_matrix_func(D1, D2, n1, n2, n12, M, dt, alpha)

    # print(sigma2s)
    try:
        ln_prob = ln_pdf_chi_squared_sum(z=z, sigma2s=sigma2s(alpha, k))
    except Exception as e:
        print(
            f'Unable to calculate the pdf of a chi-squared sum. Alpha (component) and row: alpha={alpha}, k={k}')
        print('The corresponding full sigma2 matrix:')
        print(get_sigma2_matrix_func(D1, D2, n1, n2, n12, M, dt, alpha=None)(k))
        raise e

    # print('prob: ', prob)
    # print('lambs: ', lambdas)
    # prob
    return ln_prob


def new_likelihood_2_particles_x_link_one_point(dRk, k=1, D1=1, D2=3, n1=1, n2=1, n12=1, M=999,
                                                dt=0.3, alpha=0, rotation=True, both=False,
                                                link=True):
    """
    # UPDATE
    Calculate likelihood of one power spectrum observation z for the frequency k.
    Does not work for k = 0.

    Input:
    gamma --- viscosity, kg/s,
    z --- the value of the power spectrum component,
    n1, n2 , n12 --- variables starting with n are spring constants normalizedby gamma, i.e. n1 = k1 /gamma

    Definitions:
    R --- 4D vector of locations vs. time, {x1, y1, x2, y2} x N,
    dR --- 4D vector of displacements, shorter by 1 point,
    mu = <dR>,
    ksi = dR - mu --- stochastic part of the displacement,


    Important:
    - The calculations below will be wrong if one of lambda values is 0.
    # - the Gamma and C matrices are not multiplied by

    """
    ATOL = 1e-5
    dRk = copy.copy(dRk)

    assert link != np.isclose(n12, 0), "n12=0 is only allowed when link=False"

    # print('n12', n12)
    # print(link, np.isclose(n12, 0), not np.isclose(n12, 0), link == (not np.isclose(n12, 0)),
    #       link != np.isclose(n12, 0))

    if np.any(np.array((D1, D2, n1, n2, n12)) < 0):
        return ln_neg_infty

    # %% Hard-code the eigenvalues and the diagnolizing matrix U
    # Treat separately the case of n12=0 and n12>0
    ck = exp(-2 * pi * 1j * k / M)
    if link:
        g = np.sqrt((n1 - n2) ** 2 + 4 * n12 ** 2)
        lambdas = np.array([-2 * n1, -2 * n2, -g - n1 - 2 * n12 - n2, g - n1 - 2 * n12 - n2]) / 2
        U = 1 / g * np.array([
            [0, 0, -2 * n12, 2 * n12],
            [2 * g, 0, 0, 0],
            [0, 0, g - n1 + n2, g + n1 - n2],
            [0, 2 * g, 0, 0]]) / 2

        # raise RuntimeError('Stop')

        Um1 = 1 / n12 * np.array([
            [0, 2 * n12, 0, 0],
            [0, 0, 0, 2 * n12],
            [-g - n1 + n2, 0, 2 * n12, 0],
            [g - n1 + n2, 0, 2 * n12, 0]]) / 2

        # lambdas_test, U_test = np.linalg.eig(A)

        def cj(j):
            return exp(lambdas[j - 1] * dt)

        def Q(i, j):
            r = M * (cj(i) * cj(j) - 1) / (lambdas[i - 1] +
                                           lambdas[j - 1]) / (cj(j) - ck) / (cj(i) - 1 / ck)
            return r if i != j else r.real

        # %% Get Gamma covariance matrix
        G1 = np.array([
            [2 * D1 * Q(1, 1), 0, 0, 0],
            [0, 2 * D2 * Q(2, 2), 0, 0],
            [0, 0, (D1 * (g + n1 - n2) + D2 * (g - n1 + n2)) *
             Q(3, 3) / g, -(D1 - D2) * (g + n1 - n2) * Q(3, 4) / g],
            [0, 0, -(D1 - D2) * (g - n1 + n2) * Q(4, 3) / g,
             (D1 * (g - n1 + n2) + D2 * (g + n1 - n2)) * Q(4, 4) / g]])
        Gfull = 2 * dt ** 2 * (1 - np.cos(2 * np.pi * k / M)) * U @ G1 @ Um1

        # print('U', U)
        # print('Um1', Um1)

    else:
        # g = np.sqrt((n1 - n2)**2 + 2 * n12**2)
        lambdas = np.array([-n1, -n1, -n2, -n2])

        def cj(j):
            return exp(lambdas[j - 1] * dt)

        def Q(i, j):
            r = M * (cj(i) * cj(j) - 1) / (lambdas[i - 1] +
                                           lambdas[j - 1]) / (cj(j) - ck) / (cj(i) - 1 / ck)
            return r if i != j else r.real

        # %% Get Gamma covariance matrix
        G1 = np.array([
            [2 * D1 * Q(1, 1), 0, 0, 0],
            [0, 2 * D1 * Q(2, 2), 0, 0],
            [0, 0, 2 * D2 * Q(3, 3), 0],
            [0, 0, 0, 2 * D2 * Q(4, 4)]])
        Gfull = 2 * dt ** 2 * (1 - np.cos(2 * np.pi * k / M)) * G1

    # if k == 1:
    #     print(f'Q11(k=1) = {Q(1,1)}')

    Cfull = np.zeros((4, 4)) if k > 0 else Gfull

    if not both:
        G = Gfull[:2, :2]
        C = Cfull[:2, :2]
    else:
        G, C = Gfull, Cfull

    # If inferring the angle, rotate the G matrix
    if rotation:
        S = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha), np.cos(alpha)]])
        if both:
            Z = np.zeros((2, 2))
            S = np.block([[S, Z], [Z, S]])

        # S = np.array([[np.cos(alpha), np.sin(alpha)],
        #               [-np.sin(alpha), np.cos(alpha)]])
        G = S @ G @ S.T
        C = S @ C @ S.T

    P = G.conj() - C.conj().T @ np.linalg.inv(G) @ C

    # print(f'n12={n12}, full G={G}')
    verbose = False
    if verbose:
        print(f'k={k}, n12={n12}')
        print(f'det(full G)={np.linalg.det(Gfull)}')
        print(f'eig(full G)={np.linalg.eigvals(Gfull)}')
        print('Gfull ', Gfull)
        # print('Hermitian: ',  G - G.conj().T)
        print('U=', U)
        print('Um1=', Um1)
        print('U* Um1', U @ Um1)

        print(f'det(G)={np.linalg.det(G)}')
        print(f'eig(G)={np.linalg.eigvals(G)}')
        print(f'det(G1)={np.linalg.det(G1)}')
        print(f'eig(G1)={np.linalg.eigvals(G1)}')
        print(f'G1={G1}')
        # if k == 2:
        raise RuntimeError('stop')

    # Make some checks. Do not check all k because may be heavy
    if k == 1:
        # G is a non-negative definite matrix
        det_G = np.linalg.det(G)
        if det_G < 0:
            logging.warning(
                f'Gamma matrix determinant is negative ({det_G} < 0). It is likely there is a problem in the code')
        # P is a non-negative positive definite matrix
        det_P = np.linalg.det(P)
        if det_P < 0:
            logging.warning(
                f'P matrix determinant is negative ({det_P} < 0). It is likely there is a problem in the code')

    Q = np.block([[G, C], [C.conj(), G.conj()]])
    Q_inv = np.linalg.inv(Q)

    # dRk = dRk / (np.sqrt(M) * dt)
    vec_right = np.vstack([dRk, dRk.conj()])
    vec_left = vec_right.conj().T  # np.hstack([dRk.conj().T, dRk.T])
    # print('dRk', dRk)
    # print('vl', vec_left)
    # print('vr', vec_right)

    # Calculate the log-likelihood
    d = dRk.shape[0]
    if d not in [2, 4]:
        raise RuntimeError(f'Unexpected vector dimension d={d}')

    ln_prob = (-d * log(pi) - 1 / 2 * log(np.linalg.det(G)) - 1 / 2 * log(np.linalg.det(P))
               - 1 / 2 * vec_left @ Q_inv @ vec_right)

    if ln_prob.shape != (1, 1):
        logging.warning(
            f'Log-likelihood matrix dimensions are incorrect. Check the supplied arrays. Got: {ln_prob}')
    ln_prob = ln_prob[0, 0]

    if abs(ln_prob.imag) > ATOL:
        logging.warning(
            'The imaginary part of the real likelihood is larger than tolerance. There might be an error. Result: {ln_prob}'.format(
                ln_prob=ln_prob))
    ln_prob = ln_prob.real

    return ln_prob


def get_ln_likelihood_func_2_particles_x_link(ks, M, dt, dRks=None, rotation=True, same_D=
False, both=False):
    """
    Returns ln of the likelihood function for all input data points as a function of
    specified parameters. The likelihood is not normalized over these parameters.
    """

    if not same_D:
        def ln_lklh(D1, D2, n1, n2, n12, alpha):
            ln_lklh_vals = [new_likelihood_2_particles_x_link_one_point(
                dRk=dRks[:, i, np.newaxis], k=ks[i], D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M,
                dt=dt, alpha=alpha, rotation=rotation, both=both, link=True) for i in range(len(
                ks))]
            ln_lklh_val = np.sum(ln_lklh_vals)
            return ln_lklh_val
    else:
        def ln_lklh(D1, n1, n2, n12, alpha):
            ln_lklh_vals = [new_likelihood_2_particles_x_link_one_point(
                dRk=dRks[:, i, np.newaxis], k=ks[i], D1=D1, D2=D1, n1=n1, n2=n2, n12=n12, M=M,
                dt=dt, alpha=alpha, rotation=rotation, both=both, link=True) for i in
                range(len(ks))]
            ln_lklh_val = np.sum(ln_lklh_vals)
            return ln_lklh_val
    return ln_lklh


def get_ln_likelihood_func_no_link(ks, M, dt, dRks=None, same_D=False, both=False):
    """
    Returns ln of the likelihood function for all input data points as a function of
    specified parameters. The likelihood is not normalized over these parameters.
    """
    n12 = 0
    if both:
        if same_D:
            def ln_lklh(D1, n1, n2):
                ln_lklh_vals = [new_likelihood_2_particles_x_link_one_point(
                    dRk=dRks[:, i, np.newaxis], k=ks[i], D1=D1, D2=D1, n1=n1, n2=n2, n12=n12, M=M,
                    dt=dt, both=True, link = False) for i in range(len(ks))]
                ln_lklh_val = np.sum(ln_lklh_vals)
                return ln_lklh_val
        else:
            def ln_lklh(D1, D2, n1, n2):
                ln_lklh_vals = [new_likelihood_2_particles_x_link_one_point(
                    dRk=dRks[:, i, np.newaxis], k=ks[i], D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M,
                    dt=dt, both=True, link = False) for i in range(len(ks))]
                ln_lklh_val = np.sum(ln_lklh_vals)
                return ln_lklh_val

    else:
        D2 = 1
        n2 = 1

        def ln_lklh(D1, n1):
            ln_lklh_vals = [new_likelihood_2_particles_x_link_one_point(
                dRk=dRks[:, i, np.newaxis], k=ks[i], D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M,
                dt=dt, both=False, link = False) for i in range(len(ks))]
            ln_lklh_val = np.sum(ln_lklh_vals)
            return ln_lklh_val

    return ln_lklh


def get_ln_likelihood_func_free_hookean_with_link_same_D(ks, M, dt, dRks):
    """
    Returns a ln of the likelihood function for all input data points as a function of
    specified parameters. The likelihood is not normalized over these parameters.
    """

    def ln_lklh(D1, n12, L0):
        ln_lklh_vals = [ln_likelihood_func_free_hookean_same_D_with_link_one_point(
            dRk=dRks[:, i, np.newaxis], k=ks[i], D1=D1, n12=n12, M=M,
            dt=dt, L0=L0) for i in range(len(ks))]

        ln_lklh_val = np.sum(ln_lklh_vals)

        return ln_lklh_val

    return ln_lklh


def get_ln_likelihood_func_free_hookean_no_link_same_D(ks, M, dt, dRks):
    """
    Returns a ln of the likelihood function for all input data points as a function of
    specified parameters. The likelihood is not normalized over these parameters.
    """

    def ln_lklh(D1):
        ln_lklh_vals = [ln_likelihood_func_free_hookean_same_D_no_link_one_point(
            dRk=dRks[:, i, np.newaxis], k=ks[i], D1=D1, M=M, dt=dt) for i in range(len(ks))]

        ln_lklh_val = np.sum(ln_lklh_vals)

        return ln_lklh_val

    return ln_lklh


def ln_likelihood_func_free_hookean_same_D_with_link_one_point(dRk, k=1, D1=1, n12=1, M=999,
                                                               dt=0.3, L0=1):
    """
    #TOUPDATE
    Calculate likelihood of one power spectrum observation z for the frequency k.
    Does not work for k = 0.

    Input:
    gamma --- viscosity, kg/s,
    z --- the value of the power spectrum component,
    n1, n2 , n12 --- variables starting with n are spring constants normalizedby gamma, i.e. n1 = k1 /gamma

    Definitions:
    R --- 4D vector of locations vs. time, {x1, y1, x2, y2} x N,
    dR --- 4D vector of displacements, shorter by 1 point,
    mu = <dR>,
    ksi = dR - mu --- stochastic part of the displacement,


    Important:
    - The calculations below will be wrong if one of lambda values is 0.
    # - the Gamma and C matrices are not multiplied by

    """

    ATOL = 1e-5
    dRk = copy.copy(dRk)

    if np.any(np.array((D1, n12, L0)) < 0):
        return ln_neg_infty

    # # %% Hard-code the eigenvalues and the diagonalizing matrix U
    # # Treat separately the case of n12=0 and n12>0
    # ck = exp(-2 * pi * 1j * k / M)
    # if n12 > ATOL:
    #     g = np.sqrt((n1 - n2) ** 2 + 4 * n12 ** 2)
    #     lambdas = np.array([-2 * n1, -2 * n2, -g - n1 - 2 * n12 - n2, g - n1 - 2 * n12 - n2]) / 2
    #     U = np.array([
    #         [0, 0, -2 * n12, 2 * n12],
    #         [2 * g, 0, 0, 0],
    #         [0, 0, g - n1 + n2, g + n1 - n2],
    #         [0, 2 * g, 0, 0]]) / 2 / g
    #
    #     Um1 = np.array([
    #         [0, 2 * n12, 0, 0],
    #         [0, 0, 0, 2 * n12],
    #         [-g - n1 + n2, 0, 2 * n12, 0],
    #         [g - n1 + n2, 0, 2 * n12, 0]]) / 2 / n12
    #
    #     # lambdas_test, U_test = np.linalg.eig(A)
    #
    #     def cj(j):
    #         return exp(lambdas[j - 1] * dt)
    #
    #     def Q(i, j):
    #         r = M * (cj(i) * cj(j) - 1) / (lambdas[i - 1] +
    #                                        lambdas[j - 1]) / (cj(j) - ck) / (cj(i) - 1 / ck)
    #         return r if i != j else r.real
    #
    #     # %% Get Gamma covariance matrix
    #     G1 = np.array([
    #         [2 * D1 * Q(1, 1), 0, 0, 0],
    #         [0, 2 * D2 * Q(2, 2), 0, 0],
    #         [0, 0, (D1 * (g + n1 - n2) + D2 * (g - n1 + n2)) *
    #          Q(3, 3) / g, -(D1 - D2) * (g + n1 - n2) * Q(3, 4) / g],
    #         [0, 0, -(D1 - D2) * (g - n1 + n2) * Q(4, 3) / g,
    #          (D1 * (g - n1 + n2) + D2 * (g + n1 - n2)) * Q(4, 4) / g]])
    #     Gfull = 2 * dt ** 2 * (1 - np.cos(2 * np.pi * k / M)) * U @ G1 @ Um1
    #
    #
    #
    # else:
    #     lambdas = np.array([-n1, -n1, -n2, -n2])
    #
    #     def cj(j):
    #         return exp(lambdas[j - 1] * dt)
    #
    #     def Q(i, j):
    #         r = M * (cj(i) * cj(j) - 1) / (lambdas[i - 1] +
    #                                        lambdas[j - 1]) / (cj(j) - ck) / (cj(i) - 1 / ck)
    #         return r if i != j else r.real
    #
    #     # %% Get Gamma covariance matrix
    #     G1 = np.array([
    #         [2 * D1 * Q(1, 1), 0, 0, 0],
    #         [0, 2 * D1 * Q(2, 2), 0, 0],
    #         [0, 0, 2 * D2 * Q(3, 3), 0],
    #         [0, 0, 0, 2 * D2 * Q(4, 4)]])
    #     Gfull = 2 * dt ** 2 * (1 - np.cos(2 * np.pi * k / M)) * G1
    #
    # # if k == 1:
    # #     print(f'Q11(k=1) = {Q(1,1)}')
    #
    # Cfull = np.zeros((4, 4)) if k > 0 else Gfull
    #
    # G = Gfull[:2, :2]
    # C = Cfull[:2, :2]

    # Covariance and correlation matrices G, C
    ck = exp(-2 * pi * 1j * k / M)
    c_phi = exp(-2 * D1 * dt / L0 ** 2)
    c_n12 = exp(-n12 * dt)
    avg_dx1k_abs_squared = M * dt ** 2 * (
            D1 * dt - (1 - cos(2 * pi * k / M)) / 4 * (
            L0 ** 2 * (c_phi ** 2 - 1) / np.abs(c_phi * ck - 1) ** 2
            + D1 * (c_phi ** 2 * c_n12 ** 4 - 1) / n12 / np.abs(c_phi * ck * c_n12 ** 2 - 1) ** 2
    )
    )
    G = np.identity(2) * avg_dx1k_abs_squared
    C = np.identity(2) * (M * D * dt ** 3 if k == 0 else 0)

    # # If inferring the angle, rotate the G matrix
    # if rotation:
    #     S = np.array([[np.cos(alpha), -np.sin(alpha)],
    #                   [np.sin(alpha), np.cos(alpha)]])
    #     # S = np.array([[np.cos(alpha), np.sin(alpha)],
    #     #               [-np.sin(alpha), np.cos(alpha)]])
    #     G = S @ G @ S.T
    #     C = S @ C @ S.T

    P = G.conj() - C.conj().T @ np.linalg.inv(G) @ C

    verbose = False
    if verbose:
        print(f'k={k}, n12={n12}')
        print(f'det(full G)={np.linalg.det(Gfull)}')
        print(f'eig(full G)={np.linalg.eigvals(Gfull)}')
        print('Gfull ', Gfull)
        # print('Hermitian: ',  G - G.conj().T)
        print('U=', U)
        print('Um1=', Um1)
        print('U* Um1', U @ Um1)

        print(f'det(G)={np.linalg.det(G)}')
        print(f'eig(G)={np.linalg.eigvals(G)}')
        print(f'det(G1)={np.linalg.det(G1)}')
        print(f'eig(G1)={np.linalg.eigvals(G1)}')
        print(f'G1={G1}')
        # if k == 2:
        raise RuntimeError('stop')

    # Make some checks. Do not check all k because may be heavy
    if k == 1:
        # G is a non-negative definite matrix
        det_G = np.linalg.det(G)
        if det_G < 0:
            logging.warning(
                f'Gamma matrix determinant is negative ({det_G} < 0). It is likely there is a problem in the code')
        # P is a non-negative positive definite matrix
        det_P = np.linalg.det(P)
        if det_P < 0:
            logging.warning(
                f'P matrix determinant is negative ({det_P} < 0). It is likely there is a problem in the code')

    Q = np.block([[G, C], [C.conj(), G.conj()]])
    Q_inv = np.linalg.inv(Q)

    vec_right = np.vstack([dRk, dRk.conj()])
    vec_left = vec_right.conj().T

    # Calculate the log-likelihood
    dim = 2
    ln_prob = (-dim * log(pi) - 1 / 2 * log(np.linalg.det(G)) - 1 / 2 * log(np.linalg.det(P))
               - 1 / 2 * vec_left @ Q_inv @ vec_right)

    if ln_prob.shape != (1, 1):
        logging.warning(
            f'Log-likelihood matrix dimensions are incorrect. Check the supplied arrays. Got: {ln_prob}')
    ln_prob = ln_prob[0, 0]

    if abs(ln_prob.imag) > ATOL:
        logging.warning(
            'The imaginary part of the real likelihood is larger than tolerance. There might be an error. Result: {ln_prob}'.format(
                ln_prob=ln_prob))
    ln_prob = ln_prob.real

    return ln_prob


def ln_likelihood_func_free_hookean_same_D_no_link_one_point(dRk, k=1, D1=1, M=999, dt=0.3):
    ATOL = 1e-5
    dRk = copy.copy(dRk)

    if D1 < 0:
        return ln_neg_infty

    # Covariance and correlation matrices G, C
    avg_dx1k_abs_squared = 2 * D1 * dt ** 3 * M
    G = np.identity(2) * avg_dx1k_abs_squared
    C = np.identity(2) * (2 * D * dt ** 3 * M if k == 0 else 0)

    P = G.conj() - C.conj().T @ np.linalg.inv(G) @ C

    verbose = False
    if verbose:
        print(f'k={k}, n12={n12}')
        print(f'det(full G)={np.linalg.det(Gfull)}')
        print(f'eig(full G)={np.linalg.eigvals(Gfull)}')
        print('Gfull ', Gfull)
        # print('Hermitian: ',  G - G.conj().T)
        print('U=', U)
        print('Um1=', Um1)
        print('U* Um1', U @ Um1)

        print(f'det(G)={np.linalg.det(G)}')
        print(f'eig(G)={np.linalg.eigvals(G)}')
        print(f'det(G1)={np.linalg.det(G1)}')
        print(f'eig(G1)={np.linalg.eigvals(G1)}')
        print(f'G1={G1}')
        # if k == 2:
        raise RuntimeError('stop')

    # Make some checks. Do not check all k because may be heavy
    if k == 1:
        # G is a non-negative definite matrix
        det_G = np.linalg.det(G)
        if det_G < 0:
            logging.warning(
                f'Gamma matrix determinant is negative ({det_G} < 0). It is likely there is a problem in the code')
        # P is a non-negative positive definite matrix
        det_P = np.linalg.det(P)
        if det_P < 0:
            logging.warning(
                f'P matrix determinant is negative ({det_P} < 0). It is likely there is a problem in the code')

    Q = np.block([[G, C], [C.conj(), G.conj()]])
    Q_inv = np.linalg.inv(Q)

    vec_right = np.vstack([dRk, dRk.conj()])
    vec_left = vec_right.conj().T

    # Calculate the log-likelihood
    dim = 2
    ln_prob = (-dim * log(pi) - 1 / 2 * log(np.linalg.det(G)) - 1 / 2 * log(np.linalg.det(P))
               - 1 / 2 * vec_left @ Q_inv @ vec_right)

    if ln_prob.shape != (1, 1):
        logging.warning(
            f'Log-likelihood matrix dimensions are incorrect. Check the supplied arrays. Got: {ln_prob}')
    ln_prob = ln_prob[0, 0]

    if abs(ln_prob.imag) > ATOL:
        logging.warning(
            'The imaginary part of the real likelihood is larger than tolerance. There might be an error. Result: {ln_prob}'.format(
                ln_prob=ln_prob))
    ln_prob = ln_prob.real

    return ln_prob


# def new_get_ln_likelihood_func_no_link(ks, M, dt, zs_x=None, zs_y=None):
#     """
#     Returns a log of the likelihood function for all input data points as a function of parameters (D1, D2, n1, n2, n12).
#     The likelihood is not normalized over these parameters.
#     """
#     D2 = 1
#     n2 = 1
#     n12 = 0
#
#     def ln_lklh(D1, n1):
#         ln_lklh_vals = []
#         if zs_x is not None:
#             ln_lklh_vals.append([likelihood_2_particles_x_link_one_point(
#                 z=z, k=k, D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M, dt=dt, alpha=0)
#                 for z, k in zip(zs_x, ks)])
#
#         if zs_y is not None:
#             ln_lklh_vals.append([likelihood_2_particles_x_link_one_point(
#                 z=z, k=k, D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M, dt=dt, alpha=1)
#                 for z, k in zip(zs_y, ks)])
#
#         ln_lklh_val = np.sum(ln_lklh_vals)
#
#         return ln_lklh_val
#
#     return ln_lklh


def estimate_evidence_integral(ln_posterior, MLE, lims=None):
    """
    In case of low dimensions (2), provide an estimate of the evidence integral by performing direct integration

    Input:
        ln_posterior, function:     log posterior distribution function
        MLE, 1D list or array
        lims, nx2 array:     integration limits for each variable
    """
    d = len(MLE)

    if d != 2:
        return np.nan

    max_ln_val = ln_posterior(*MLE)

    def rescaled(*args):
        r = ln_posterior(*args) - max_ln_val
        return np.exp(r)

    # Integrate
    if d == 2:
        lims = np.array([[0, 10 * val] for val in MLE])

        print(f'\nStart direct evidence integration for dim={d}')
        ev1 = integrate.dblquad(
            rescaled, lims[0, 0], MLE[0], lambda x: lims[1, 0], lambda x: MLE[1])
        print('Completed 1/4')

        ev2 = integrate.dblquad(
            rescaled, lims[0, 0], MLE[0], lambda x: MLE[1], lambda x: lims[1, 1])
        print('Completed 2/4')

        ev3 = integrate.dblquad(
            rescaled, MLE[0], lims[0, 1], lambda x: lims[1, 0], lambda x: MLE[1])
        print('Completed 3/4')

        ev4 = integrate.dblquad(
            rescaled, MLE[0], lims[0, 1], lambda x: MLE[1], lambda x: lims[1, 1])
        print('Integration terminated')

        ev = ev1 + ev2 + ev3 + ev4
    else:
        ev = np.nan

    # Rescale
    ln_evidence = max_ln_val + log(ev)

    return ln_evidence


def get_mean(k=1, D1=1, D2=3, n1=1, n2=1, n12=1, M=999, dt=0.3, alpha=0):
    """Get the mean of the stochastic variable corresponding to given springs, diffusivities and frequency"""


def get_ln_prior_func(rotation=True):
    # (D1, n1, D2=None, n2=None, n12=None):
    """
    Prior for the corresponding likelihoods.

    Also provide a sampler to sample from the prior.
    """
    ################# Diffusivities D1, D2 #################

    tau = 1e-2
    k = 2

    def eqn(z):
        # Condition: decrease on the right border as compared to mode is tau
        return z * exp(1 - z) - tau

    sol = root_scalar(eqn, bracket=[1, 1e5])
    z = sol.root
    theta_D = max_expected_D / z

    # print('theta_n', theta)
    # raise RuntimeError('stop')

    def ln_D_prior(D):
        """
        Gamma distribution
        """
        theta = theta_D
        return - gammaln(k) - k * log(theta) + (k - 1) * log(D) - D / theta

    def cdf_D_prior(D):
        theta = theta_D
        if D > 0:
            return gammainc(k, D / theta)  # gamma
        else:
            return 0

    ################# Localization strength n1, n2 #################
    # Gamma distribution. Set scale by right boundary

    tau = 1e-2
    k = 2

    def eqn(z):
        # Condition: decrease on the right border as compared to mode is tau
        return z * exp(1 - z) - tau

    sol = root_scalar(eqn, bracket=[1, 1e5])
    z = sol.root
    theta_n = max_expected_n / z

    # print('theta_n', theta)
    # raise RuntimeError('stop')

    def ln_n_prior(n):
        """
        Gamma distribution
        """
        theta = theta_n
        # return alpha * log(beta) - gammaln(alpha) - (alpha + 1) * log(n) - beta / n

        # return -log(n) - 1 / 2 * log(2 * pi * sigma2_n) - (log(n) - mu_n)**2 / 2 / sigma2_n
        # # inverse gamma distribution
        return - gammaln(k) - k * log(theta) + (k - 1) * log(n) - n / theta

    def cdf_n_prior(n):
        # return 1 / 2 * (1 + erf((log(n) - mu_n) / np.sqrt(2 * sigma2_n)))
        theta = theta_n
        if n > 0:
            return gammainc(k, n / theta)  # gamma
            # return gammaincc(alpha, beta / n)  # inverse-gamma
        else:
            return 0

    ################# Link strength n12 #################

    tau = 1e-2
    k = 2

    def eqn(z):
        # Condition: decrease on the right border as compared to mode is tau
        return z * exp(1 - z) - tau

    sol = root_scalar(eqn, bracket=[1, 1e5])
    z = sol.root
    theta_n12 = max_expected_n12 / z

    # if tau == 1 / 100:
    #     a = 7.638352067993813
    # else:
    #     # Conditions the right boundary of the interval. Only valid for k=2
    #     a = root_scalar(lambda a: exp(-1 + a) - a / tau, bracket=[1, 1e7])
    # theta = n12_interval[1] / a

    # a = 1
    # lam = n12_interval[1] / (tau**(-1 / (a + 1)) - 1)

    # Gamma distribution

    def ln_n_link_prior(n):
        """
        n_{12} link prior.
        A Lomax distribution.
        """
        theta = theta_n12
        return -gammaln(k) - k * log(theta) + (k - 1) * log(n) - n / theta
        # return log(a) - log(lam) - (a + 1) * log(1 + n / lam)

    def cdf_n_link_prior(n):
        """CDF for the prior"""
        theta = theta_n12
        return gammainc(k, n / theta)
        # return 1 - (1 + n / lam)**(-a)

    ################# Interaction angle alpha #################
    mu = 0
    sigma = np.pi / 4

    def ln_alpha_prior(alpha):
        return -1 / 2 * np.log(2 * np.pi * sigma ** 2) - (alpha - mu) ** 2 / 2 / sigma ** 2

    def cdf_alpha_prior(alpha):
        return 1 / 2 * (1 + erf((alpha - mu) / sigma / np.sqrt(2)))

    ################# Link length L0 #################

    tau = 1e-2
    k = 2

    def eqn(z):
        # Condition: decrease on the right border as compared to mode is tau
        return z * exp(1 - z) - tau

    sol = root_scalar(eqn, bracket=[1, 1e5])
    z = sol.root
    theta_L0 = max_expected_L0 / z

    # print('theta_n', theta)
    # raise RuntimeError('stop')

    def ln_L0_prior(L0):
        """
        Gamma distribution
        """
        theta = theta_L0
        return - gammaln(k) - k * log(theta) + (k - 1) * log(L0) - L0 / theta

    def cdf_L0_prior(L0):
        theta = theta_L0
        if L0 > 0:
            return gammainc(k, L0 / theta)  # gamma
        else:
            return 0

    ################# Assemble the prior function #################
    def ln_prior(D1, n1=None, D2=None, n2=None, n12=None, alpha=None, L0=None):
        ln_result = 0

        # D1
        if D1 <= 0:
            return ln_neg_infty
        ln_result += ln_D_prior(D1)

        # D2
        if D2 is not None:
            if D2 <= 0:
                return ln_neg_infty
            ln_result += ln_D_prior(D2)

        # n1
        if n1 is not None:
            if n1 <= 0:
                return ln_neg_infty
            ln_result += ln_n_prior(n1)

        # n2
        if n2 is not None:
            if n2 <= 0:
                return ln_neg_infty
            ln_result += ln_n_prior(n2)

        # n12
        if n12 is not None:
            if n12 < 0:
                return ln_neg_infty
            ln_result += ln_n_link_prior(n12)

        # angle alpha
        if alpha is not None:
            ln_result += ln_alpha_prior(alpha)

        # L0
        if L0 is not None:
            if L0 <= 0:
                return ln_neg_infty
            ln_result += ln_L0_prior(L0)

        return ln_result

    # Assemble a sampler from the prior
    def sample_from_the_prior(names):
        uni_sample = np.random.uniform(size=7)

        # Convert the uniform sample into parameter values by sampling from that equation
        sample = {}
        # if not link:
        #     cdfs = [cdf_D_prior, cdf_n_prior]
        #     names = ('D1', 'n1')
        # elif link and not rotation:
        #     names = ('D1', 'D2', 'n1', 'n2', 'n12')
        #     cdfs = [cdf_D_prior, cdf_D_prior, cdf_n_prior, cdf_n_prior, cdf_n_link_prior]
        # elif link and rotation:

        # names = ('D1', 'D2', 'n1', 'n2', 'n12', 'alpha', 'L0')
        cdfs = {'D1': cdf_D_prior, 'D2': cdf_D_prior, 'n1': cdf_n_prior,
                'n2': cdf_n_prior, 'n12': cdf_n_link_prior, 'alpha': cdf_alpha_prior, \
                'L0': cdf_L0_prior}

        for i, name in enumerate(names):
            if name is 'n12':
                bracket = [0, 1e5]
            elif name is 'alpha':
                bracket = [-100, 100]
            else:
                bracket = [1e-10, 1e5]
            sol = root_scalar(lambda y: cdfs[name](y) - uni_sample[i], bracket=bracket)
            sample[name] = sol.root

        return sample

    return ln_prior, sample_from_the_prior


def get_MLE(ln_posterior, names, sample_from_the_prior, hash_no_trial, link, verbose= \
        False, log_lklh=None, **kwargs):
    """
    Locate the MLE of the posterior. Estimate the evidence integral through Laplace approximation. Since the parameters have different scales, the search is conducted in the log space of the parameters.

    Make a guess of the starting point if none is supplied.

    Input:
        link, bool: whether a link between 2 particles should be considered in the likelihood
        method: {'BFGS', 'Nelder-Mead'}
        rotation: if True, also infer the orientation angle. Measured as counterclockwise rotation from x+ interaction direction
    """

    # number of random starting points for the MLE search before abandoning
    # Based on 99% chance estimate in a simple binomial model.

    d = len(names)
    tol = 1e-5  # search tolerance
    options_BFGS = {'disp': verbose, 'gtol': tol}
    fatol = 1e-2
    xatol = 1e-6
    options_NM = {'disp': verbose, 'maxiter': d * 1000,
                  'xatol': xatol, 'fatol': fatol, 'disp': False}

    bl_log_parameter_search = False
    np.random.seed()

    def to_list(dict):
        return [dict[key] for key in names]

    def to_dict(*args):
        return {a: args[i] for i, a in enumerate(names)}

    def minimize_me(args):
        """-ln posterior to minimize"""
        args_dict = {a: args[i] for i, a in enumerate(names)}
        # return -ln_lklh_func(**args_dict) - ln_prior(**args_dict)
        return -ln_posterior(**args_dict)

    #  Find MLE
    if verbose:
        print('Started MLE search')

    if 'method' in kwargs:
        warnings.warn(DeprecationWarning('Method parameter is deprecated'))

    # # method = 'Nelder-Mead'
    # if method == 'BFGS':
    #     options = options_BFGS
    # elif method == 'Nelder-Mead':
    #     options = options_NM
    # else:
    #     raise RuntimeError(f'Method "{method}" not implemented')

    def retry(i, _min_x, ln_evidence, fun):
        print('Found minimum: ', fun, 'with ln evidence', ln_evidence, ' at ', to_dict(*_min_x))
        return

    # def calculate_evidence_integral(points=None):
    #     """
    #     Calculate the evidence integral by numerical integration without Laplace approximation.
    #     Use the found MLE as a single breakpoint in the integration for the link model. Currently deactivated.
    #     """
    #     if link:
    #         return np.nan
    #     points_in = points.copy()
    #
    #     # These intervals need to be updated if the a priori working region changes
    #     all_integration_limits = {'D1': [0.01 / 10, 5 * 10], 'D2': [0.01 / 10, 5 * 10],
    #                               'n1': [0.01 / 10, 10 * 10], 'n2': [0.01 / 10, 10 * 10],
    #                               'n12': [0, 10 * 10]}
    #
    #     # Get the scale of the maximum of the posterior to normalize the integrand
    #     # def ln_lklh(x): -mini
    #     largest_ln_value = max([-minimize_me(point) for point in points])
    #
    #     # If fitting the link model, use only 1 breakpoint - the MLE
    #     # Else use all break points
    #     if link:
    #         points = sorted(points_in, key=lambda x: -minimize_me(x))[-1:]
    #
    #     # Filter break points be removing those that are too close, separately along each axis
    #     atol_points = 1e0
    #     points = zip(*points)  # combine break points per axis
    #     new_points = []
    #     for points_1d in points:
    #         points_1d = np.sort(points_1d)
    #         new_points_1d = [points_1d[0]]
    #         for p in points_1d[1:]:
    #             if p - new_points_1d[-1] >= atol_points:
    #                 new_points_1d.append(p)
    #         new_points.append(new_points_1d)
    #
    #     print('Break points for direct integration:', new_points)
    #
    #     def integrand(*args):
    #         """Renormalize the function before integration"""
    #         return exp(-minimize_me(args) - largest_ln_value)
    #
    #     tol = 1e-1
    #     opts = [{'points': el, 'epsabs': tol, 'epsrel': tol} for el in new_points]
    #     integration_limits = [all_integration_limits[name] for name in names]
    #     res = nquad(integrand, ranges=integration_limits, opts=opts)
    #     print('Integration output: ', res)
    #     # print('Largest ln value: ', largest_ln_value)
    #
    #     return log(res[0]) + largest_ln_value

    # def calculate_laplace_approximation(_min_x):
    #     """
    #     The function locally calculates the integral of a given minimum by using a Laplace approximation (approximating the shape of the peak by a non-uniform Gaussian).
    #     If the Hessian was calculated by the BFGS method, its values is used, otherwise, it is evaluated through finite differences.
    #     """
    #     if 'hess_inv' in _min_x and np.abs(np.linalg.det(_min_x.hess_inv) - 1) >= 1e-8:
    #         # Require that the returned Hessian has really been evaluated
    #         hess_inv = BFGS_hess_inv = _min_x.hess_inv
    #         det_inv_hess = BFGS_det_inv_hess = np.linalg.det(hess_inv)
    #
    #     # else:
    #     # # Otherwise, manually estimate the Hessian
    #     print('Manually calculating the Hessian')
    #     hess, info = nd.Hessian(minimize_me, full_output=True)(_min_x.x)
    #     man_hess = hess
    #     print('Info: ', info)
    #
    #     det_inv_hess = man_det_inv_hess = 1 / np.linalg.det(hess)
    #
    #     # hess_inv = np.nan
    #
    #     try:
    #         print('BFGS det. hess_inv: ', BFGS_det_inv_hess)
    #     except:
    #         pass
    #     print('Manual det. hess_inv: ', man_det_inv_hess, 1 / man_det_inv_hess,
    #           np.linalg.det(np.linalg.inv(man_hess)))
    #     try:
    #         print('\nBFGS inv. hess: ', BFGS_hess_inv)
    #     except:
    #         pass
    #     print('\n Manual inv. hess: ', np.linalg.inv(man_hess))
    #     try:
    #         print('Eigenvalues of the BFGS hessian: ', np.linalg.eigvals(BFGS_hess_inv))
    #     except:
    #         pass
    #     print('Eigenvalues of the manual hessian: ', np.linalg.eigvals(np.linalg.inv(man_hess)))
    #     print('And eigenvectors: ', np.linalg.eig(np.linalg.inv(man_hess)))
    #
    #     if det_inv_hess <= 0:
    #         # If the Hessian is non-positive, it was not a minimum
    #         return np.nan
    #
    #     # Calcualte evidence
    #     # Remember `minimize_me` is the negative log likelihood
    #     ln_model_evidence = ((d / 2) * log(2 * pi)
    #                          + 1 / 2 * log(det_inv_hess)
    #                          - minimize_me(_min_x.x))
    #     print(
    #         'Laplace approximation of the normalized integral ',
    #         exp((d / 2) * log(2 * pi)
    #             + 1 / 2 * log(det_inv_hess)))
    #     # print(' Hess_inv', det_inv_hess, hess_inv)
    #     return ln_model_evidence

    mins = []
    for i in range(max_tries):
        print(f'\nMLE search. Try {i + 1}/{max_tries}...')

        # On the first try, load the MLE guess from file. Else sample from the prior
        if i == 0:
            start_point, old_ln_value, success_load = load_MLE_guess(
                hash_no_trial=hash_no_trial, link=link)
            if not success_load or np.any([not name in start_point for name in names]):
                start_point = sample_from_the_prior(names)
                print(
                    f'Sampling an origin point from the prior:\n', start_point)
            else:
                print('Starting MLE guess loaded successfully:\n', (start_point, old_ln_value))

        #

        else:  # sample from the prior
            start_point = sample_from_the_prior(names)
            print(
                f'Sampling an origin point from the prior:\n', start_point)

        start_point_vals = to_list(start_point)

        # _min_x = minimize(fnc, start_point_vals, tol=1e-5, method=method,
        #                options=options)
        _min_x, fun, det_inv_hess, ln_evidence, success = get_MAP_and_hessian(minimize_me,
                                                                              start_point_vals)

        # # Save function for analysis
        # filename = os.path.join('function.pyc')
        # try:
        #     with open(filename, 'wb') as file:
        #         dill.dump(minimize_me, file, pickle.HIGHEST_PROTOCOL)
        # except Exception as e:
        #     logging.warning("Enoucntered unhandled exception while saving a data file: ", e)

        # Check if alpha angle is within [-pi/2; pi/2]
        if 'alpha' in names:
            def shift_angle(alpha):
                n = np.floor(abs(alpha) / np.pi + 0.5)
                return alpha - np.sign(alpha) * n * np.pi, n != 0

            alpha_ind = names.index('alpha')
            alpha_old = _min_x[alpha_ind]
            alpha, shifted = shift_angle(_min_x[alpha_ind])
            if shifted:
                print(
                    f'Interaction angle shifted to [-pi/2, pi/2] interval from {alpha_old} to '
                    f'{alpha}, and then recalculated.')
                _min_x[alpha_ind] = alpha
                # _min_x = minimize(fnc, _min_x.x, tol=1e-5, method=method,
                #                options=options)
                _min_x, fun, det_inv_hess, ln_evidence, success = get_MAP_and_hessian(minimize_me,
                                                                                      _min_x)

        print('Result. Function value:\t', fun, '\tLocation:\t', to_dict(*_min_x))

        # print(np.array(_min_x.x).squeeze())
        grad = nd.Gradient(minimize_me)(_min_x)
        grad_norm = np.max(abs(grad))
        print('\nGradient in the minimum:\t', grad, '\tMax. norm:\t', grad_norm)
        print('\nManual Jacobian:\t', grad)
        # print('Calculated Jacobian:\t', _min_x.jac, '\n')
        # print('A', out)

        # Store the found point if the hessian is not diagonal (because it's not a new point then)
        # if np.abs(np.linalg.det(_min_x.hess_inv)) >= 1e-8:
        GRAD_NORM_TOL = 1
        if grad_norm < GRAD_NORM_TOL:
            pass
            # ln_evidence = calculate_laplace_approximation(_min_x)

        else:
            print(f'Warning! Gradient too large (norm = {grad_norm}). Check your prior! \nThe '
                  f'current result will be ignored')
            ln_evidence = np.nan

        # # if not _min_x.success:
        # print('Relaunching from the minimum')
        #
        # min1 = minimize(fnc, _min_x.x, tol=1e-5, method=method, options=options_BFGS)
        # print('\nBFGS Restart from found point:\n', min1)
        # calculate_laplace_approximation(min1)
        #
        # minNM = minimize(fnc, _min_x.x, tol=1e-5, method='Nelder-Mead', options=options_NM)
        # print('Neldear-Mead from the found point:\n', minNM)
        # calculate_laplace_approximation(minNM)
        # #
        # # sys.exit(0)

        element = (fun, ln_evidence, _min_x, success)
        mins.append(element)

        # Plot to check the minimum
        if plot:
            print('Plotting...')

            def plot_func(xs, dim):
                out = []
                for x in xs:
                    args = _min_x.copy()
                    args[dim] = x
                    out.append(minimize_me(args))
                return out

            # step = 0.1
            for dim in range(len(names)):
                x_d = _min_x[dim]
                if names[dim] is not 'alpha':
                    xs = np.linspace(np.min([x_d * 0.5, 0.1]),
                                     np.max([x_d * 2, 1]), num=100, endpoint=True)
                else:
                    xs = np.linspace(-np.pi,
                                     np.pi, num=100, endpoint=True)
                plt.figure(6 + int(link), clear=True)
                plt.plot(xs, plot_func(xs, dim), label=names[dim])  # , color=color_sequence[dim])
                plt.scatter(x_d, plot_func([x_d], dim)[0])  # , color=color_sequence[dim])

                plt.legend()
                plt.show(block=False)
                # import time
                # time.sleep(3)
                # print('Plotting... done!')

                # if link:
                #     # print(M, link)
                #     # if link and M == 100:
                #     #     raise RuntimeError('stop')
                #     raise RuntimeError('Stop after plot')
                # plt.xlim([0, 40])
                figname = f'minimum_{i}_{dim}.png'
                plt.tight_layout()
                plt.savefig(figname)
        # else:
        #     ln_model_evidence = ln_true_evidence

        # Estimate how many of the best values are close
        mins.sort(key=itemgetter(0))
        fun_vals = [min[0] for min in mins]
        diffs = np.array([np.abs(fun_vals[i] - fun_vals[0]) for i in range(1, len(fun_vals))])
        # add 1 to count the value itself
        times_best_found = np.sum(diffs < MINIMA_CLOSE_ATOL) + 1
        print(
            f'\n(Number of best minima so far) / (stop criterion): {times_best_found} /'
            f' {SAME_MINIMA_STOP_CRITERION}')
        if times_best_found >= SAME_MINIMA_STOP_CRITERION:
            break

        retry(i, _min_x, ln_evidence, fun)

    # Sort the results
    mins.sort(key=itemgetter(0))

    # Calculate evidence integral
    ln_true_evidence = np.nan

    tries = i + 1
    print(f'\nFound the following (minimi, ln_evidence) in {tries} tries:\n', [
        (min[0], min[1]) for min in mins])

    # Estimate how many of the best values are close
    fun_vals = [min[0] for min in mins]
    diffs = np.array([np.abs(fun_vals[i] - fun_vals[0]) for i in range(1, len(fun_vals))])
    # add 1 to count the value itself
    times_best_found = np.sum(diffs < MINIMA_CLOSE_ATOL) + 1
    # Store the value in a file.
    # frac is the fraction of tries that the best minimum was found
    frac = times_best_found / tries
    save_number_of_close_values(link, times_best_found, tries, frac)
    if times_best_found < SAME_MINIMA_STOP_CRITERION:
        logging.warning('The minimum number of same minima stop criterion not satisfied!')

    # if np.isnan(ln_true_evidence):
    # Choose the best valid MLE (the det hess should be > 0 for it to be a minimum)
    # success = False
    for fun, ln_laplace_evidence, _min_x, success in mins:
        if success:

            if not np.isnan(ln_laplace_evidence):
                # If this was not a minimum, the returned value will be nan.
                # success = True
                break
    # Compare with true minimum if calculated and take the largest.
    # This is because the Laplace approximation always provides a lower-bound estimate
    # if np.isnan(ln_true_evidence):
    #     ln_model_evidence = ln_laplace_evidence
    # elif np.isnan(ln_laplace_evidence):
    #     ln_model_evidence = ln_true_evidence
    # else:
    ln_model_evidence = np.nanmax([ln_laplace_evidence, ln_true_evidence])
    success = not np.isnan(ln_model_evidence)
    # if success:
    #     print('Rerunning BFGS on the best _min_x:\n',
    #           minimize(fnc, _min_x.x, tol=1e-5, method='BFGS',
    #                    options=options_BFGS))

    # Restore the parameters from log
    if bl_log_parameter_search:
        MLE = {names[i]: exp(_min_x[i]) for i in range(len(_min_x))}
    else:
        MLE = {names[i]: _min_x[i] for i in range(len(_min_x))}

    if success:
        print(
            f'MLE search with (link = {link}) converged.\nTaking the best point seen so far:\n',
            (fun, MLE))
        try:
            lam, v = np.linalg.eig(np.linalg.inv(_min_x.hess_inv))
            # print(f'Eigenvalues and eigenvectors of the Hessian for the MLE:\n', )
            # print('Eigenvalues of the Hessian: ', lam)
            # print('Eigenvectors: ', v)
        except Exception:
            pass
        # Save the MLE guess for further use
        save_MLE_guess(hash_no_trial=hash_no_trial, MLE_guess=MLE,
                       ln_posterior_value=-fun, link=link)
    else:
        print(
            f'MLE search procedure with link={link} failed to converge in {tries} tries.')

    # sys.exit(0)
    if verbose:
        1
        # print('Full results:\n', _min_x)
        # print('det_inv_hess: ', det_inv_hess)

    return MLE, ln_model_evidence, _min_x, success


# def pymc_MLE(ks,
#              M,
#              dt,
#              dRks,
#              rotation=True,
#              same_D=False,
#              both=False,
#              ln_posterior=None,
#              names=None,
#              sample_from_the_prior=None,
#              hash_no_trial=None,
#              link=True,
#              method='BFGS',
#              verbose=False,
#              log_lklh=None,
#              ):
#     # if not same_D:
#     #     def ln_lklh(D1, D2, n1, n2, n12, alpha):
#     #         ln_lklh_vals = [new_likelihood_2_particles_x_link_one_point(
#     #             dRk=dRks[:, i, np.newaxis], k=ks[i], D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M,
#     #             dt=dt, alpha=alpha, rotation=rotation, both=both, link=True) for i in range(len(
#     #             ks))]
#     #         ln_lklh_val = np.sum(ln_lklh_vals)
#     #         return ln_lklh_val
#     # else:
#     #     def ln_lklh(D1, n1, n2, n12, alpha):
#     #         ln_lklh_vals = [new_likelihood_2_particles_x_link_one_point(
#     #             dRk=dRks[:, i, np.newaxis], k=ks[i], D1=D1, D2=D1, n1=n1, n2=n2, n12=n12, M=M,
#     #             dt=dt, alpha=alpha, rotation=rotation, both=both, link=True) for i in
#     #             range(len(ks))]
#     #         ln_lklh_val = np.sum(ln_lklh_vals)
#     #         return ln_lklh_val
#     # return ln_lklh
#     # print(ks)
#     # print(dRks)
#
#     # %%% Parameters of the prior
#     ################# Diffusivities D1, D2 #################
#
#     def get_theta(tau, max_expected):
#         # Condition: decrease on the right border as compared to mode is tau
#         # Assuming gamma distribution
#         sol = root_scalar(lambda z: z * exp(1 - z) - tau, bracket=[1, 1e5])
#         z = sol.root
#         theta = max_expected / z
#         return theta
#
#     tau_D = 1e-2
#     k_D = 2
#     tau_n = 1e-2
#     k_n = 2
#     tau_n12 = 1e-2
#     k_n12 = 2
#     mu_alpha = 0
#     sigma_alpha = np.pi / 4
#
#     ### Try a search based on PyMC
#     def to_1D(x):
#         return np.vstack([np.real(x), np.imag(x)])
#
#     if link:
#         # print(log_lklh)
#         with pm.Model() as model:
#             # Priors
#             D1 = pm.Gamma('D1', alpha=k_D, beta=1 / get_theta(tau_D, max_expected_D))
#             D2 = pm.Gamma('D2', alpha=k_D, beta=1 / get_theta(tau_D, max_expected_D))
#             n1 = pm.Gamma('n1', alpha=k_n, beta=1 / get_theta(tau_n, max_expected_n))
#             n2 = pm.Gamma('n2', alpha=k_n, beta=1 / get_theta(tau_n, max_expected_n))
#             n12 = pm.Gamma('n12', alpha=k_n12, beta=1 / get_theta(tau_n12, max_expected_n12))
#             alpha = pm.Normal('alpha', mu=mu_alpha, sigma=sigma_alpha)
#
#             for ind in range(2):  # len(ks)):
#
#                 k = ks[ind]
#                 dRk = dRks[:, ind, np.newaxis]
#
#                 # def log_lklh(n1, n2, n12, D1, D2, alpha):
#                 # ck = exp(-2 * pi * 1j * k / M)
#                 Re_ck = np.cos(-2 * pi * k / M)
#                 Im_ck = np.sin(-2 * pi * k / M)
#                 # if link:
#
#                 g = pm.math.sqrt((n1 - n2) ** 2 + 4 * n12 ** 2)
#
#                 # print(model)
#                 # print(model.basic_RVs)
#
#                 lambdas = tt.stack([-2 * n1, -2 * n2,
#                                     -g - n1 - 2 * n12 - n2, g - n1 - 2 * n12 - n2]) / 2
#                 U = tt.stacklists([
#                     [0, 0, -2 * n12, 2 * n12],
#                     [2 * g, 0, 0, 0],
#                     [0, 0, g - n1 + n2, g + n1 - n2],
#                     [0, 2 * g, 0, 0]]) / 2 / g
#
#                 Um1 = tt.stacklists([
#                     [0, 2 * n12, 0, 0],
#                     [0, 0, 0, 2 * n12],
#                     [-g - n1 + n2, 0, 2 * n12, 0],
#                     [g - n1 + n2, 0, 2 * n12, 0]]) / 2 / n12
#
#                 def cj(j):
#                     return tt.exp(lambdas[j - 1] * dt)
#
#                 def s_abs_2(i):
#                     r = (cj(i) - Re_ck) ** 2 + Im_ck ** 2
#                     return 1 / r
#
#                 def Re_Q(i, j):
#                     """
#                     Manually calculate the Real and Imaginary parts of Q because theano does not
#                     work with complex numbers
#                     """
#                     r = M * (cj(i) * cj(j) - 1) / (lambdas[i - 1] + lambdas[j - 1])
#                     r *= s_abs_2(i) * s_abs_2(j)
#                     r *= 1 + cj(i) * cj(j) - cj(i) * Re_ck - cj(j) * Re_ck
#
#                     return r
#
#                 def Im_Q(i, j):
#                     """
#                     Manually calculate the Real and Imaginary parts of Q because theano does not
#                     work with complex numbers
#                     """
#                     r = M * (cj(i) * cj(j) - 1) / (lambdas[i - 1] + lambdas[j - 1])
#                     r *= s_abs_2(i) * s_abs_2(j)
#                     r *= 1 + cj(i) * cj(j) + cj(i) * Im_ck - cj(j) * Im_ck
#
#                     return r
#
#                 # Gamma covariance matrix
#                 def G1(real):
#                     Q = Re_Q if real else Im_Q
#
#                     G1 = tt.stacklists([
#                         [2 * D1 * Q(1, 1), 0, 0, 0],
#                         [0, 2 * D2 * Q(2, 2), 0, 0],
#                         [0, 0, (D1 * (g + n1 - n2) + D2 * (g - n1 + n2)) *
#                          Q(3, 3) / g, -(D1 - D2) * (g + n1 - n2) * Q(3, 4) / g],
#                         [0, 0, -(D1 - D2) * (g - n1 + n2) * Q(4, 3) / g,
#                          (D1 * (g - n1 + n2) + D2 * (g + n1 - n2)) * Q(4, 4) / g]])
#                     return G1
#
#                 def Gfull(real):
#                     G = 2 * dt ** 2 * (1 - np.cos(2 * np.pi * k / M)) * U.dot(G1(real)).dot(Um1)
#                     if not both:
#                         return G[:2, :2]
#                     else:
#                         return G
#
#                 def Cfull(real):
#                     C = Gfull(real) * np.float(k != 0)
#                     if not both:
#                         return C[:2, :2]
#                     else:
#                         return C
#
#                 #
#                 # # # else:
#                 # # #     # g = np.sqrt((n1 - n2)**2 + 2 * n12**2)
#                 # # #     lambdas = np.array([-n1, -n1, -n2, -n2])
#                 # # #
#                 # # #     def cj(j):
#                 # # #         return exp(lambdas[j - 1] * dt)
#                 # # #
#                 # # #     def Q(i, j):
#                 # # #         r = M * (cj(i) * cj(j) - 1) / (lambdas[i - 1] +
#                 # # #                                        lambdas[j - 1]) / (cj(j) - ck) / (cj(i) - 1 / ck)
#                 # # #         return r if i != j else r.real
#                 # # #
#                 # # #     # %% Get Gamma covariance matrix
#                 # # #     G1 = np.array([
#                 # # #         [2 * D1 * Q(1, 1), 0, 0, 0],
#                 # # #         [0, 2 * D1 * Q(2, 2), 0, 0],
#                 # # #         [0, 0, 2 * D2 * Q(3, 3), 0],
#                 # # #         [0, 0, 0, 2 * D2 * Q(4, 4)]])
#                 # # #     Gfull = 2 * dt ** 2 * (1 - np.cos(2 * np.pi * k / M)) * G1
#                 # # #
#                 # # # # if k == 1:
#                 # # # #     print(f'Q11(k=1) = {Q(1,1)}')
#                 # #
#                 # Cfull = np.zeros((4, 4)) if k > 0 else Gfull
#                 #
#                 # if not both:
#                 #     G = Gfull[:2, :2]
#                 #     C = Cfull[:2, :2]
#                 # else:
#                 #     G, C = Gfull, Cfull
#
#                 # If inferring the angle, rotate the G matrix
#                 if rotation:
#                     S = tt.stacklists([[tt.cos(alpha), -tt.sin(alpha)],
#                                        [tt.sin(alpha), tt.cos(alpha)]])
#                     if both:
#                         Z = tt.zeros((2, 2))
#                         S = tt.stacklists([[S, Z], [Z, S]])
#                 else:
#                     S = tt.ones((4, 4))
#
#                 def G(real):
#                     return S.dot(Gfull(real)).dot(S.T)
#
#                 def C(real):
#                     return S.dot(Cfull(real)).dot(S.T)
#
#                 Vxx = (G(1) + C(1)) / 2
#                 Vxy = (-G(0) + C(0)) / 2
#                 Vyx = (G(0) + C(0)) / 2
#                 Vyy = (G(1) - C(1)) / 2
#                 # print(tt.ones_like(Vxx[0]))
#
#                 full_cov_real = tt.concatenate([
#                     tt.concatenate([Vxx, Vxy], axis=1),
#                     tt.concatenate([Vyx, Vyy], axis=1)],
#                     axis=0)
#
#                 shape = 8 if both else 4
#                 mu = np.zeros(shape=shape)
#
#                 pm.MvNormal('k' + str(k), mu=mu, cov=full_cov_real, observed=to_1D(dRk))
#
#             print('Start sampling')
#             with stopwatch('Traces: '):
#                 traces = pm.sample(10)
#             print('Results: ', traces)
#
#             print('Starting MAP calculation')
#             with stopwatch('MAP'):
#                 map = pm.find_MAP(model=model, progressbar=True)
#             print('MAP: ', map)


def get_MAP_and_hessian(minimize_me, x0, verbose=False, tol=1e-5):
    """
    Get MLE and reliably calculate the Hessian determinant (through rescaling).

    Returns:
        (_min, det_inv_hess, success)
    """
    with stopwatch('MAP and Hessian calculation'):
        min_scale = tol
        d = len(x0)
        options_NM = {'disp': verbose, 'xatol': tol, 'fatol': tol, 'maxiter': 1000 * d}
        options_BFGS = {'disp': verbose, 'gtol': tol}

        with stopwatch('Minimum search (initial)'):
            _min = minimize(minimize_me, x0, method='BFGS', options=options_BFGS)
        hess_diag = nd.Hessdiag(minimize_me)(_min.x)
        if verbose:
            print('NM search 1:\n', _min)
            print('Det. inv. Hessian:\t', 1 / np.linalg.det(nd.Hessian(minimize_me)(_min.x)))
            print('\nDiagonal Hessian: ', hess_diag)

        # Rescale
        shift = _min.x
        scales = np.array([np.max([np.abs(hd) ** (-1 / 2), min_scale]) for hd in hess_diag])
        inv_hessian_rescale_factor = (scales ** 2).prod()
        if verbose:
            print('\n\nShift:\t', shift)
            print('Scales:\t', scales)
            print('\nAfter rescaling:')

        def rescaled_minimize_me(args):
            args = args * scales + shift
            return minimize_me(args)

        # Rerun search
        with stopwatch('Minimum search (rescaled)'):
            _min = minimize(rescaled_minimize_me, np.zeros_like(_min.x), method='Nelder-Mead',
                            options=options_NM)
        if verbose:
            print('\nNM search 2', _min)
            print('\nNew Hess:\n', nd.Hessian(rescaled_minimize_me)(_min.x))

        # Evaluate inverse Hessian determinant
        with stopwatch('Inverse Hessian calculation'):
            det_inv_hess = (1 / np.linalg.det(nd.Hessian(rescaled_minimize_me)(_min.x))
                            * inv_hessian_rescale_factor)
        if verbose:
            print('Det. inv. Hessian:\t', det_inv_hess)
            print('\n\n')

        if not _min.success:
            logging.warning('MAP search failed to converge')

        # Calculate evidence
        ln_evidence = ((d / 2) * log(2 * pi)
                       + 1 / 2 * log(det_inv_hess)
                       - _min.fun)

    return _min.x * scales + shift, _min.fun, det_inv_hess, ln_evidence, _min.success
