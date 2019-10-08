"""
Calculate likelihoods that will be used for model comparison.
Those are mainly likelihoods for periodogram components.
"""

import copy
import functools
import logging
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
from scipy.optimize import minimize, root_scalar
from scipy.special import erf, gamma, gammaincc, gammaln, logsumexp

# from plot import plot_periodogram
from support import (delete_data, hash_from_dictionary, load_data,
                     load_MLE_guess, save_data, save_MLE_guess,
                     save_number_of_close_values)

ln_infty = - 20 * log(10)

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
            new_js = self.js + other.js     # list concat
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
        new_array = self.array**pow
        return matrix(new_array)


@functools.lru_cache(maxsize=128)    # Cache the return values
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
            f'Unable to calcualte eigenvalues of the A matrix. Check out the n12 value ({n12}). The matrix: ', A)
        raise(e)
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
                J = M * dt**3 / 2
            else:
                c1 = exp(lamb * dt)
                J = -(((M * dt ** 2) * ((1 - c1 ** 2) * (1 - cos((2 * pi * k) / M)))) / (
                    (2 * lamb) * (1 + c1 ** 2 - 2 * c1 * cos((2 * pi * k) / M))))

        else:
            lamb1, lamb2 = lambdas[[i, j]]
            J = -(((M * dt ** 2) * ((1 - exp(lamb1 * dt) * exp(lamb2 * dt)) * (1 + exp(lamb1 * dt) * exp(lamb2 * dt) - (exp(lamb1 * dt) + exp(lamb2 * dt)) * cos((2 * pi * k) / M)) * (1 - cos((2 * pi * k) / M)))) /
                  ((lamb1 + lamb2) * ((1 + exp(lamb1 * dt) ** 2 - 2 * exp(lamb1 * dt) * cos((2 * pi * k) / M)) * (1 + exp(lamb2 * dt) ** 2 - 2 * exp(lamb2 * dt) * cos((2 * pi * k) / M)))))
        return J

    def sigma2s_row(alpha, k):
        """Return sigma2 values that correspond to the component alpha and frequency k.
        The result is divided by (M*dt) to directly give the power spectrum
        """
        diag_J = np.diag([J_class(i, k, J) for i in range(1, 5)])
        mat = matrix(U) @ matrix(diag_J) @ matrix(Um1)

        new_mat = [((mat @ matrix(b[:, i, np.newaxis]))**2).array for i in range(4)]
        sigma2s = np.concatenate(new_mat, axis=1)

        return sigma2s[alpha, :] / (M * dt)

    def sigma2s_full(k):
        """Return sigma2 values that correspond to the component alpha and frequency k.
        The result is divided by (M*dt) to directly give the power spectrum
        """
        diag_J = np.diag([J_class(i, k, J) for i in range(1, 5)])
        mat = matrix(U) @ matrix(diag_J) @ matrix(Um1)

        new_mat = [((mat @ matrix(b[:, i, np.newaxis]))**2).array for i in range(4)]
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


def likelihood_2_particles_x_link_one_point(z, k=1, D1=1, D2=3, n1=1, n2=1, n12=1, M=999, dt=0.3, alpha=0):
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
        return ln_infty

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
            logging.warn('All input sigma2 are effectively zero. Returning 0 probability. Input lg sigmas: ' +
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


def get_ln_likelihood_func_2_particles_x_link(ks, M, dt, zs_x=None, zs_y=None):
    """
    Returns a log_10 of the likelihood function for all input data points as a function of parameters (D1, D2, n1, n2, n12).
    The likelihood is not normalized over these parameters.

    Use alpha to specifythe coordinate for the likelihood:
        alpha = 0 --- x1
        alpha = 1 --- y1
        alpha = [0,1] --- both x1 and y1
    """
    # if isinstance(alpha, int):
    #     alpha = [alpha]

    def ln_lklh(D1, D2, n1, n2, n12):
        ln_lklh_vals = []
        if zs_x is not None:
            ln_lklh_vals.append([likelihood_2_particles_x_link_one_point(
                z=z, k=k, D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M, dt=dt, alpha=0)
                for z, k in zip(zs_x, ks)])

        if zs_y is not None:
            ln_lklh_vals.append([likelihood_2_particles_x_link_one_point(
                z=z, k=k, D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M, dt=dt, alpha=1)
                for z, k in zip(zs_y, ks)])

        ln_lklh_val = np.sum(ln_lklh_vals)

        return ln_lklh_val

    return ln_lklh


def get_ln_likelihood_func_no_link(ks, M, dt, zs_x=None, zs_y=None):
    """
    Returns a log_10 of the likelihood function for all input data points as a function of parameters (D1, D2, n1, n2, n12).
    The likelihood is not normalized over these parameters.

    Use alpha to specifythe coordinate for the likelihood:
        alpha = 0 --- x1
        alpha = 1 --- y1
        alpha = [0,1] --- both x1 and y1
    """
    D2 = 1
    n2 = 1
    n12 = 0

    def ln_lklh(D1, n1):
        ln_lklh_vals = []
        if zs_x is not None:
            ln_lklh_vals.append([likelihood_2_particles_x_link_one_point(
                z=z, k=k, D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M, dt=dt, alpha=0)
                for z, k in zip(zs_x, ks)])

        if zs_y is not None:
            ln_lklh_vals.append([likelihood_2_particles_x_link_one_point(
                z=z, k=k, D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M, dt=dt, alpha=1)
                for z, k in zip(zs_y, ks)])

        ln_lklh_val = np.sum(ln_lklh_vals)

        return ln_lklh_val

    return ln_lklh


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


def get_ln_prior_func():
    # (D1, n1, D2=None, n2=None, n12=None):
    """
    Prior for the corresponding likelihoods.

    Also provide a sampler to sample from the prior.
    """

    beta = 0.5  # um^2/s
    a = 0.5     # no units

    def ln_D_prior(D):
        """D_i prior"""
        return a * log(beta) - gammaln(a) - (a + 1) * log(D) - beta / D

    def cdf_D_prior(D):
        if D > 0:
            return gammaincc(a, beta / D)
        else:
            return 0

    n0 = 10**3      # s^{-1}, spring constant prior scale
    sigma_n = log(10) / 2
    mu_n = log(n0) + sigma_n**2

    def ln_n_prior(n):
        """
        n_i prior

        This prior is on log values of n allowing very wide distributions. I need to make them smaller, so that n1 = 1e-20 be not accessible..

        Currently it is a Gaussian distribution for log(n), which does not prohibit 0. I must change it.

        """
        return -log(n) - 1 / 2 * log(2 * pi * sigma_n**2) - (log(n) - mu_n)**2 / 2 / sigma_n**2

    def cdf_n_prior(n):
        return 1 / 2 * (1 + erf((log(n) - mu_n) / np.sqrt(2) / sigma_n))

    a = 1

    def ln_n_link_prior(n):
        """
        n_{12} link prior.
        A Lomax distribution.
        """

        return log(a) - log(n0) - (a + 1) * log(1 + n / n0)

    def cdf_n_link_prior(n):
        """CDF for the prior"""
        return 1 - (1 + n / n0)**(-a)

    # Assemble the prior function
    def ln_prior(D1, n1, D2=None, n2=None, n12=None):
        ln_result = 0
        if np.any(np.array([D1, n1]) <= 0):
            return ln_infty
        ln_result += ln_D_prior(D1)
        ln_result += ln_n_prior(n1)

        if all(v is not None for v in (D2, n2, n12)):
            if np.any(np.array((D2, n2)) <= 0) or n12 < 0:
                return ln_infty
            ln_result += ln_D_prior(D2)
            ln_result += ln_n_prior(n2)
            ln_result += ln_n_link_prior(n12)
        return ln_result

    # Assemble a sampler from the prior
    def sample_from_the_prior(link):
        """
        Parameters:
        link {bool} - set to true if need a prior with link
        """
        uni_sample = np.random.uniform(size=5)

        # Convert the uniform sample into parameter values by sampling from that equation
        sample = {}
        if not link:
            cdfs = [cdf_D_prior, cdf_n_prior]
            names = ('D1', 'n1')
        else:
            names = ('D1', 'D2', 'n1', 'n2', 'n12')
            cdfs = [cdf_D_prior, cdf_n_prior, cdf_D_prior, cdf_n_prior, cdf_n_link_prior]

        for i, (cdf, name) in enumerate(zip(cdfs, names)):
            if name is 'n12':
                bracket = [0, 1e20]
            else:
                bracket = [1e-10, 1e20]
            sol = root_scalar(lambda y: cdf(y) - uni_sample[i], bracket=bracket)
            sample[name] = sol.root

        return sample

    return ln_prior, sample_from_the_prior


def get_MLE(ks, M, dt, link, hash_no_trial, zs_x=None, zs_y=None, start_point=None, verbose=False, method='BFGS'):
    """
    Locate the MLE of the posterior. Estimate the evidence integral through Laplace approximation. Since the parameters have different scales, the search is conducted in the log space of the parameters.

    Make a guess of the starting point if none is supplied.

    Input:
        link, bool: whether a link between 2 particles should be considered in the likelihood
        method: 'BFGS'. 'Nelder-Mead' is not yet implemented.
    """

    # number of random starting points for the MLE search before abandoning
    # Based on 99% chance estimate in a simple binomial model.
    # tries = 2 + 1 if not link else 40 + 1
    tries = 2 + 1
    tol = 1e-5  # search tolerance
    grad_tol = tol * 10  # gradient check tolerance
    ln_model_evidence = np.nan
    success = True
    bl_log_parameter_search = False
    # prob_new_start = 2 / 3
    np.random.seed()

    if link:
        names = ('D1', 'D2', 'n1', 'n2', 'n12')

        def to_dict(D1, D2, n1, n2, n12):
            return {key: val for key, val in zip(names, (D1, D2, n1, n2, n12))}
    else:
        names = ('D1', 'n1')

        def to_dict(D1, n1):
            return {key: val for key, val in zip(names, (D1, n1))}

    def to_list(dict):
        return [dict[key] for key in names]

    d = len(names)

    # Choose the appropriate likelihood
    if link:
        ln_lklh_func = get_ln_likelihood_func_2_particles_x_link(
            ks=ks, zs_x=zs_x, zs_y=zs_y, M=M, dt=dt)

        start_point_est = {'D1': 1, 'n1': 1e3, 'D2': 1, 'n2': 1e3, 'n12': 1e3}
    else:
        ln_lklh_func = get_ln_likelihood_func_no_link(
            ks=ks, zs_x=zs_x, zs_y=zs_y, M=M, dt=dt)

        start_point_est = {'D1': 1, 'n1': 1e3}

    ln_prior, sample_from_the_prior = get_ln_prior_func()
    # If not start point provided, sample a point from the prior
    if not start_point:
        start_point = start_point_est

    # Define two functions to minimize

    def minimize_me(args):
        """-ln posterior to minimize"""
        args_dict = {a: args[i] for i, a in enumerate(names)}
        return -ln_lklh_func(**args_dict) - ln_prior(**args_dict)

    def minimize_me_log_params(ln_args):
        """
        Same as minimize me, but the args correspond to log of the parameters.
        This seem to accelerate convergence, when D and n have different scales.
        """
        exponentiated = exp(ln_args)
        # Check for infinite values after exponentiation
        if np.any(~np.isfinite(exponentiated)):
            return -np.inf

        args_dict = {a: exponentiated[i] for i, a in enumerate(names)}

        return -ln_lklh_func(**args_dict) - ln_prior(**args_dict)

    # %% Find MLE
    if verbose:
        print('Started MLE search')

    # print(method, type(method), method == 'Nelder-Mead', method is 'Nelder-Mead')
    # print(method, type(method), method == 'BFGS', method is 'BFGS')

    if method == 'BFGS':
        options = {'disp': verbose, 'gtol': tol}

    elif method == 'Nelder-Mead':
        options = {'disp': verbose, 'maxiter': d * 1000}
    else:
        raise RuntimeError(f'Method "{method}" not implemented')

    def retry(i, min, ln_ev):
        print('Found minimum: ', min.fun, 'with ln evidence', ln_ev, ' at ', to_dict(*min.x))
        # print(
        #     f'Retrying with another starting point. Finished try {i+1}/{tries}.\n')
        # print('Full output: ', min)
        # verbose_tries = True
        return

    def calculate_laplace_approximation(min):
        """
        The function locally calculates the integral of a given minimum by using a Laplace approximation (approximating the shape of the peak by a non-uniform Gaussian).
        If the Hessian was calculated by the BFGS method, its values is used, otherwise, it is evaluated through finite differences.
        """
        if method is 'BFGS' and np.linalg.det(min.hess_inv) >= 1e-8:
            # Require that the returned Hessian has really been evaluated
            hess_inv = min.hess_inv
            det_inv_hess = np.linalg.det(hess_inv)
        else:
            # Otherwise, manually estimate the Hessian
            hess = nd.Hessian(minimize_me)(min.x)
            det_inv_hess = 1 / np.linalg.det(hess)

        if det_inv_hess <= 0:
            # If the Hessian is non-positive, it was not a minimum
            return np.nan

        # Calcualte evidence
        # In the following, we subtract minimize_me because `fnc` is the negative log likelihood
        ln_model_evidence = ((d / 2) * log(2 * pi)
                             + 1 / 2 * log(det_inv_hess)
                             - fnc(min.x))
        return ln_model_evidence

    def calculate_evidence_integral(points=None):
        """
        Calculate the evidence integral by numerical integration without Laplace approximation.
        """
        if link:
            return np.nan

        lims_zero = 1e-8
        infty = 1e7
        all_integration_limits = {'D1': [lims_zero, infty], 'D2': [lims_zero, infty], 'n1': [
            lims_zero, infty], 'n2': [lims_zero, infty], 'n12': [0, infty]}

        # Get the scale of the maximum of the posterior to normalize the integrand
        largest_ln_value = max([-minimize_me(point) for point in points])

        # Filter break points be removing those that are too close, separately along each axis
        atol_points = 1e-1
        points = zip(*points)   # combine break points per axis
        new_points = []
        for points_1d in points:
            points_1d = np.sort(points_1d)
            new_points_1d = [points_1d[0]]
            for p in points_1d[1:]:
                if p - new_points_1d[-1] >= atol_points:
                    new_points_1d.append(p)
            new_points.append(new_points_1d)

        print('Break points for direct integration:', new_points)

        def integrand(*args):
            """Renormalize the function before integration"""
            return exp(-minimize_me(args) - largest_ln_value)

        opts = [{'points': el, 'epsabs': 1e-1, 'epsrel': 1e-1} for el in new_points]
        integration_limits = [all_integration_limits[name] for name in names]
        res = nquad(integrand, ranges=integration_limits, opts=opts)

        return log(res[0]) + largest_ln_value

    verbose_tries = True
    mins = []
    can_repeat = True
    for i in range(tries):
        print(f'\nMLE search. Try {i+1}/{tries}...')

        # On the first try, load the MLE guess from file. Else sample from the prior
        if i == 0:
            start_point, old_ln_value, success_load = load_MLE_guess(
                hash_no_trial=hash_no_trial, link=link)
            if not success_load:
                start_point = sample_from_the_prior(link)
                print(
                    f'Sampling an origin point from the prior:\n', start_point)
            else:
                print('Starting MLE guess loaded successfully:\n', (start_point, old_ln_value))

        elif i == tries - 1:
            # On the last try, retry from the best guess so far
            mins.sort(key=itemgetter(0))
            start_point = to_dict(*mins[0][2].x)
            print('On the last try, retry from the best guess so far:\n', start_point)

            # Also estimate how many of the best values are close
            # frac is the fraction of tries that the best minimum was found if there were no retrials from the best
            atol = 0.1
            fun_vals = [min[0] for min in mins]
            diffs = np.array([np.abs(fun_vals[i] - fun_vals[0]) for i in range(1, len(fun_vals))])
            # add 1 to count the value itself
            times_best_found = np.sum(diffs < atol) + 1
            real_tries = tries - 1  # subtract 1 because repeating from the best point
            # Store the value in a file. Remember there is always at least 1 because we recalculate the best point
            frac = times_best_found / real_tries
            save_number_of_close_values(link, times_best_found, real_tries, frac)

        else:  # sample from the prior
            start_point = sample_from_the_prior(link)
            print(
                f'Sampling an origin point from the prior:\n', start_point)

        # if verbose_tries:
        #     print(f'Starting point: {start_point}')
        if bl_log_parameter_search:
            start_point_vals = log(to_list(start_point))
            fnc = minimize_me_log_params
        else:
            start_point_vals = to_list(start_point)
            fnc = minimize_me

        min = minimize(fnc, start_point_vals, tol=tol, method=method,
                       options=options)

        # Store the found point if the hessian is not diagonal (because it's not a new point then)
        # if np.abs(np.linalg.det(min.hess_inv)) >= 1e-8:
        ln_evidence = calculate_laplace_approximation(min)
        element = (min.fun, ln_evidence, min)
        mins.append(element)
        retry(i, min, ln_evidence)

    # Sort the results
    mins.sort(key=itemgetter(0))

    # Calculate evidence integral
    points = [el[2].x for el in mins]
    ln_true_evidence = calculate_evidence_integral(points)
    print('Direct numerical evaluation of the evidence integral: ', ln_true_evidence)

    print(f'\nFound the following (minimi, ln_evidence) in {tries} tries:\n', [
          (min[0], min[1]) for min in mins])

    # Choose the best valid MLE (the det hess should be > 0 for it to be a minimum)
    success = False
    for _, _, min in mins:
        if min.success or (not min.success and min.status == 2):
            # Estimate the Hessian in the MLE
            ln_model_evidence = calculate_laplace_approximation(min)

            # if ln_model_evidence <= 0:
            #     # If this was not a minimum
            #     continue

            if not np.isnan(ln_model_evidence):
                # If this was not a minimum, the returned value will be nan.
                success = True
                break

    # Restore the parameters from log
    if bl_log_parameter_search:
        MLE = {names[i]: exp(min.x[i]) for i in range(len(min.x))}
    else:
        MLE = {names[i]: min.x[i] for i in range(len(min.x))}

    if success:
        print(
            f'MLE search with link = {link} converged.\nTaking the best point seen so far:\n', (min.fun, MLE))
        # Save the MLE guess for further use
        save_MLE_guess(hash_no_trial=hash_no_trial, MLE_guess=MLE,
                       ln_posterior_value=-min.fun, link=link)
    else:
        print(
            f'MLE search procedure with link={link} failed to converge in {tries} tries.')

    if verbose:
        1
        # print('Full results:\n', min)
        # print('det_inv_hess: ', det_inv_hess)

    return MLE, ln_model_evidence, min, success
