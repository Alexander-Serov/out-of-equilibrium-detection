"""
Calculate likelihoods that will be used for model comparison.
Those are mainly likelihoods for periodogram components.
"""

import copy
import functools
import logging
from datetime import datetime

import numdifftools as nd
import numpy as np
import scipy
from filelock import FileLock
from matplotlib import pyplot as plt
from numpy import cos, exp, log, pi, sin
from numpy.linalg import eig, inv
from scipy import integrate
from scipy.fftpack import fft
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

# from plot import plot_periodogram
from support import hash_from_dictionary, load_data, save_data

ln_infty = - 20 * log(10)


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


def get_ln_prior_func():
    # (D1, n1, D2=None, n2=None, n12=None):
    """
    Prior for the corresponding likelihoods.

    Also provide a sampler to sample from the prior.
    """

    ln_prior = 0
    n0 = 10**3      # s^{-1}, spring constant prior scale

    def ln_D_prior(D):
        """D_i prior"""
        beta = 0.5  # um^2/s
        a = 0.5     # no units
        return a * log(beta) - gammaln(a) - (a + 1) * log(D) - beta / D

    def ln_n_prior(n):
        """
        n_i prior

        This prior is on log values of n allowing very wide distributions. I need to make them smaller, so that n1 = 1e-20 be not accessible..

        Currently it is a Gaussian distribution for log(n), which does not prohibit 0. I must change it.

        """
        sigma_n = 2 * log(10)

        return -log(n) - 1 / 2 * log(2 * pi * sigma_n**2) - (log(n) - log(n0) - sigma_n**2)**2 / 2 / sigma_n**2

    def ln_n_link_prior(n):
        """n_{12} link prior"""
        a = 1

        return log(a) - log(n0) - (a + 1) * log(1 + n / n0)

    # Assemble the prior
    if np.any(np.array([D1, n1]) <= 0):
        return ln_infty
    # print(1, D1, n1)
    ln_prior += ln_D_prior(D1)
    ln_prior += ln_n_prior(n1)

    if all(v is not None for v in (D2, n2, n12)):
        if np.any(np.array((D2, n2)) <= 0) or n12 < 0:
            return ln_infty
        # assert np.all([D2, n2, n12] >= 0)
        # print(2, D2, n2, n12)
        ln_prior += ln_D_prior(D2)
        ln_prior += ln_n_prior(n2)
        ln_prior += ln_n_link_prior(n12)

    return ln_prior


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
            print('sigma2s: ', sigma2s)
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


def get_MLE(ks, M, dt, link, zs_x=None, zs_y=None, start_point=None, verbose=False, method='BFGS'):
    """
    Locate the MLE of the posterior. Estimate the evidence integral through Laplace approximation. Since the parameters have different scales, the search is conducted in the log space of the parameters.

    Make a guess of the starting point if none is supplied.

    Input:
        link, bool: whether a link between 2 particles should be considered in the likelihood
        method: 'BFGS' or 'Nelder-Mead'
    """

    tol = 1e-5  # search tolerance
    grad_tol = tol * 10  # gradient check tolerance
    ln_model_evidence = 0
    np.random.seed()

    # Choose the appropriate likelihood
    if link:
        ln_lklh_func = get_ln_likelihood_func_2_particles_x_link(
            ks=ks, zs_x=zs_x, zs_y=zs_y, M=M, dt=dt)
        names = ('D1', 'D2', 'n1', 'n2', 'n12')
        start_point_est = {'D1': 1, 'n1': 1e3, 'D2': 1, 'n2': 1e3, 'n12': 1e3}
    else:
        ln_lklh_func = get_ln_likelihood_func_no_link(
            ks=ks, zs_x=zs_x, zs_y=zs_y, M=M, dt=dt)
        names = ('D1', 'n1')
        start_point_est = {'D1': 1, 'n1': 1e3}
    d = len(names)

    # If not start point provided, use the default one
    if not start_point:
        start_point = start_point_est
    start_point_vals = log(list(start_point.values()))

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

    if method is 'BFGS':
        options = {'disp': verbose, 'gtol': tol}

    elif method is 'Nelder-Mead':
        options = {'disp': verbose, 'maxiter': d * 1000}
    else:
        options = {}

    # print('minmz', start_point_vals, tol, method, options)
    max = minimize(minimize_me_log_params, start_point_vals, tol=tol, method=method,
                   options=options)
    # Restore the parameters from log
    MLE = {names[i]: exp(max.x[i]) for i in range(len(max.x))}

    # Check convergence state
    if not max.success:
        logging.warning(f'MLE search procedure with link = {link} failed to converge')
    # # Log the parameter values to a file
    # convergence_errors_file = 'convergence errors.dat'
    # lock_file = convergence_errors_file + '.lock'
    #
    # lock = FileLock(lock_file, timeout=1)
    # with lock:
    #
    #     # str = f'Time: {datetime.now}, parameters: D1={D1}, D2={D2}, n1={n1}, n2={n2}, n12={n12}, M={M}, dt={dt}.\n'
    #     str = f'Time: {datetime.now}, parameters: {true_parameters}.\n'
    #
    #     open(convergence_errors_file, 'a').write(str)

    # # Double check if a local optimum was found
    # grad = nd.Gradient(minimize_me)(exp(max.x))
    # if not np.all(abs(grad) <= grad_tol):
    #     logging.warn(
    #         'The output gradient after optimization is too high. The algorithm might not have converged! Gradient: ' + repr(grad))

    # Estimate the Hessian in the MLE
    hess = nd.Hessian(minimize_me)(exp(max.x))
    det_inv_hess = 1 / np.linalg.det(hess)
    if det_inv_hess < 0:
        logging.warning(
            f'The Hessian deteriminant is negative. Cannot calculate the evidence with link = {link}.')

    # Print some output if verbose
    if verbose:
        print('Check gradient at the minimum: ', nd.Gradient(minimize_me)(max.x))
        print('MLE found: ', exp(max.x))
        print('Prior ln value at MLE: ', ln_prior(**MLE), '\n')
        print('Full results:\n', max)
        print('det_inv_hess: ', det_inv_hess)

    # %% Estimate evidence as the integral across the parameters
    # In the following, we subtract minimize_me because it is the negative log likelihood
    ln_model_evidence = ((d / 2) * log(2 * pi)
                         + 1 / 2 * log(det_inv_hess)
                         - minimize_me_log_params(max.x))

    return MLE, ln_model_evidence, max
