"""
Calculate likelihoods that will be used for model comparison.
"""

import copy
import functools
import logging

import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy import cos, exp, log, pi, sin
from numpy.linalg import eig, inv
from scipy import integrate
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

# from scipy.stats import chi2

ln_infty = - 20 * log(10)


class J_class:
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
def get_sigma2_matrix_func(D1=1, D2=3, n1=1, n2=1, n12=1, M=999, dt=0.3, alpha=0):
    # atol = 1e-8

    # Check values
    if np.any(np.array([D1, D2, n1, n2, n12]) < 0):
        return np.nan

    # if k == 0:
    #     logging.error(
    #         'The current implementation does not calculate the likelihood for k=0. Skipping')
    #     return None

    A = np.array([[-n1 - n12, 0, n12, 0],
                  [0, -n1, 0, 0],
                  [n12, 0, -n2 - n12, 0],
                  [0, 0, 0, -n2]])

    b = np.diag(np.sqrt([2 * D1, 2 * D1, 2 * D2, 2 * D2]))

    lambdas, U = np.linalg.eig(A)
    # print(lambdas)
    Um1 = inv(U)

    # print('A', A)
    # print('U check', U @ np.diag(lambdas) @ Um1 - A)
    # print('l1', lambdas)
    # print('U', U)
    # print('v', U @ A @ Um1)

    # # Analytical results from Mathematica
    # g = np.sqrt((n1 - n2)**2 + 4 * n12**2)
    # lambdas = np.array([-n1,
    #                     -n2,
    #                     (-g - n1 - 2 * n12 - n2) / 2,
    #                     (g - n1 - 2 * n12 - n2) / 2])

    @functools.lru_cache(maxsize=None)
    def J(i, j, k):
        """Return the Fourier image of the average correlation between the stochastic integrals"""
        i -= 1
        j -= 1
        if i == j:
            lamb = lambdas[i]
            if lamb == 0:
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


def ln_prior(D1, n1, D2=None, n2=None, n12=None):
    """
    Prior for the corresponding likelihoods
    """

    ln_prior = 0
    n0 = 10**3      # s^{-1}, spring constant prior scale

    def ln_D_prior(D):
        """D_i prior"""
        beta = 0.5  # um^2/s
        a = 0.5     # no units
        return a * log(beta) - gammaln(a) - (a + 1) * log(D) - beta / D

    def ln_n_prior(n):
        """n_i prior"""
        sigma_n = 3 * log(10)

        return -log(n) - 1 / 2 * log(2 * pi * sigma_n**2) - (log(n) - log(n0))**2 / 2 / sigma_n**2

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
        """Calculate the pdf for a sum of chi-squared of order 2 through residues of the characteristic function

        Formula:
        f(z) = 1/2 * \sum_j \e^{-z/2 \sigma^2_j} / (\prod_{n \neq j} (sigma2^_j - \sigma^2_n))
        """
        atol = 1e-18

        # print('s2: ', sigma2s)
        sigma2s_nonzero = np.array([sigma2 for sigma2 in sigma2s if abs(sigma2) > atol])

        # Check if we have same sigmas
        if len(sigma2s_nonzero) > 1 and np.any(np.abs(np.diff(np.sort(sigma2s_nonzero))) < 1e-8):
            str = "Encountered same sigmas. The current likelihood calculation procedure may fail. Sigma2 values: " + \
                repr(sigma2s_nonzero)
            logging.warning(str)
            # print(str)

        if len(sigma2s_nonzero) == 0:
            raise RuntimeError('Sigma2 filtering removed all values. Input sigmas: ' +
                               np.array_str(sigma2s)
                               + f'.\n Other parameters: k={k}, D1={D1}, D2={D2}, n1={n1}, n2={n2}, n12={n12}')
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
        print(sigma2s, D1, D2, n1, n2, n12, M, dt, alpha)
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
    # Returns a log_10 of the likelihood function for all input data points as a function of parameters (D1, D2, n1, n2, n12).
    # The likelihood is not normalized over these parameters.
    #
    # Use alpha to specifythe coordinate for the likelihood:
    #     alpha = 0 --- x1
    #     alpha = 1 --- y1
    #     alpha = [0,1] --- both x1 and y1
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


def get_MLE(ks, M, dt, link, zs_x=None, zs_y=None, start_point=None):
    """
    Input:
        link, bool: whether a link between 2 particles should be considered in the likelihood
    """

    atol = 1e-8

    # % Make a guess of the starting point
    est = {}
    r1 = np.concatenate([zs_x, zs_y]) / dt**2 / 2
    est['D1'] = np.quantile(r1, 0.95)
    est['D2'] = est['D1']
    # print(r1, est)

    if link:
        ln_lklh_func = get_ln_likelihood_func_2_particles_x_link(
            ks=ks, zs_x=zs_x, zs_y=zs_y, M=M, dt=dt)
        names = ('D1', 'D2', 'n1', 'n2', 'n12')
        start_point_est = {'D1': 1, 'n1': 1, 'D2': 1, 'n2': 1, 'n12': 1}

        # def ln_prior_func(D1, D2, n1, n2, n12):
        #     return ln_prior(D1=D1, D2=D2, n1=n1, n2=n2, n12=n12)

        # other_args = {}
    else:
        ln_lklh_func = get_ln_likelihood_func_no_link(ks=ks, zs_x=zs_x, zs_y=zs_y, M=M, dt=dt)
        names = ('D1', 'n1')
        start_point_est = {'D1': 1, 'n1': 1}

        # def ln_prior_func(kwargs):
        #     return ln_prior(D1=D1, n1=n1)
        # other_args = {}
    d = len(names)

    if not start_point:
        start_point = start_point_est

    # bnds = ((atol, None),) * 5

    # elif alpha == 1:
    #     # The y coordinate is only defined by D1 and n1, since it is independent.
    #     start_point = {'D1': 1, 'n1': 1}
    #     other_args = {a: 1 for a in ('D2', 'n2', 'n12')}
    #     bnds = ((atol, None),) * 2
    #     names = list(start_point.keys())

    start_point_vals = list(start_point.values())
    # print(names)

    def minimize_me(args):
        # print(args[0], names[0])
        args_dict = {a: args[i] for i, a in enumerate(names)}
        return -ln_lklh_func(**args_dict) - ln_prior(**args_dict)
        # return - ln_prior(**args_dict)

    print('Started MLE searh')
    #  bounds=bnds,
    # print(len(start_point), len(bnds))
    # method='BFGS'

    if 1:
        method = 'BFGS'
        options = {'disp': True}
        tol = 1e-5
    else:
        method = 'Nelder-Mead'
        options = {'disp': True, 'maxiter': d * 1000}
        tol = 1e-5

    max = minimize(minimize_me, start_point_vals, tol=tol, method=method,
                   options=options)
    print('MLE found: ', max.x, '\n')
    print(max)
    MLE = {names[i]: max.x[i] for i in range(len(max.x))}

    # Estimate ln evidence through direct integration
    # ln_model_evidence_direct = estimate_evidence_integral(ln_lklh_func, MLE=list(MLE.values()))
    ln_model_evidence_direct = None

    if method is 'Nelder-Mead':

        return MLE, None, max, ln_model_evidence_direct

    elif method is 'BFGS':
        # Estimate the Bayes factor assuming a normal posterior distribution
        # TODO: add prior
        inv_hess = max.hess_inv
        ln_model_evidence = ((d / 2) * log(2 * pi)
                             + 1 / 2 * np.log(np.linalg.det(inv_hess))
                             + ln_lklh_func(**MLE))

        return MLE, ln_model_evidence, max, ln_model_evidence_direct


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


# % == Tests
if __name__ == '__main__':
    k = 50
    D1 = 1
    dt = 0.3
    z = D1 * dt**2
    a = likelihood_2_particles_x_link_one_point(z, k=k, D1=D1, dt=dt)
    # print(a)

    ks = [400, 401]
    zs = np.array([0.5, 2]) * dt**2
    M = 999
    alpha = 0
    lklh_func = get_log_likelihood_func_2_particles_x_link(
        ks=ks, zs=zs, M=M, dt=dt, alpha=alpha)
    # print('f', lklh_func(1, 1, 1e-8, 1e-8, 1e-8, 1e-8))

    # start_point = (1, 1, 1e-8, 1e-8, 1e-8, 1e-8)
    #
    # def minimize_me(args):
    #     # print('d', *args, type(args))
    #     return -lklh_func(*args)
    #
    # # print('c', minimize_me(*start_point))
    # max = minimize(minimize_me, start_point)
    # print(max.x)
    print(get_MLE(ks=ks, zs=zs, M=M, dt=dt, alpha=alpha))
# -lklh_func(*[1.e+00, 1.e+00, 1.e-08, 1.e-08, 1.e-08, 1.e-08])
