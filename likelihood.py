"""
Calculate likelihoods that will be used for model comparison.
"""

import logging

import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy import cos, exp, pi, sin
from scipy.optimize import minimize

# from scipy.stats import chi2


def likelihood_2_particles_x_link_one_point(z, k=1, D1=1, D2=3, k1=1e-8, k2=1e-8, k12=1e-8, gamma=1e-8, M=999, dt=0.3, alpha=0):
    """
    Calculate likelihood of one power spectrum observation z for the frequency k.
    Does not work for k = 0.

    Input:
    gamma --- viscosity, kg/s,
    z --- the value of the power spectrum component,

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
    atol = 1e-8
    # print(k1)

    # Dummy constants
    # k1 = 1e-8
    # k2 = 1e-8
    # k12 = 1e-8
    # D1 = 1
    # D2 = 3
    # gamma = 1e-8
    # M = 999
    # dt = 0.3
    # alpha = 0

    if k == 0:
        logging.error(
            'The current implementation does not calculate the likelihood for k=0. Skipping')
        return None

    # Analytical results from Mathematica
    g = np.sqrt((k1 - k2)**2 + 4 * k12**2)
    # print('11')
    lambdas = np.array([-(k1 / gamma),
                        -(k2 / gamma),
                        (-g - k1 - 2 * k12 - k2) / (2 * gamma),
                        (g - k1 - 2 * k12 - k2) / (2 * gamma)])

    def J(i, j, k):
        """Return the Fourier image of the average correlation between the stochastic integrals"""
        i -= 1
        j -= 1
        if i == j:
            lamb = lambdas[i]
            J = -(((M * dt ** 2) * ((1 - exp(lamb * dt) ** 2) * (1 - cos((2 * pi * k) / M)))) / (
                (2 * lamb) * (1 + exp(lamb * dt) ** 2 - 2 * exp(lamb * dt) * cos((2 * pi * k) / M))))

        else:
            lamb1, lamb2 = lambdas[[i, j]]
            J = -(((M * dt ** 2) * ((1 - exp(lamb1 * dt) * exp(lamb2 * dt)) * (1 + exp(lamb1 * dt) * exp(lamb2 * dt) - (exp(lamb1 * dt) + exp(lamb2 * dt)) * cos((2 * pi * k) / M)) * (1 - cos((2 * pi * k) / M)))) /
                  ((lamb1 + lamb2) * ((1 + exp(lamb1 * dt) ** 2 - 2 * exp(lamb1 * dt) * cos((2 * pi * k) / M)) * (1 + exp(lamb2 * dt) ** 2 - 2 * exp(lamb2 * dt) * cos((2 * pi * k) / M)))))
        return J

    def sigma2s(alpha, k):
        """Return sigma2 values that correspond to the component alpha and frequency k.
        The result is divided by (M*dt) to directly give the power spectrum
        """
        mat = np.array([
            [(D1 / 2 + (D1 * k1) / g + (D1 * k1 ** 2) / (2 * g ** 2) - (D1 * k2) / g - (D1 * k1 * k2) / g ** 2 + (D1 * k2 ** 2) / (2 * g ** 2)) * J(3, 3, k) +
             (D1 - (D1 * k1 ** 2) / g ** 2 + (2 * D1 * k1 * k2) / g ** 2 - (D1 * k2 ** 2) / g ** 2) * J(3, 4, k) +
             (D1 / 2 - (D1 * k1) / g + (D1 * k1 ** 2) / (2 * g ** 2) + (D1 * k2) /
                g - (D1 * k1 * k2) / g ** 2 + (D1 * k2 ** 2) / (2 * g ** 2)) * J(4, 4, k),
             0,
             ((D1 * g ** 2) / (8 * k12 ** 2) - (D1 * k1 ** 2) / (4 * k12 ** 2) + (D1 * k1 ** 4) / (8 * g ** 2 * k12 ** 2) + (D1 * k1 * k2) / (2 * k12 ** 2) -
              (D1 * k1 ** 3 * k2) / (2 * g ** 2 * k12 ** 2) - (D1 * k2 ** 2) / (4 * k12 ** 2) + (3 * D1 * k1 ** 2 * k2 ** 2) / (4 * g ** 2 * k12 ** 2) -
              (D1 * k1 * k2 ** 3) / (2 * g ** 2 * k12 ** 2) + (D1 * k2 ** 4) / (8 * g ** 2 * k12 ** 2)) * J(3, 3, k) +
             (-(D1 * g ** 2) / (4 * k12 ** 2) + (D1 * k1 ** 2) / (2 * k12 ** 2) - (D1 * k1 ** 4) / (4 * g ** 2 * k12 ** 2) - (D1 * k1 * k2) / k12 ** 2 +
              (D1 * k1 ** 3 * k2) / (g ** 2 * k12 ** 2) + (D1 * k2 ** 2) / (2 * k12 ** 2) - (3 * D1 * k1 ** 2 * k2 ** 2) / (2 * g ** 2 * k12 ** 2) +
              (D1 * k1 * k2 ** 3) / (g ** 2 * k12 ** 2) - (D1 * k2 ** 4) / (4 * g ** 2 * k12 ** 2)) * J(3, 4, k) +
             ((D1 * g ** 2) / (8 * k12 ** 2) - (D1 * k1 ** 2) / (4 * k12 ** 2) + (D1 * k1 ** 4) / (8 * g ** 2 * k12 ** 2) + (D1 * k1 * k2) / (2 * k12 ** 2) -
              (D1 * k1 ** 3 * k2) / (2 * g ** 2 * k12 ** 2) - (D1 * k2 ** 2) / (4 * k12 ** 2) + (3 * D1 * k1 ** 2 * k2 ** 2) / (4 * g ** 2 * k12 ** 2) -
                  (D1 * k1 * k2 ** 3) / (2 * g ** 2 * k12 ** 2) + (D1 * k2 ** 4) / (8 * g ** 2 * k12 ** 2)) * J(4, 4, k),
             0],
            [0, 2 * D1 * J(1, 1, k), 0, 0],
            [(2 * D2 * k12 ** 2 * J(3, 3, k)) / g ** 2 - (4 * D2 * k12 ** 2 * J(3, 4, k)) / g ** 2 + (2 * D2 * k12 ** 2 * J(4, 4, k)) / g ** 2,
             0,
             (D2 / 2 - (D2 * k1) / g + (D2 * k1 ** 2) / (2 * g ** 2) + (D2 * k2) / g - (D2 * k1 * k2) / g ** 2 + (D2 * k2 ** 2) / (2 * g ** 2)) * J(3, 3, k) +
             (D2 - (D2 * k1 ** 2) / g ** 2 + (2 * D2 * k1 * k2) / g ** 2 - (D2 * k2 ** 2) / g ** 2) * J(3, 4, k) +
             (D2 / 2 + (D2 * k1) / g + (D2 * k1 ** 2) / (2 * g ** 2) - (D2 * k2) /
              g - (D2 * k1 * k2) / g ** 2 + (D2 * k2 ** 2) / (2 * g ** 2)) * J(4, 4, k),
             0],
            [0, 0, 0, 2 * D2 * J(2, 2, k)]
        ])
        return mat[alpha, :] / (M * dt)

    # Defined the pdf function for a sum of chi-squared
    def pdf_chi_squared_sum(z, sigma2s):
        """Calculate the pdf for a sum of chi-squared of order 2 through residues of the characteristic function"""

        # print('s2: ', sigma2s)
        sigma2s_nonzero = np.array([sigma2 for sigma2 in sigma2s if abs(sigma2) > atol])

        n = len(sigma2s_nonzero)
        full_set = set(range(n))

        terms = np.full(n, np.nan, dtype=np.float)

        for j in range(n):
            other_inds = list(full_set - set([j]))
            terms[j] = exp(-z / 2 / sigma2s_nonzero[j])
            terms[j] /= (sigma2s_nonzero[j] - sigma2s_nonzero[other_inds]).prod()
            # print('b ', exp(-z / 2 / sigma2s_nonzero[j]),
            #       (sigma2s_nonzero[j] - sigma2s_nonzero[other_inds]).prod())

        val = terms.sum() / 2

        return val

    # Calculate the likelihood of the spectrum
    # sigma2s = sigma2_matrix[alpha, [0, 2]]

    prob = pdf_chi_squared_sum(z=z, sigma2s=sigma2s(alpha, k))
    # print('prob: ', prob)
    # print('lambs: ', lambdas)
    # prob
    return prob


def get_log_likelihood_func_2_particles_x_link(ks, zs, M, dt, alpha):
    """
    Returns a log_10 of the likelihood function for all input data points as a function of parameters (D1, D2, k1, k2, k12, gamma).
    The likelihood is not normalized over these parameters.
    """

    def lg_lklh(D1, D2, k1, k2, k12, gamma):
        lg_lklh_vals = np.log10([likelihood_2_particles_x_link_one_point(z, k=k, D1=D1,
                                                                         D2=D2, k1=k1, k2=k2, k12=k12, gamma=gamma, M=M, dt=dt, alpha=alpha)
                                 for z, k in zip(zs, ks)])
        # print('e', np.isfinite(lklh_vals))
        lg_lklh_vals = lg_lklh_vals[np.isfinite(lg_lklh_vals)]
        lg_lklh_val = lg_lklh_vals.sum()
        # print(lg_lklh_val)
        #
        # lg_lklh_val = 0
        # for k, z in zip(ks, zs):
        #     lg_lklh_val += np.log10(likelihood_2_particles_x_link_one_point(z, k=k, D1=D1,
        #                                                                     D2=D2, k1=k1, k2=k2, k12=k12, gamma=gamma, M=M, dt=dt, alpha=alpha))
        # print(lg_lklh_val)
        return lg_lklh_val

    return lg_lklh


def get_MLE(ks, zs, M, dt, alpha):
    lg_lklh_func = get_log_likelihood_func_2_particles_x_link(ks, zs, M, dt, alpha)
    start_point = (1, 1, 1e-8, 1e-8, 1e-8, 1e-8)

    def minimize_me(args):
        return -lg_lklh_func(*args)

    logging.info('Started MLE searh')
    max = minimize(minimize_me, start_point, options={'disp': True})
    logging.info('Completed MLE searh')

    names = ('D1', 'D2', 'k1', 'k2', 'k12', 'gamma')
    MLE = {name: max.x[i] for i, name in enumerate(names)}

    return MLE


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
    lklh_func = get_log_likelihood_func_2_particles_x_link(ks=ks, zs=zs, M=M, dt=dt, alpha=alpha)
    # print('f', lklh_func(1, 1, 1e-8, 1e-8, 1e-8, 1e-8))

    start_point = (1, 1, 1e-8, 1e-8, 1e-8, 1e-8)

    def minimize_me(args):
        # print('d', *args, type(args))
        return -lklh_func(*args)

    # print('c', minimize_me(*start_point))
    max = minimize(minimize_me, start_point)
    print(max.x)
# -lklh_func(*[1.e+00, 1.e+00, 1.e-08, 1.e-08, 1.e-08, 1.e-08])
