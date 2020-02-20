"""
Simulate random walk of two particles connected by a spring subject to two different diffusivities D1 and D2
"""

# if __name__ == '__main__':
#     try:
#         bl_has_run
#
#     except Exception:
#         # %matplotlib
#         %load_ext autoreload
#         %autoreload 2
#         bl_has_run = True

import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import cos, exp, sin
from scipy.optimize import root
from tqdm import tqdm, trange

from stopwatch import stopwatch
from support import hash_from_dictionary, load_data, save_data
import sys


def simulate_2_confined_particles_with_fixed_angle_bond(parameters, plot=False,
                                                        recalculate=False, save_figure=False,
                                                        file=r'.\trajectory.dat', seed=None,
                                                        verbose=False, show=True):
    """
    Simulate the trajectories of two particles connected by 1 spring and only along the x axis.
    Units of measurements:
    D1, D2 --- um^2/s,
    gamma --- kg/s,
    k12 --- kg/s^2,
    T --- s,
    angle --- angle of bond beetween the particles in the lab system measured counterclockwise (in radians).

    Input:
    true_parameters --- a dictionary that must contain D1, D2, k12, T, dt, gamma, angle, L
    """

    # Load parameters
    # hash_sequence =
    D1, D2, n1, n2, n12, M, dt, angle, L = [parameters[key]
                                            for key in 'D1 D2 n1 n2 n12 M dt angle L0'.split()]
    N = M

    # % Constants
    # kB = 1.38e-11  # kg*um^2/s^2/K
    atol = 1e-16
    rtol = 1e-6
    R0 = [0, 0, L, 0]
    bl_loaded = False
    # max_terms = 100
    # min_N = int(1e4)

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    # n1, n2, n12 = np.array([k1, k2, k12]) / gamma

    # Hash the parameters to be able to reload
    hash, _ = hash_from_dictionary(dim=2, parameters=parameters)

    # Reload if requested
    if not recalculate:
        dict_data, loaded = load_data(hash)
        # print('sim', dict_data, loaded)
        # return
        if loaded and np.all([key in dict_data.keys() for key in 't R dR'.split()]):
            t, R, dR = [dict_data[key] for key in 't R dR'.split()]

            # Plot
            if plot:
                plot_trajectories(t=t, R=R, dR=dR, true_parameters=parameters, save=save_figure)

            # print(f'Trajectories reloaded. Hash: {hash}')
            return t, R, dR, hash

    A = np.array([[-n1 - n12, 0, n12, 0],
                  [0, -n1, 0, 0],
                  [n12, 0, -n2 - n12, 0],
                  [0, 0, 0, -n2]], dtype=np.float64)
    a = L * np.array([[-n12, 0, n2 + n12, 0]]).transpose()
    b = np.diag(np.sqrt([2 * D1, 2 * D1, 2 * D2, 2 * D2]))
    # print(b)

    lambdas, U = np.linalg.eig(A)
    Um1 = np.linalg.inv(U)
    diag = np.diag(lambdas)
    # Am1 = np.linalg.inv(A)
    # print('lambs = ', lambdas)
    # print('Um1 @ A @ U = ', Um1 @ A @ U)

    # Choose dt and N (number of jumps)
    # dt = 1e-2
    # N = np.ceil(T / dt).astype(int)

    T = dt * N
    t = np.arange(N + 1) * dt

    # R0 = np.transpose([np.hstack([r10, r20])])
    R = np.zeros((4, N + 1)) * np.nan
    R[:, 0] = R0

    # Q0 = R0 + Am1 @ a
    # print('R0', R0)
    # print('R init', R)

    def check_zero(a):
        """Check if all the values of the input array are 0"""
        sum = np.sum(np.array(a)[:])
        if np.isclose(sum, 0, atol=atol, rtol=rtol):
            return True
        else:
            return False

    # One-step covariance matrix for the diagonal elements (stochastic integrals)
    # If lambdas are the same, must return the same values for them
    mean_1_step = [0] * 4
    cov_1_step = np.full_like(A, np.nan, dtype=np.float32)
    # print(cov_1_step, cov_1_step.dtype)
    atol_local = 1e-6
    # print('dt = ', dt)
    for i, li in enumerate(lambdas):
        for j, lj in enumerate(lambdas):
            exponent = dt * (li + lj)

            # For the exponent smaller than atol=1e-6, use a series
            if abs(exponent) < atol_local:
                cov_1_step[i, j] = dt + exponent * dt / 2
            else:
                # print('test', (np.exp(dt * (li + lj)) - 1) / (li + lj))
                cov_1_step[i, j] = (np.exp(dt * (li + lj)) - 1) / (li + lj)
            # print('exponent', i, j, li, lj, dt, exponent, cov_1_step[i, j])

    # Sample the diagonal elements from the distribution:
    # Generation dimensions: 4 noise sources x N x 4 elements
    # print('cov_1_step', cov_1_step)
    diag_noise_integrals = np.random.multivariate_normal(mean_1_step, cov_1_step, size=(4, N))

    # Calculate the return force integrals that do not change from step to step
    diag_return_force_integrals = np.zeros(4) * np.nan
    for i, l in enumerate(lambdas):
        if check_zero([l]):
            diag_return_force_integrals[i] = dt
        else:
            diag_return_force_integrals[i] = (np.exp(l * dt) - 1) / l

    # Iterate over steps
    for i in range(N):
        R_next = (
                U @ np.diag(np.exp(lambdas * dt)) @ Um1 @ R[:, i, None]
                + U @ np.diag(diag_return_force_integrals) @ Um1 @ a
                + U @ np.diag(diag_noise_integrals[0, i, :]) @ Um1 @ b[:, 0, None]
                + U @ np.diag(diag_noise_integrals[1, i, :]) @ Um1 @ b[:, 1, None]
                + U @ np.diag(diag_noise_integrals[2, i, :]) @ Um1 @ b[:, 2, None]
                + U @ np.diag(diag_noise_integrals[3, i, :]) @ Um1 @ b[:, 3, None]
        )
        R[:, i + 1] = R_next[:, 0]

    # print('R not rotated', R)
    # Rotate the result if necessary. Q is the rotation matrix
    # phi = angle / 180 * np.pi
    phi = angle
    # print('phi', phi)
    # raise RuntimeError('stop')
    Q = np.array([[cos(phi), - sin(phi), 0, 0],
                  [sin(phi), cos(phi), 0, 0],
                  [0, 0, cos(phi), -sin(phi)],
                  [0, 0, sin(phi), cos(phi)]
                  ])
    # print(Q)
    # print(R[:, 0])

    R_rotated = R.copy()
    if angle:
        for i in range(N + 1):
            # print(Q @ R[:, i, None])
            R_rotated[:, i, None] = Q @ R[:, i, None]

    R = R_rotated
    # print(R[:, 0])
    #  Displacements
    dR = R[:, 1:] - R[:, :-1]

    # Save the trajectories and simulation parameters
    # print(t, R, dR)
    dict_data = {'t': t, 'R': R, 'dR': dR, **parameters}
    # print('full dict', dict_data)
    save_data(dict_data=dict_data, hash=hash)

    if plot:
        plot_trajectories(t, R, dR, parameters, save=save_figure, show=show)
    #
    # return (t, X, dX, Y, dY)
    return t, R, dR, hash


#

# @profile
def simulate_a_free_hookean_dumbbell(parameters, plot=False, recalculate=False, save_figure=False,
                                     file=r'.\trajectory.dat', seed=None, verbose=False,
                                     show=True):
    """
    Simulate the trajectories of two particles connected by 1 spring and only along the x axis.
    Units of measurements:
    D1, D2 --- um^2/s,
    gamma --- kg/s,
    k12 --- kg/s^2,
    T --- s,
    angle --- angle of bond beetween the particles in the lab system measured counterclockwise (in radians).

    Input:
    true_parameters --- a dictionary that must contain D1, D2, k12, T, dt, gamma, angle, L
    """

    # Load parameters
    # hash_sequence =
    D1, D2, n1, n2, n12, M, dt, L0 = [parameters[key]
                                      for key in 'D1 D2 n1 n2 n12 M dt L0'.split()]
    N = M

    # % Constants
    # kB = 1.38e-11  # kg*um^2/s^2/K
    atol = 1e-16
    rtol = 1e-6
    alpha = 1  # the degree of implicitness
    bl_loaded = False
    min_dt_factor = 100  # Make sure dt used in simulation is smaller than system time scale by
    # at least this factor
    # max_terms = 100
    # min_N = int(1e4)

    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()

    phi0 = np.random.uniform(0, 2 * np.pi)
    R0 = np.array([-L0 * np.cos(phi0), -L0 * np.sin(phi0),
                   L0 * np.cos(phi0), L0 * np.sin(phi0)]) / 2

    # Calculate on a smaller mesh if necessary
    max_eta = 1e-2
    min_l0 = 20
    N_intermediate_points = 1

    l0 = L0 / np.sqrt(4 * np.max([D1, D2]) * dt)
    if l0 < min_l0:
        N_intermediate_points = int(np.ceil(min_l0 ** 2 / l0 ** 2))

    if n12 > 0:
        eta = n12 * dt
        if eta > max_eta:
            N_intermediate_points = np.max([N_intermediate_points, int(np.ceil(eta / max_eta))])
    # print(l0, eta)

    # max_dt = np.min(time_scales) / min_dt_factor
    # print('Time scales / dt: ', time_scales / dt)
    # print('dt: {0:.2g}'.format(dt))
    true_dt = dt
    if N_intermediate_points > 1:
        # N_intermediate_points = int(np.ceil(dt / max_dt))
        dt = dt / N_intermediate_points
        # bl_rescaled = True
        # N *= N_intermediate_points
        if verbose:
            print(
                f'For the accuracy of the simulations, time step reduced by the factor of '
                f'{N_intermediate_points} from {true_dt:.2g} to {dt:.2g}')
    else:
        N_intermediate_points = 1

    if verbose:
        print(f'l0 = {l0:g}, eta = {eta:g}')

    # n1, n2, n12 = np.array([k1, k2, k12]) / gamma

    # # Hash the parameters to be able to reload
    hash = None
    # hash, _ = hash_from_dictionary(dim=2, true_parameters=true_parameters)
    #
    # # Reload if requested
    # if not recalculate:
    #     dict_data, loaded = load_data(hash)
    #     # print('sim', dict_data, loaded)
    #     # return
    #     if loaded:
    #         t, R, dR = [dict_data[key] for key in 't R dR'.split()]
    #
    #         # Plot
    #         if plot:
    #             plot_trajectories(R, save=save_figure)
    #
    #         # print(f'Trajectories reloaded. Hash: {hash}')
    #         return t, R, dR, hash

    # A = np.array([[-n1 - n12, 0, n12, 0],
    #               [0, -n1, 0, 0],
    #               [n12, 0, -n2 - n12, 0],
    #               [0, 0, 0, -n2]], dtype=np.float64)
    # a = L0 * np.array([[-n12, 0, n2 + n12, 0]]).transpose()
    # b = np.diag(np.sqrt([2 * D1, 2 * D1, 2 * D2, 2 * D2]))
    # print(b)
    b = np.transpose(np.sqrt(2 * np.array([[D1, D1, D2, D2]])))

    # lambdas, U = np.linalg.eig(A)
    # Um1 = np.linalg.inv(U)
    # diag = np.diag(lambdas)
    # # Am1 = np.linalg.inv(A)
    # # print('lambs = ', lambdas)
    # # print('Um1 @ A @ U = ', Um1 @ A @ U)

    # Choose dt and N (number of jumps)
    # dt = 1e-2
    # N = np.ceil(T / dt).astype(int)

    # T = dt * N
    t = np.arange(N + 1) * true_dt

    # R0 = np.transpose([np.hstack([r10, r20])])
    R = np.zeros((4, N + 1)) * np.nan
    R[:, 0] = R0

    # print(f'R array size is {sys.getsizeof(R) / 2 ** 20} MB')

    # Q0 = R0 + Am1 @ a
    # print('R0', R0)
    # print('R init', R)

    def a(_R):
        out = np.empty((4, 1))
        R_diff = _R[2:] - _R[:2]  # R2 - R1
        out[:2, :] = R_diff
        out[2:, :] = -R_diff
        L = np.linalg.norm(R_diff)
        out = out * n12 * (L - L0) / L
        return out

    # Generate noise

    def equation(R_next, dW, R_current):
        R_next = np.reshape(R_next, (-1, 1))
        eqn = R_next - (R_current + (alpha * a(R_next) + (1 - alpha) * a(R_current))
                        * dt + b * dW)
        return np.reshape(eqn, -1)

    # Iterate over steps
    R_next = R0.reshape((4, 1))
    with stopwatch('Simulation'):
        for i in trange(N, desc='Simulating'):
            for j in range(N_intermediate_points):
                # Solve the equation
                # print('a', R[:, i, np.newaxis])
                # print('g', np.shape(R[:, i, np.newaxis]))
                dW = np.random.normal(0, np.sqrt(dt), size=(4, 1))
                sol = root(equation, R_next, args=(dW, R_next))
                # print(sol)
                R_next = np.reshape(sol.x, (4, 1))

            R[:, i + 1] = R_next[:, 0]

    # # print('R not rotated', R)
    # # Rotate the result if necessary. Q is the rotation matrix
    # # phi = angle / 180 * np.pi
    # phi = angle
    # # print('phi', phi)
    # # raise RuntimeError('stop')
    # Q = np.array([[cos(phi), - sin(phi), 0, 0],
    #               [sin(phi), cos(phi), 0, 0],
    #               [0, 0, cos(phi), -sin(phi)],
    #               [0, 0, sin(phi), cos(phi)]
    #               ])
    # # print(Q)
    # # print(R[:, 0])
    #
    # R_rotated = R.copy()
    # if angle:
    #     for i in range(N + 1):
    #         # print(Q @ R[:, i, None])
    #         R_rotated[:, i, None] = Q @ R[:, i, None]
    #
    # R = R_rotated
    # # print(R[:, 0])

    # # Reorder components to [x1, y1, x2, y2]
    # new_R = R[(0, 2, 1, 3), :]
    # print('new_R', new_R)

    # # Resample to the original time step
    # R = R[:, ::N_intermediate_points]
    # t = t[::N_intermediate_points]
    print('Calculated number of points: ', np.shape(R)[1])

    #  Displacements
    dR = R[:, 1:] - R[:, :-1]
    # print(f'R array size is {sys.getsizeof(R) / 2 ** 20} MB')

    # Save the trajectories and simulation parameters
    # print(t, R, dR)
    dict_data = {'t': t, 'R': R, 'dR': dR, **parameters}
    # print('full dict', dict_data)
    # save_data(dict_data=dict_data, hash=hash)

    # print('R: ', R)

    if plot:
        plot_trajectories(t, R, dR, parameters, save=save_figure, show=show)
    #
    # return (t, X, dX, Y, dY)
    return t, R, dR, hash


def plot_trajectories(t, R, dR, true_parameters, save=False, show=True):
    # Calculate and plot the link angle
    ratio = (R[3, :] - R[1, :]) / (R[2, :] - R[0, :])
    angle = 180 / np.pi * np.arctan((R[3, :] - R[1, :]) / (R[2, :] - R[0, :]))
    phi0 = 180 / np.pi * np.arctan2(R[3, 0] - R[1, 0], R[2, 0] - R[0, 0])
    Ls = np.sqrt(((R[0, :] - R[2, :])) ** 2 + ((R[1, :] - R[3, :])) ** 2)

    fig = plt.figure(1, clear=True)
    plt.plot(R[0, :], R[1, :], label='1')
    plt.plot(R[2, :], R[3, :], label='2')
    plt.plot((R[2, :] + R[0, :]) / 2, (R[3, :] + R[1, :]) / 2, label='baricenter')
    plt.xlabel('$x, \mu$m')
    plt.ylabel('$y, \mu$m')
    plt.axis('equal')
    plt.legend()
    plt.title(r'$\phi_0 = {0:.0f}\degree$'.format(phi0))

    fig = plt.figure(2, clear=True)
    plt.plot(t, angle)
    plt.xlabel('$t$, s')
    plt.ylabel('angle, $\degree$')
    plt.ylim([-90, 90])

    fig = plt.figure(3, clear=True)
    plt.plot(t, Ls)
    plt.xlabel('$t$, s')
    plt.ylabel('L, $\mu$m')
    plt.ylim(ymin=0)

    # plt.axis('equal')
    # plt.legend()

    # Estimate the parameters of the baricenter to check simulations
    dt = t[1] - t[0]
    dR2_baricenter = ((dR[0, :] + dR[2, :]) / 2) ** 2 + ((dR[1, :] + dR[3, :]) / 2) ** 2
    var_estimate_baricenter = np.mean(dR2_baricenter) * \
                              len(dR2_baricenter) / (len(dR2_baricenter) - 1)
    var_expected_baricenter = dt * (true_parameters['D1'] + true_parameters['D2'])
    print('\nSimulation completed.'
          '\nBaricenter variance estimated (um^2): {0:.2g}\texpected: {1:.2g}\tratio: {2:.2g}'.format(
        var_estimate_baricenter, var_expected_baricenter,
        var_estimate_baricenter / var_expected_baricenter))

    # Check the evolution of the link length
    dLs = Ls[1:] - Ls[:-1]
    L_mean = np.mean(Ls)
    L_var = np.var(dLs, ddof=1)
    L0, n12, D1, D2 = [true_parameters[key] for key in 'L0 n12 D1 D2'.split()]
    if n12 != 0:
        L_var_expected = (D1 + D2) / 2 / n12 * (1 - np.exp(-4 * n12 * dt))
    else:
        L_var_expected = np.inf
    print('\nLink length mean estimated (um): {0:.2g}\texpected: {1:.2g}'.format(L_mean, L0))
    print(
        'Link length variance estimated (um^2): {0:.2g}\texpected: {1:.2g}\tratio: {2:.2g}'.format(
            L_var, L_var_expected, L_var / L_var_expected))

    # Check the evolution of the link angle
    d_angle = (angle[1:] - angle[:-1]) / 180 * np.pi
    angle_mean = np.mean(d_angle)
    angle_var = np.var(d_angle, ddof=1)
    angle_var_expected = 2 * (D1 + D2) * dt / L0 ** 2
    print('\nLink angle mean estimated (rad): {0:.2g}\texpected: {1:.2g}'.format(angle_mean, 0))
    print(
        'Link angle variance estimated (rad^2): {0:.2g}\texpected: {1:.2g}\tratio: {2:.2g}'.format(
            angle_var, angle_var_expected, angle_var / angle_var_expected))

    if show:
        plt.show()
    if save:
        plt.savefig('trajectory.png')


# # %% Tests
if __name__ == '__main__':
    # % matplotlib
    # T = 1e-0
    # # dt = 2e-8
    # k12 = 1e-8  # kg/s^2
    # gamma = 1e-8  # kg/s
    # print('k/gamma*dt:', k12 / gamma * dt)
    # print('k/gamma*T:', k12 / gamma * T)
    true_parameters = {'D1': 0.4,  # 2.0
                       'D2': 0.4,
                       'n1': 0,
                       'n2': 0,
                       'n12': 2e-5 / 0.05,
                       'dt': 0.05,
                       'angle': -np.pi / 3,  # rad
                       'L0': 100 * np.sqrt(4 * 0.4 * 0.05),
                       'trial': 0,
                       'M': 2000}

    t, R, dR, hash = simulate_a_free_hookean_dumbbell(
        parameters=true_parameters, recalculate=True, verbose=True, plot=True)

    li = -1
    lj = -1
    dt = 0.05
    (np.exp(dt * (li + lj)) - 1) / (li + lj)
    np.arctan(-4.15101287)
    np.arctan(np.array([323.60059621, -550.72591132], dtype=np.float64))

    exponent = dt * (li + lj)
    exponent
    dt + exponent * dt / 2

    # print('X:', X)
    # print('dx:', dX)
    # #
    #
    # print('D_hat:', np.var(dX[:, :-1], axis=1, ddof=1) / 2 / dt)
    # # print([D1, D2])
    # print('L12:', X[1, :] - X[0, :])
    # print('<L12>:', np.mean(X[1, :] - X[0, :]))
    z = np.array([1, 2, 3, 4])
    z[::2]
    z = np.array([[2], [3], [4]])
    z[:2]
