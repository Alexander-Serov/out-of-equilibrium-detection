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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import cos, exp, sin
from tqdm import tqdm, trange

from stopwatch import stopwatch

# from hash_me import hash_me


def simulate_2_confined_particles_with_fixed_angle_bond(D1=1.0, D2=1.0, k12=4e-7, k1=4e-7, k2=4e-7, gamma=1e-8, angle=30, dt=0.01, L=2, T=1, plot=True, save=True, file=r'.\trajectory.dat', seed=None):
    """
    Simulate the trajectories of two particles connected by 1 spring and only along the x axis.
    Units of measurements:
    D1, D2 --- um^2/s,
    gamma --- kg/s,
    k12 --- kg/s^2,
    T --- s,
    angle --- angle of bond beetween the particles in the lab system measured counterclockwise (in degrees).
    """

    # % Constants
    # kB = 1.38e-11  # kg*um^2/s^2/K
    atol = 1e-16
    rtol = 1e-6
    R0 = [0, 0, L, 0]
    # max_terms = 100
    # min_N = int(1e4)
    n1, n2, n12 = np.array([k1, k2, k12]) / gamma

    # w0 = k12 / gamma
    A = np.array([[-n1 - n12, 0, n12, 0],
                  [0, -n1, 0, 0],
                  [n12, 0, -n2 - n12, 0],
                  [0, 0, 0, -n2]])
    a = L * np.array([[-n12, 0, n2 + n12, 0]]).transpose()
    b = np.diag(np.sqrt([2 * D1, 2 * D1, 2 * D2, 2 * D2]))
    # print(b)

    lambdas, U = np.linalg.eig(A)
    Um1 = np.linalg.inv(U)
    diag = np.diag(lambdas)
    # Am1 = np.linalg.inv(A)

    # Choose dt and N (number of jumps)
    # dt = 1e-2
    N = np.ceil(T / dt).astype(int)
    t = np.arange(N + 1) * dt

    # R0 = np.transpose([np.hstack([r10, r20])])
    R = np.zeros((4, N + 1)) * np.nan
    R[:, 0] = R0
    # Q0 = R0 + Am1 @ a

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
    cov_1_step = np.full_like(A, np.nan)
    atol_local = 1e-6
    for i, li in enumerate(lambdas):
        for j, lj in enumerate(lambdas):
            exponent = dt * (li + lj)
            # For the exponent smaller than atol=1e-6, use a series
            if abs(exponent) < atol_local:
                cov_1_step[i, j] = dt + exponent * dt / 2
            else:
                cov_1_step[i, j] = (np.exp(dt * (li + lj)) - 1) / (li + lj)

    # Sample the diagonal elements from the distribution:
    # Generation dimensions: 4 noise sources x N x 4 elements
    diag_noise_integrals = np.random.multivariate_normal(mean_1_step, cov_1_step, size=(4, N))

    # Calculate the return force integrals that do not change from step to step
    diag_return_force_integrals = np.zeros(4) * np.nan
    for i, l in enumerate(lambdas):
        if check_zero([l]):
            diag_return_force_integrals[i] = dt
        else:
            diag_return_force_integrals[i] = (np.exp(l * dt) - 1) / l

    # Iterate over steps
    for i in trange(N):
        R_next = (
            U @ np.diag(np.exp(lambdas * dt)) @ Um1 @ R[:, i, None]
            + U @ np.diag(diag_return_force_integrals) @ Um1 @ a
            + U @ np.diag(diag_noise_integrals[0, i, :]) @ Um1 @ b[:, 0, None]
            + U @ np.diag(diag_noise_integrals[1, i, :]) @ Um1 @ b[:, 1, None]
            + U @ np.diag(diag_noise_integrals[2, i, :]) @ Um1 @ b[:, 2, None]
            + U @ np.diag(diag_noise_integrals[3, i, :]) @ Um1 @ b[:, 3, None]
        )
        R[:, i + 1] = R_next[:, 0]

    # Rotate the result if necessary. Q is the rotation matrix
    phi = angle / 180 * np.pi
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

    if plot:
        fig = plt.figure(1, clear=True)
        plt.plot(R[0, :], R[1, :], label='1')
        plt.plot(R[2, :], R[3, :], label='2')
        plt.xlabel('$x, \mu$m')
        plt.ylabel('$y, \mu$m')
        plt.axis('equal')
        plt.legend()

        plt.show()
        if save:
            plt.savefig('trajectory.png')
    #
    # return (t, X, dX, Y, dY)
    return (t, R, dR)
#


# # %% Tests
if __name__ == '__main__':
    # % matplotlib
    T = 1e-0
    # dt = 2e-8
    k12 = 1e-8  # kg/s^2
    gamma = 1e-8  # kg/s
    # print('k/gamma*dt:', k12 / gamma * dt)
    # print('k/gamma*T:', k12 / gamma * T)
    simulate_2_confined_particles_with_fixed_angle_bond(save=False, angle=0)

    # print('X:', X)
    # print('dx:', dX)
    # #
    #
    # print('D_hat:', np.var(dX[:, :-1], axis=1, ddof=1) / 2 / dt)
    # # print([D1, D2])
    # print('L12:', X[1, :] - X[0, :])
    # print('<L12>:', np.mean(X[1, :] - X[0, :]))
