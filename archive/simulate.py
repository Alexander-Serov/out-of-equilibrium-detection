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
from tqdm import tqdm, trange

from stopwatch import stopwatch


def simulate_2_particles(D1=0.4, D2=2.0, k1=0, k2=0, k12=4e-4, gamma=1e-8, L12=2, T=1, x10=-1, x20=1, plot=True, save=True, file=r'.\trajectory.dat', seed=None):
    """
    Simulate the trajectories of two particles controlled by 3 springs.
    Units of measurements:
    D1, D2 --- um^2/s,
    gamma --- kg/s,
    k1, k2, k12 --- kg/s^2,
    x10, x20, L12 --- um,
    T --- s
    """

    # % Constants
    kB = 1.38e-11  # kg*um^2/s^2/K
    atol = 1e-16
    rtol = 1e-6
    max_terms = 100
    min_N = int(1e4)

    # D1 = 0.4  # um^2/s; 0.4
    # D2 = 5 * D1  # um^2/s
    # k1 = k2 = 0 * 1e-6   # spring constant, N/m = kg/s^2; 1e-6; 2e-5 - estimation from Berg-Sorensen article
    # k12 = 1e-1   # spring constant, N/m = kg/s^2; 2e-3, 3e-3; 1.5e-3
    #
    # gamma = 1e-8  # viscous drag, in kg/s; 4e-8
    # L12 = 2  # um
    # dt = 2e-7  # seconds
    # N = 50001  # time step
    # x10 = -L12 / 2
    # x20 = L12 / 2
    # filename = 'trajectory'
    # output_folder = r'D:\calculated_data\out-of-equilibrium_detection'
    # seed = None
    # D1 * gamma / kB

    # % Initialize
    np.random.seed(seed)
    A = np.array(
        [[-(k1 + k12), k12],
         [k12, -(k2 + k12)]]) / gamma
    a = np.array([k1 * x10 - k12 * L12, k2 * x20 + k12 * L12]) / gamma
    b1 = np.array([[1], [0]]) * np.sqrt(2 * D1)
    b2 = np.array([[0], [1]]) * np.sqrt(2 * D2)
    x0 = np.array([x10, x20])

    lambdas, U = np.linalg.eig(A)
    Um1 = np.linalg.inv(U)
    if np.max([k12, k1, k2]) / gamma * T < 10:
        logging.warning('Total simulation time may be too short to capture particle behavior.\nConsider simulating at least {:.1g} s'.format(
            10 * gamma / np.max([k12, k1, k2])))

    # Check integration step
    dt = np.min(1 / np.abs(2 * lambdas))
    N = 1 + int(np.ceil(T / dt))
    if N < min_N or np.all(np.isclose(lambdas, 0, atol=atol)):
        N = min_N
        dt = T / (N - 1)
    print('Automatically selected time step dt = {} s'.format(dt))

    # Make the time array
    t = np.arange(N) * dt

    def sum_exp_series(z):
        """
        Calculate the sum (exp(z*dt) - 1)/z
        """
        def series(k): return (dt * z)**(k - 1) / np.math.factorial(k)
        res = np.sum([series(k) for k in range(2, max_terms + 1)])

        if np.isnan(res):
            logging.exception(
                'Overflow encountered. Calculation aborted. Consider decreasing the time step')
        res = dt * (1 + res)
        return res

    # Term 1: initial condition
    def mat_exp(M):
        """Calculate matrix exponent
        """
        vambdas, V = np.linalg.eig(M)
        Vm1 = np.linalg.inv(V)
        return V @ np.diag(np.exp(vambdas)) @ Vm1

    # Term 2: the return force contribution
    diags_return_force = [sum_exp_series(l) for l in lambdas]
    Y2 = U @ np.diag(diags_return_force) @ Um1 @ a

    # Term 3: Noise contribution. The two integrals are correlated
    # Calculate covariance matrix
    mu = [0, 0]
    l0, l1 = lambdas
    cov_mat = np.zeros((2, 2))
    cov_mat[0, 0] = sum_exp_series(2 * l0)
    cov_mat[1, 1] = sum_exp_series(2 * l1)
    cov_mat[1, 0] = cov_mat[0, 1] = sum_exp_series(l0 + l1)

    # Sample
    def G():
        """
        Sample the multivariate distribution of the integrated noise
        """
        noise_integrals = np.random.multivariate_normal(mu, cov_mat)
        return np.diag(noise_integrals)

    def Y3():
        """Combine the noise integral"""
        Y3 = U @ G() @ Um1 @ b1 + U @ G() @ Um1 @ b2
        return Y3

    # Iterative calculations over a time step
    X = np.zeros((2, N))
    X[:, 0] = x0
    A_exponent = mat_exp(A * dt)
    for i in trange(N - 1):
        X[:, i + 1] += A_exponent @ X[:, i]
        X[:, i + 1] += Y2
        X[:, i + 1] += Y3()[:, 0]

    # Save
    dX = X[:, 1:] - X[:, 0:-1]
    dX = np.concatenate([dX, [[np.nan], [np.nan]]], axis=1)
    # print('D_hat:', np.var(dX[:, :-1], axis=1, ddof=1) / 2 / dt)
    # print('L12:', X[1, :] - X[0, :])
    # print('<L12>:', np.mean(X[1, :] - X[0, :]))

    if save:
        output = np.stack([t, X[0, :], dX[0, :], X[1, :], dX[1, :]], axis=1)
        output = pd.DataFrame(data=output, columns=['t', 'x', 'dx', 'x2', 'dx2'])
        output.to_csv(file, sep=';', index=False)

    if plot:
        fig = plt.figure(1, clear=True)
        plt.plot(t, X[0, :])
        plt.plot(t, X[1, :])
        plt.xlabel('t, s')
        plt.ylabel('$x, \mu$m')

        plt.show()
        if save:
            plt.savefig('trajectory.png')

    return (t, X, dX)
#


# # %% Tests
if __name__ == '__main__':
    # %matplotlib
    T = 1e-0
    # dt = 2e-8
    k12 = 1e-8  # kg/s^2
    gamma = 1e-8  # kg/s
    # print('k/gamma*dt:', k12 / gamma * dt)
    print('k/gamma*T:', k12 / gamma * T)
    _, X, dX = simulate_2_particles(save=False, T=T, k12=k12, x10=-1.0, x20=1.0, seed=0)

    print('X:', X)
    print('dx:', dX)
    #

    print('D_hat:', np.var(dX[:, :-1], axis=1, ddof=1) / 2 / dt)
    # print([D1, D2])
    print('L12:', X[1, :] - X[0, :])
    print('<L12>:', np.mean(X[1, :] - X[0, :]))
