"""
Simulate random walk of two particles connected by a spring subject to two different diffusivities D1 and D2
"""

# try:
#     bl_has_run
#
# except Exception:
#     # %matplotlib
#     %load_ext autoreload
#     %autoreload 2
#     bl_has_run = True

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stopwatch import stopwatch

# from tqdm import tqdm, trange


def simulate_2_particles(D1=0.4, D2=2.0, k1=0, k2=0, k12=2e-4, gamma=1e-8, L12=2, dt=1e-4, N=10001, x10=-1, x20=1, plot=True, save=True, file=r'.\trajectory.dat', seed=None):
    """
    Simulate the trajectories of two particles controlled by 3 springs.
    Units of measurements:
    D1, D2 --- um^2/s,
    gamma --- kg/s,
    k1, k2, k12 --- kg/s^2,
    x10, x20, L12 --- um,
    dt --- s
    """

    # % Constants
    kB = 1.38e-11  # kg*um^2/s^2/K
    # D1 = 0.4  # um^2/s; 0.4
    # D2 = 5 * D1  # um^2/s
    # k1 = k2 = 0 * 1e-6   # spring constant, N/m = kg/s^2; 1e-6; 2e-5 - estimation from Berg-Sorensen article
    # k12 = 1e-4   # spring constant, N/m = kg/s^2; 2e-3, 3e-3; 1.5e-3
    #
    # gamma = 1e-8  # viscous drag, in kg/s; 4e-8
    # L12 = 2  # um
    # dt = 1e-4  # seconds
    # substeps = 100
    # N = 1 + int(1e4)  # time step
    # x10 = -L12 / 2
    # x20 = L12 / 2
    # filename = 'trajectory'
    # output_folder = r'D:\calculated_data\out-of-equilibrium_detection'
    # D1 * gamma / kB

    # % Initialize
    t = np.arange(N) * dt

    # % Calculation
    A = np.array(
        [[-(k1 + k12), k12],
         [k12, -(k2 + k12)]]) / gamma
    a = np.array([k1 * x10 - k12 * L12, k2 * x20 + k12 * L12]) / gamma
    b = np.sqrt(2 * np.array([D1, D2]))
    x0 = np.array([x10, x20])

    # noise
    print(x0)
    np.random.seed(seed)
    dW = np.random.randn(2, N) * np.sqrt(dt)

    def mat_exp(M):
        """Calculate matrix exponent
        """
        lambdas, U = np.linalg.eig(M)
        Um1 = np.linalg.inv(U)
        return U @ np.diag(np.exp(lambdas)) @ Um1

    # Calculate the return force contribution (term 2)
    lambdas, U = np.linalg.eig(A)
    # print(lambdas)
    # print(U)
    # print(A @ U[:, 1])
    Um1 = np.linalg.inv(U)
    # print(Um1 @ A @ U)
    diags_integrated = []
    for l in lambdas:
        if np.isclose(l, 0):
            diags_integrated.append(dt)
        else:
            diags_integrated.append((np.exp(l * dt) - 1) / l)
    Y2 = U @ np.diag(diags_integrated) @ Um1 @ a

    # Iterative calculations
    X = np.zeros((2, N))
    X[:, 0] = x0
    A_exponent = mat_exp(A * dt)
    # negative_A_exponent = mat_exp(-A * dt)
    # with stopwatch('Simulation'):
    # i=1
    for i in range(N - 1):
        X[:, i + 1] += A_exponent @ X[:, i]
        X[:, i + 1] += Y2
        X[:, i + 1] += A_exponent @ b * dW[:, i]

    np.mean(X, axis=1)
    A_exponent @ X[:, 2]

    if plot:
        fig = plt.figure(1, clear=True)
        plt.plot(t, X[0, :])
        plt.plot(t, X[1, :])
        plt.xlabel('t, s')
        plt.ylabel('$x, \mu$m')

        plt.show()
        if save:
            plt.savefig('trajectory.png')

    # % Save
    dX = X[:, 1:] - X[:, 0:-1]
    dX = np.concatenate([dX, [[np.nan], [np.nan]]], axis=1)

    if save:
        output = np.stack([t, X[0, :], dX[0, :], X[1, :], dX[1, :]], axis=1)
        output = pd.DataFrame(data=output, columns=['t', 'x', 'dx', 'x2', 'dx2'])
        # output = pd.DataFrame(data=output, columns=['t', 'x2', 'dx2', 'x', 'dx'])
        # file = os.path.join(output_folder, filename + '_k12={:2.2e}'.format(k12) + '.dat')
        output.to_csv(file, sep=';', index=False)

    return (t, X, dX)
#


# %% Tests of scales
# %matplotlib
simulate_2_particles(save=False)
#


# print(np.var(dX[:, :-1], axis=1, ddof=1) / 2 / dt)
# print([D1, D2])
# print(X[1, :] - X[0, :])
# np.var(dx1[:N - 1]) / 2 / dt
# np.sqrt(2 * dt) / L12
# np.sqrt(2 * dt) / 0.05
# np.sqrt(2 * dt_sim) / L12
# k12 * L12 / 4 / gamma * dt_sim
# np.sqrt(2 * D1 * dt_sim) * noise[0, 0]
# D1 * gamma / kB
# k12 * np.sqrt(2 * D1) * dt**(3 / 2) / gamma
