# infer_D_through_autocorrelation_1D.py

try:
    bl_has_run

except Exception:
    %matplotlib
    %load_ext autoreload
    %autoreload 2
    bl_has_run = True

import hashlib
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from hash_me import hash_me
from simulate import simulate_2_particles

# %% Constants
D1 = 0.1  # um^2/s; 0.4
D2 = 1  # um^2/s
k12 = 1e-6  # kg/s^2. Somehting interesting happens at 5e-7
T = 5e1  # s
gamma = 1e-8    # viscous drag, in kg/s
D_ratios = [0.1, 0.5, 1.0,  5, 10]

# D1/D2
fig = plt.figure(num=2, clear=0)
fig.clear()
for D_ratio in D_ratios:
    # print(D_ratio)
    D1 = D2 * D_ratio

    recalculate = 0
    hash = hash_me(D1, D2, k12, T, gamma)
    filename = 'data/data_' + hash + '.pyc'

    if os.path.exists(filename) and recalculate == False:
        with open(filename, 'rb') as file:
            t, X, dX = pickle.load(file)
    else:
        t, X, dX = simulate_2_particles(D1=D1, D2=D2, k12=k12, gamma=gamma,
                                        T=T, plot=True, save=False)
        with open(filename, 'wb') as file:
            pickle.dump([t, X, dX], file, pickle.HIGHEST_PROTOCOL)

    dt = t[1] - t[0]

    # Covariance of dX
    particle = 2  # 1 or 2
    lags = range(10)
    cov = np.zeros(len(lags)) * np.nan
    variable = dX[particle - 1, :]
    # variable = np.sum(dX, axis=0) / 2

    N = len(variable)
    for i, lag in enumerate(lags):
        cov[i] = np.mean(variable[0:N - lag] * variable[lag:N])
    cov_norm = cov / (D2 * dt)
    # cn1 = cov_norm
    print(cov_norm)
    # print(cn10 - cn1)

    # Plot the covariances
    ax = fig.gca()
    ax.plot(lags, cov_norm, '-o', label=f'$D_1/D_2 = $ {D_ratio:.2f}')
    # plt.plot(lags, cn10-cn1, '-x')


ax.plot([0, np.max(lags)], [0] * 2, 'k')
plt.ylabel('$\\overline{\\Delta x_n \\Delta x_{n+j}}$')
plt.xlabel('Lag $j$')
plt.ylim([-1, 3.0])
plt.title(f'$D_2 = {D2:.2f}$')
plt.legend()

fig.show()
# %%
# figname = 'figs/covariance_dif_increase_other_D'
# fig.savefig(figname + '.pdf')
# fig.savefig(figname + '.png')
