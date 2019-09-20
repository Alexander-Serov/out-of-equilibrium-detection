"""
Analyze the inferred diffusivity in a system of 2 particles and 1 spring. The center of mass performs free diffusion.
"""


try:
    bl_has_run

except Exception:
    %matplotlib
    %load_ext autoreload
    %autoreload 2
    bl_has_run = True

import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from Bin import Bin, make_adaptive_mesh
from simulate import simulate_2_particles

D1 = 1  # um^2/s; 0.4
D2 = 10  # um^2/s
k12_lims = [1e-10, 1e-1]    # kg/s^2
M = 12
T = 1.0  # s
# dt = 1e-5
gamma = 1e-8    # viscous drag, in kg/s; 4e-8
# N = 1 + int(T / dt)  # trajectory length
k12s = 10**np.linspace(np.log10(k12_lims[0]), np.log10(k12_lims[1]), num=M)

1e-5 / gamma

D_infs = np.zeros((2, M)) * np.nan
L12s = np.zeros(M)

fig = plt.figure(2, clear=True)
for i in trange(M):
    k12 = k12s[i]
    t, X, dX = simulate_2_particles(D1=D1, D2=D2, k12=k12, gamma=gamma,
                                    T=T, plot=False, save=False)
    dt = t[1] - t[0]
    D_infs[:, i] = np.var(dX[:, :-1], axis=1, ddof=1) / 2 / dt
    L12s[i] = np.mean(X[1, :] - X[0, :])

    if not i:
        L120 = X[1, 0] - X[0, 0]

    # Plot
    fig.clf()
    plt.semilogx(k12s, D_infs[0, :], '-o', label='D1')
    plt.semilogx(k12s, D_infs[1, :], '-x', label='D2')

    # limit
    D_lim = (D1 + D2) / 2
    xlims = plt.xlim()
    plt.semilogx(xlims, np.ones(2) * D_lim, '-r')

    plt.xlabel('$k_{12}$, $kg/s^2$')
    plt.ylabel('$\hat D, \mu m^2/s$')
    plt.title('Calculated: {}/{}'.format(i + 1, M))
    plt.legend()

    plt.draw()
    plt.pause(1e-3)

plt.show()
D_infs


# plt.savefig('./D_vs_k12.png')

# Interparticle distance
# fig = plt.figure(3, clear=True)
# plt.plot(k12s, L12s)
#
# # expected
# xlims = plt.xlim()
# plt.plot(xlims, np.ones(2) * L120, '-r')
#
# plt.xlabel('$k_{12}$, $kg/s^2$')
# plt.ylabel('$<x_2-x_1>, \mu m$')
#
# plt.ylim(bottom=0)
#
# plt.show()


# %% Tests
# simulate_2_particles(D1=D1, D2=D2, k12=k12s[-1], N=int(1e6), dt=1e-6, plot=True, save=False)
