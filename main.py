"""
Investigate the behavior of connected particles through statistical tests based on a periodogram
"""


# %%
try:
    bl_has_run

except Exception:
    %matplotlib
    %load_ext autoreload
    %autoreload 2
    bl_has_run = True

import cProfile
import os
import pickle
import pstats
import re

import matplotlib.pyplot as plt
import numpy as np
from numpy import arctan, cos, log, pi, sin
from scipy.fftpack import fft
from scipy.optimize import minimize
from tqdm import tqdm

from hash_me import hash_me
from likelihood import (get_ln_likelihood_func_2_particles_x_link,
                        get_ln_likelihood_func_no_link, get_MLE,
                        get_sigma2_matrix_func)
from simulate import simulate_2_confined_particles_with_fixed_angle_bond
from support import (average_over_modes, get_rotation_matrix,
                     locally_rotate_a_vector)

# %% Constants
D1 = 1   # um^2/s; 0.4
D2 = 1  # um^2/s
n12 = 0  # s^{-1}. Somehting interesting happens between [1e-9; 1e-6]
n1 = 1
n2 = 1
N = 101  # required number of points in a trajectory
dt = 0.3  # s

gamma = 1e-8    # viscous drag, in kg/s
angle = 0
L = 20

k1, k2, k12 = np.array([n1, n2, n12]) * gamma
T = dt * (N - 1)  # s

print(f'A*dt scale: {np.array([n1, n2, n12]) * dt}')


# %% Simulate the particle system
# Set recalculate = True to force recalculation at every start
# Otherwise the last results will be reloaded
recalculate = 1
# Hash the parameters to be able to reload
hash = hash_me('2D', D1, D2, k12, T, dt, gamma, angle, L)
filename = 'data/data_' + hash + '.pyc'

if os.path.exists(filename) and not recalculate:
    with open(filename, 'rb') as file:
        t, R, dR = pickle.load(file)
else:
    t, R, dR = simulate_2_confined_particles_with_fixed_angle_bond(
        D1=D1, D2=D2, k12=k12, k1=k1, k2=k2, L=L, gamma=gamma, T=T, dt=dt, angle=angle, plot=True, save=False, seed=0)
    # Save
    with open(filename, 'wb') as file:
        pickle.dump([t, R, dR], file, pickle.HIGHEST_PROTOCOL)


# %% Resample the trajectory with the given time step and number of points
M = N - 1
R.shape
indices = range(M)  # np.arange(0, (M + 1) * scale, scale)
X1 = R[0, indices]
len(X1)
X2 = R[2, indices]
Y1 = R[1, indices]
dX1 = X1[1:] - X1[:-1]
dY1 = Y1[1:] - Y1[:-1]
dX2 = X2[1:] - X2[:-1]

# %% Calculate the periodogram P
dX1_image = dt * fft(dX1)
dY1_image = dt * fft(dY1)
df = 1 / dt / M
len(dX1_image)
# print(np.fft.fftfreq(M))
modes = np.arange(M)
PX = np.abs(dX1_image)**2 / (M * dt)
PY = np.abs(dY1_image)**2 / (M * dt)

# Normalize to D1 * dt
PX_norm = PX / (D1 * dt**2)
PY_norm = PY / (D1 * dt**2)

dk = 10
PX_norm_avg, modes_avg = average_over_modes(PX_norm, dk)
PY_norm_avg, modes_avg = average_over_modes(PY_norm, dk)

fig = plt.figure(num=2, clear=0)
ax = fig.gca()
plt.figure(num=2)
markersize = 15
ax.clear()
ax.scatter(modes_avg, PX_norm_avg, marker='o', label='x', s=markersize)
ax.scatter(modes_avg, PY_norm_avg, marker='x', label='y', s=markersize)
ax.scatter(modes_avg, PX_norm_avg - PY_norm_avg, marker='x', label='x-y', s=markersize)

# Add (D1+D2)/2
xlims = plt.xlim()
y_theor = [(1 + D2 / D1) / 2] * 2
plt.plot(xlims, y_theor, 'olivedrab')

plt.xlim([0, M])
plt.ylim(ymin=-2)
plt.xlabel('Mode k')
plt.ylabel('$P_k / ( D_1\Delta t^2)$')
plt.legend()
fig.show()
# help(ax.scatter)


# Note: for even M, one must be careful with fitting k=0, k=M/2 because they are real,
# for odd M, k=0 is real
# For the moment, I ignore these frequencies
if not M % 2:    # even M
    fit_indices = np.arange(1, M / 2, dtype=np.int)
else:  # odd M
    fit_indices = np.arange(1, (M - 1) / 2 + 1, dtype=np.int)

ks_fit = fit_indices
PX_fit = PX[fit_indices]
PY_fit = PY[fit_indices]

# plt.figure(4)
# plt.scatter(ks_fit, PY_fit)
true_parameters = {name: val for name, val in zip(
    ('D1', 'D2', 'n1', 'n2', 'n12'),
    (D1, D2, k1 / gamma, k2 / gamma, k12 / gamma))}


def estimate_sigma2_matrrix(fit_params):
    k = ks_fit[-1]

    s2_mat_func = get_sigma2_matrix_func(M=M, dt=dt, alpha=None, **true_parameters)
    s2_mat_true = s2_mat_func(k=k)

    s2_mat_func = get_sigma2_matrix_func(M=M, dt=dt, alpha=None, **fit_params)
    s2_mat_fit = s2_mat_func(k=k)

    print('True sigma2 matrix: ', s2_mat_true)
    print('Fit sigma2 matrix: ', s2_mat_fit)
    print('Fit-true sigma2 matrix: ', s2_mat_fit - s2_mat_true)


# %% MLE and evidence for the model without a link
no_link_true_params = {key: true_parameters[key] for key in ('D1', 'n1')}
other_args = {a: 1 for a in ('D2', 'n2', 'n12')}
# ln_lklh_no_link = get_ln_likelihood_func_no_link(ks=ks_fit, zs_x=PX_fit, zs_y=PY_fit, M=M, dt=dt)
MLE_free, ln_evidence_free, max_free, ln_model_evidence_direct = get_MLE(
    ks=ks_fit, zs_x=PX_fit, zs_y=PY_fit, M=M, dt=dt, link=False)
print('all-free. Optimization results:', MLE_free)
print('n1/D1. True: %g, Inferred: %g ' % (n1 / D1, MLE_free['n1'] / MLE_free['D1']))
print('all-free-evidence', ln_evidence_free)
# print('ln lklh free', ln_lklh_no_link(**MLE_free))
# print('ln lklh free true params', ln_lklh_no_link(**no_link_true_params))
# estimate_sigma2_matrrix(MLE_free)

# %% Infer the MLE for the model with link
if 0:
    ln_lklh = get_ln_likelihood_func_2_particles_x_link(
        ks=ks_fit, zs_x=PX_fit, zs_y=PY_fit, M=M, dt=dt)

    MLE_link, ln_evidence_with_link, max_link, ln_model_evidence_direct = get_MLE(
        ks=ks_fit, zs_x=PX_fit, zs_y=PY_fit, M=M, dt=dt, link=True)
    print('all-link', MLE_link)
    print('all-link-evidence', ln_evidence_with_link)
    print('ln lklh link', ln_lklh(**MLE_link))
    print('ln lklh link true params', ln_lklh(**true_parameters))
    estimate_sigma2_matrrix(MLE_link)

    try:
        print('det', np.sqrt(np.linalg.det(max_link.hess_inv)))
    except:
        pass


# %% The Bayes factor
# if 0:
    # print('ln evidence with and without link', ln_evidence_with_link, ln_evidence_free)
    # print('lg link Bayes factor', (ln_evidence_with_link - ln_evidence_free) / log(10))

    # # % Estimate sigma2 matrix
    # true_parameters = {name: val for name, val in zip(
    #     ('D1', 'D2', 'n1', 'n2', 'n12'),
    #     (D1, D2, k1 / gamma, k2 / gamma, k12 / gamma))}
    # other_args = {a: 1 for a in ('D2', 'n2', 'n12')}
    #
    #
    #
    # # true_parameters = (D1, k1 / gamma)
    # # , lg_lklh(**MLE), lg_lklh(*true_parameters))
    # print('True values: ', true_parameters)
    # print('Found values: ', MLE)
    # print('True weight of the 2nd particle: %.2g' % np.abs(s2_mat_true[0, 2] / s2_mat_true[0, 0]))
    # print('Evals: ', ln_lklh(**true_parameters), ln_lklh(**MLE))

    # # % Analyze the no-link model
    #
    # ln_lklh_no_link(1, 1)
    #
    # no_link_true_params = {name: val for name, val in zip(
    #     ('D1', 'D2', 'n1', 'n2', 'n12'),
    #     (D1, 1, k1 / gamma, k2 / gamma, 0 * k12 / gamma))}
    # s2_mat_func = get_sigma2_matrix_func(M=M, dt=dt, alpha=None, **no_link_true_params)
    # s2_mat_true_no_link = s2_mat_func(k=k)

    # , lg_lklh(**MLE)
    # PX_fit
    #
    # start_point = (1, 1, 1e-8, 1e-8, 1e-8, 1e-8)
    #
    #
    # def minimize_me(args):
    #     return -lg_lklh(*args)
    #
    #
    # max = minimize(minimize_me, start_point, options={'disp': True})
    # max.x

    # # % Plot phase space of Battle
    # fig = plt.figure(num=3)
    # fig.clf()
    # plt.plot(X1, X2)
    # plt.xlabel('X1')
    # plt.ylabel('X2')
    # fig.show()
