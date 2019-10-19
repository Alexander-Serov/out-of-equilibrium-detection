"""
Investigate the behavior of connected particles through statistical tests based on a periodogram
"""

# try:
#     bl_has_run
#
# except Exception:
#     %matplotlib
#     %load_ext autoreload
#     %autoreload 2
#     bl_has_run = True

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

from likelihood import (estimate_sigma2_matrix,
                        get_ln_likelihood_func_2_particles_x_link,
                        get_ln_likelihood_func_no_link, get_MLE,
                        get_sigma2_matrix_func)
#  Load and plot data of the test result as a function of the link strength
from plot import (plot_diffusivity_dependence, plot_link_strength_dependence,
                  plot_localization_dependence, plot_periodogram)
from simulate import simulate_2_confined_particles_with_fixed_angle_bond
from support import get_rotation_matrix, locally_rotate_a_vector

#  Constants
D2 = 0.4  # um^2/s
D1 = 5 * D2   # um^2/s; 0.4
n1 = 2e3
n2 = 2e3
n12 = 10 * n2  # s^{-1}. Somehting interesting happens between [1e-9; 1e-6]
N = 101  # required number of points in a trajectory
dt = 0.05  # s 0.3
gamma = 1e-8    # viscous drag, in kg/s
L = 0.5
angle = 0
trial = 0   # the trial number
recalculate = False

k1, k2, k12 = np.array([n1, n2, n12]) * gamma
T = dt * (N - 1)  # s
M = N - 1

true_parameters = {name: val for name, val in zip(
    ('D1 D2 n1 n2 n12 gamma T dt angle L trial M'.split()),
    (D1, D2, n1, n2, n12, gamma, T, dt, angle, L, trial, M))}

# np.log(0.0167)
# np.log(10)
# np.log(2 * np.pi)
# np.log(0.2)
# print(f'A*dt scale: {np.array([n1, n2, n12]) * dt}')
# %%
lg_BF_vals = plot_link_strength_dependence(seed=0, verbose=False, recalculate=True, dry_run=False)

# %%
lg_BF_vals = plot_diffusivity_dependence(seed=0, verbose=False, recalculate=False, dry_run=False)

# %%
lg_BF_vals = plot_localization_dependence(seed=0, verbose=False, recalculate=False, dry_run=False)

# %%
np.log(0.01) / np.log(1 - 20658 / 22540)
3460 / 22540
np.log(0.01) / np.log(1 - 3460 / 22540)
1
# fit_params = {'D1': 0.009556111157819758, 'D2': 0.16340672297424766,
#               'n1': 5.050123555707306e2, 'n2': 127.64011605246695, 'n12': 23.693359560798886 * 0}
# estimate_sigma2_matrix(fit_params=fit_params, ks_fit=[
#                        11], true_parameters=true_parameters)  # %% Simulate the particle system
# # Set recalculate = True to force recalculation at every start
# # Otherwise the last results will be reloaded
# t, R, dR, hash = simulate_2_confined_particles_with_fixed_angle_bond(
#     true_parameters=true_parameters, plot=True, save_figure=False, recalculate=recalculate, seed=None)
# # print(hash)
#
#
# # %% Calculate the Bayes factor
# lg_bayes_factor, ln_evidence_with_link, ln_evidence_free = calculate_bayes_factor(
#     t=t, dR=dR, true_parameters=true_parameters, hash=hash, recalculate=False,  plot=False)
# print('lg Bayes factor: ', lg_bayes_factor)


# %%
1

# # %% Separate into coordinates
# X1 = R[0, :]
# X2 = R[2, :]
# Y1 = R[1, :]
# dX1 = dR[0, :]
# dX2 = dR[2, :]
# dY1 = dR[1, :]
#
# M = len(dX1)
#
# # Calculate the periodogram
# PX, PY, PX_norm, PY_norm, modes = calculate_periodogram(
#     dX1=dX1, dY1=dY1, dt=dt, D1=D1, D2=D2, dk=10, plot=True)
#
#
# # Construct a dictionary of true parameters
# true_parameters = {name: val for name, val in zip(
#     ('D1', 'D2', 'n1', 'n2', 'n12'),
#     (D1, D2, k1 / gamma, k2 / gamma, k12 / gamma))}
#
#
# # %% MLE and evidence for the model without a link
# no_link_true_params = {key: true_parameters[key] for key in ('D1', 'n1')}
# other_args = {a: 1 for a in ('D2', 'n2', 'n12')}
# # ln_lklh_no_link = get_ln_likelihood_func_no_link(ks=ks_fit, zs_x=PX_fit, zs_y=PY_fit, M=M, dt=dt)
# MLE_free, ln_evidence_free, max_free, ln_model_evidence_direct = get_MLE(
#     ks=ks_fit, zs_x=PX_fit, zs_y=PY_fit, M=M, dt=dt, link=False, start_point=no_link_true_params)
# print('> all-free. Optimization results:', MLE_free)
# print('> n1/D1. True: %g, Inferred: %g ' % (n1 / D1, MLE_free['n1'] / MLE_free['D1']))
# print('> **all-free-evidence', ln_evidence_free, '**')
# no_link_hess = np.linalg.inv(max_free.hess_inv)
# # print('Hessian: ', link_hess)
# print('Hessian eigenvalues: ', np.linalg.eigvals(no_link_hess))
# print('Hessian eigensystem: ', np.linalg.eig(no_link_hess))
#
# # no prior: -11.57, with prior: -17.6
#
# # print('ln lklh free', ln_lklh_no_link(**MLE_free))
# # print('ln lklh free true params', ln_lklh_no_link(**no_link_true_params))
# # estimate_sigma2_matrrix(MLE_free)
#
# # %% Infer the MLE for the model with link
# if 1:
#     ln_lklh = get_ln_likelihood_func_2_particles_x_link(
#         ks=ks_fit, zs_x=PX_fit, zs_y=PY_fit, M=M, dt=dt)
#
#     MLE_link, ln_evidence_with_link, max_link, ln_model_evidence_direct = get_MLE(
#         ks=ks_fit, zs_x=PX_fit, zs_y=PY_fit, M=M, dt=dt, link=True, start_point=true_parameters)
#     print('all-link', MLE_link)
#     print('True params: ', true_parameters)
#     print('**all-link-evidence', ln_evidence_with_link, '**')
#     print('ln lklh link', ln_lklh(**MLE_link))
#     print('ln lklh link true params', ln_lklh(**true_parameters))
#     # estimate_sigma2_matrrix(MLE_link, M=M, dt=dt, ks_fit=ks_fit, true_parameters=true_parameters)
#
#     try:
#         print('det', np.sqrt(np.linalg.det(max_link.hess_inv)))
#     except:
#         pass
#     link_hess = np.linalg.inv(max_link.hess_inv)
#     # print('Hessian: ', link_hess)
#     print('Hessian eigenvalues: ', np.linalg.eigvals(link_hess))
#     print('Hessian eigensystem: ', np.linalg.eig(link_hess))
# # no prior: -21.94, with prior:-38.45
# # was (with a wrong start): 199; with the exact start: 196. It's even worth (i.e. probably random). This suggests I converge to different MLE from different starts, and the evidence estimate in them changes.
#
# # %% The Bayes factor
# print('ln evidence with and without link', ln_evidence_with_link, ln_evidence_free)
# print('lg Bayes factor for the presence of the link',
#       (ln_evidence_with_link - ln_evidence_free) / log(10))

# % Estimate sigma2 matrix
# true_parameters = {name: val for name, val in zip(
#     ('D1', 'D2', 'n1', 'n2', 'n12'),
#     (D1, D2, k1 / gamma, k2 / gamma, k12 / gamma))}
# other_args = {a: 1 for a in ('D2', 'n2', 'n12')}

# # true_parameters = (D1, k1 / gamma)
# # , lg_lklh(**MLE), lg_lklh(*true_parameters))
# print('True values: ', true_parameters)
# print('Found values: ', MLE)
# print('True weight of the 2nd particle: %.2g' % np.abs(s2_mat_true[0, 2] / s2_mat_true[0, 0]))
# print('Evals: ', ln_lklh(**true_parameters), ln_lklh(**MLE))
#
# # % Analyze the no-link model
#
# ln_lklh_no_link(1, 1)
#
# no_link_true_params = {name: val for name, val in zip(
#     ('D1', 'D2', 'n1', 'n2', 'n12'),
#     (D1, 1, k1 / gamma, k2 / gamma, 0 * k12 / gamma))}
# s2_mat_func = get_sigma2_matrix_func(M=M, dt=dt, alpha=None, **no_link_true_params)
# s2_mat_true_no_link = s2_mat_func(k=k)
#
# , lg_lklh(**MLE)
# PX_fit
#
# start_point = (1, 1, 1e-8, 1e-8, 1e-8, 1e-8)
#
# def minimize_me(args):
#     return -lg_lklh(*args)
#
# max = minimize(minimize_me, start_point, options={'disp': True})
# max.x
#
# # % Plot phase space of Battle
# fig = plt.figure(num=3)
# fig.clf()
# plt.plot(X1, X2)
# plt.xlabel('X1')
# plt.ylabel('X2')
# fig.show()

# f'{False:b}'

np.linalg.det(np.ones((2, 2)))
list(zip(*[(1, 1, 1), (2, 2, 2)]))

opts = [{'points': (0.3403266133358649, 0.3403266133358649, 0.34033181423430503)}, {
    'points': (494.8060386783946, 494.8060386783946, 494.8147221045491)}]
[x for x in opts[0]['points']]


a = [1, 2, 0.1]
sorted(a)[-1:]
