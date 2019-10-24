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
from scipy.optimize import minimize, root_scalar
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
np.log10(10)
# %%
lg_BF_vals = plot_link_strength_dependence(seed=0, verbose=False, recalculate=False, dry_run=False)

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

log(0.01) / log(1 - 0.059)


a = [(-30.04810117530901, 32.27033782453755), (-30.01432284829057, 26.355612276667458), (-29.997582022660815, np.nan), (-29.97355548125588, 28.611895275727036), (-29.96787446073086, 26.51190492202739), (-29.888495404011287, 24.593342475424077), (-29.786553630524743, 24.924299022330406), (-29.743730662266664, 25.489111556367966), (-29.719167407551936, 24.532257585456986), (-29.71486478399119, 23.81624515576479), (-29.65172186565715, 22.63467779939017), (-29.579593043415166, 23.22781042962825), (-29.571536333003728, 26.022702792326697), (-29.55882936776872, 27.527183092029574), (-29.463757902545886, 22.426147630952347), (-29.360040621109352, 25.359669140724325), (-29.358125110457383, 26.427054284930698), (-29.32948890938202, 23.348005224429226), (-29.262966957796205, 22.33501379927597), (-29.214034892733352, 25.853165955980664), (-29.168492143923437, 23.74669867746363), (-29.167714731193914, 22.340766163022913), (-29.007963731308685, 25.126777661002258), (-28.90388874242639, 22.666082008805443), (-28.854055792897714, 23.1174532174566), (-28.741179510893346, 22.56179060903569), (-28.643552180934414, 24.284255395809975), (-28.631988729728327, 26.22941536919992), (-28.61699511269636, 26.74071449571761), (-28.56375400813524, 26.08981126575205), (-28.55189236058577, 26.005602309462297), (-28.520308639415486, 24.010211395508875), (-28.460477913814135, 21.592641235078368), (-28.377794228440962, 23.680611554703297), (-28.376412811115323, 23.40837894739604), (-28.375054336390768, 27.415590281415064), (-28.338926103435593, 24.489930019511885), (-28.32619868634734, 23.68704930186904), (-28.311908933779655, 24.36365825742888), (-28.268323942797576, 22.244606211557226), (-28.179774918930747,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     22.2363309322808), (-28.082721431534914, 27.16056087490334), (-28.052582308534028, 21.556520526400256), (-27.982000124439274, 23.554397618796244), (-27.97485953247545, 20.621069921580307), (-27.922241950727543, 26.43556838037704), (-27.82968156018906, 23.855275283841912), (-27.820290549440426, 26.22993058047085), (-27.814949158576294, 26.65222672639886), (-27.80940163464408, 23.877489389872757), (-27.79301604466975, 21.764582601036143), (-27.761543176055405, 20.556065530975232), (-27.67155546582449, 28.862186034428507), (-27.598434420049927, 21.366361895548664), (-27.485954308382123, 24.244639941978605), (-27.27476577410269, 22.262733879406696), (-27.264248556155827, 26.879746675106276), (-27.260927890012336, 25.287190063959102), (-27.2569587326413, 20.170015959597645), (-27.186483718432306, 21.855773638757), (-27.116933711892724, 28.191924910537193), (-26.917221819066132, 21.503321604934442), (-26.87111674746021, 25.50330562349118), (-26.741232209173326, 22.41036873012941), (-26.449642625979042, 24.47885730008089), (-26.172993736017574, 16.351182563267805), (-26.101485723303046, 16.945636670282667), (-26.083216148590978, 22.03098121343651), (-25.73995651452972, 19.687993330276925), (-25.56312740093471, 21.682341431087465), (-25.508579793567485, 24.630587812082247), (-25.051042710322122, 20.237665233082048), (-24.46194531816514, 22.7221060953509), (-24.388190313727154, 22.058516431854372), (-23.475835463467032, 21.807258320001022), (-22.350389796481167, 22.04902150309465), (-20.91407833587448, 19.618018500083874), (-19.18964562747231, 20.35136144346226), (-17.486313538455512, 18.715535477972022), (-14.737127113659843, 17.306711737959823), (1.2834176160190027, -0.7808687246205561)]
# %%
plt.figure(4, clear=True)
plt.hist([a1[0] for a1 in a], bins=100)
plt.figure(5, clear=True)
plt.hist([a1[1] for a1 in a], bins=100)

tau = 1 / 100
root_scalar(lambda a: np.exp(-1 + a) - a / tau, bracket=[1, 1e7])
