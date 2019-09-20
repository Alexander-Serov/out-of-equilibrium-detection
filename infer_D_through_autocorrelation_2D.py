# infer_D_through_autocorrelation_1D.py
# %%
try:
    bl_has_run

except Exception:
    %matplotlib
    %load_ext autoreload
    %autoreload 2
    bl_has_run = True

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from numpy import arctan, cos, pi, sin
from tqdm import tqdm

from hash_me import hash_me
from simulate_2d_confined import \
    simulate_2_confined_particles_with_fixed_angle_bond

# # %% Constants
# dr = np.array([-0.9, 18])
# U = get_rotation_matrix(dr)
# U @ dr
# %% Constants
D1 = 5   # um^2/s; 0.4
D2 = 1  # um^2/s
k12 = 1e-6   # kg/s^2. Somehting interesting happens at 5e-7
T = 5e1  # s
gamma = 1e-8    # viscous drag, in kg/s
lags = range(10)
# D_ratios = [0.1, 0.5, 1.0,  5, 10]


def get_rotation_matrix(dr):
    """Return a rotation matrix that rotates the given vector to be aligned with the positive direction of x axis"""
    tan_theta = - dr[1] / dr[0] if dr[0] != 0 else np.inf * (-dr[1])
    if dr[0] >= 0:
        theta = arctan(tan_theta)
    else:
        theta = arctan(tan_theta) + pi
    # print(theta / pi)
    # theta = 0
    R = np.array([[cos(theta), - sin(theta)], [sin(theta), cos(theta)]])
    return R


def locally_rotate_a_vector(dR, lag):
    """Create a vector, such that its component at position j+lag were defined by a rotation matrix for position j, which aligns the vector at position j with the axis x"""
    N = dR.shape[1]

    rotated = np.full_like(dR, np.nan)
    for j in range(N - lag):
        dr = dR[:, j]
        RM = get_rotation_matrix(dr)
        rotated[:, j] = RM @  dR[:, j + lag]
    return rotated


# 4.86/2.36
# D1/D2
fig = plt.figure(num=2, clear=0)
fig.clear()
ax = fig.gca()
ax.plot([0, np.max(lags)], [0] * 2, 'k')
# for D_ratio in D_ratios:
# print(D_ratio)
# D1 = D2 * D_ratio

recalculate = 1
hash = hash_me('2D', D1, D2, k12, T, gamma)
filename = 'data/data_' + hash + '.pyc'

if os.path.exists(filename) and recalculate == False:
    with open(filename, 'rb') as file:
        t, X, dX, Y, dY = pickle.load(file)
else:
    t, X, dX, Y, dY = simulate_2_confined_particles_with_fixed_angle_bond(D1=D1, D2=D2, k12=k12, gamma=gamma,
                                                                          T=T, plot=True, save=False)
    with open(filename, 'wb') as file:
        pickle.dump([t, X, dX, Y, dY], file, pickle.HIGHEST_PROTOCOL)

dt = t[1] - t[0]

# %

# Covariance of dX
particle = 2  # 1 or 2
variable = dX[particle - 1, :]
dR1 = np.array([dX[0, :], dY[0, :]])
dR2 = np.array([dX[1, :], dY[1, :]])

# variable = np.sum(dX, axis=0) / 2

# I will be looking at particle 2. The particle 1 will be the changing invisible one
N = dR2.shape[1]
cov_parallel = np.zeros(len(lags)) * np.nan
mean_orthogonal = np.full_like(cov_parallel, np.nan)
var_orthogonal = np.full_like(cov_parallel, np.nan)
cov_orthogonal = np.full_like(cov_parallel, np.nan)
cov_y = np.full_like(cov_parallel, np.nan)
# For each dx I need to find the appropriate rotated lagged vector
for i, lag in enumerate(tqdm(lags, desc='Calculating covariances')):
    # i = 0
    # lag = 0

    rot_0 = locally_rotate_a_vector(dR2, 0)
    rot_lag = locally_rotate_a_vector(dR2, lag)

    # Calculate the covariance between the x componnents of the new vectors
    cov_parallel[i] = np.nanmean(rot_0[0, :] * rot_lag[0, :])

    # Calculate the mean orthogonal component
    mean_orthogonal[i] = np.nanmean(rot_lag[1, :])

    # Calculate the variance of the orthogonal component
    var_orthogonal[i] = np.nanvar(rot_lag[1, :])
    cov_orthogonal[i] = np.nanmean(rot_0[0, :] * rot_lag[1, :])
    cov_y[i] = np.nanmean(rot_0[1, :] * rot_lag[1, :])

# print(cov_x / D2 / dt)

# cov[i] = np.mean(variable[0:N - 1 - lag] * variable[lag:N - 1])
cov_parallel_norm = cov_parallel / (D2 * dt)
mean_orthogonal_norm = mean_orthogonal / np.sqrt(D2 * dt)
var_orthogonal_norm = var_orthogonal / (D2 * dt)
cov_orthogonal_norm = cov_orthogonal / (D2 * dt)
cov_y_norm = cov_y / (D2 * dt)
# cn1 = cov_norm
print(cov_parallel_norm)
print(mean_orthogonal_norm)
print(var_orthogonal_norm)
print(cov_orthogonal_norm)
# print(cn10 - cn1)

# Plot the covariances
# , label=f'$D_1/D_2 = $ {D_ratio:.2f}')
ax.plot(lags, cov_parallel_norm, '-o',
        label='$\\overline{\\Delta x_n \\Delta x_{n+j}}$')
ax.plot(lags, mean_orthogonal_norm, '-o',
        label='$\\overline{\\Delta y_{n+j}}$')
ax.plot(lags, var_orthogonal_norm, '-o',
        label='$\\overline{(\\Delta y_{n+j})^2}$')
# ax.plot(lags, cov_orthogonal_norm, '-o',
#         label='$\\overline{\\Delta x_n \\Delta y_{n+j}}$')
ax.plot(lags, cov_y_norm, '-o',
        label='$\\overline{\\Delta y_n \\Delta y_{n+j}}$')
# plt.plot(lags, cn10-cn1, '-x')

plt.figure(num=2)
# plt.ylabel('$\\overline{\\Delta x_n \\Delta x_{n+j}}$')
plt.ylabel('Covariances')
plt.xlabel('Lag $j$')
plt.ylim([-1, 5.0])
plt.title(f'$D_2 = {D2:.2f}$')
plt.legend()

fig.show()
# %%
# figname = 'figs/covariance_dif_increase_other_D'
# fig.savefig(figname + '.pdf')
# fig.savefig(figname + '.png')
