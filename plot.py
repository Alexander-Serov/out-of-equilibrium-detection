"""
Contains all plot functions
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from numpy import log10
from tqdm import tqdm, trange

from calculate import (calculate_bayes_factor,
                       simulate_and_calculate_Bayes_factor_terminal)
from constants import color_sequence
from simulate import simulate_2_confined_particles_with_fixed_angle_bond
from support import get_cluster_args_string

# Plot parameters
alpha_shade = 0.3
confidence_level = 0.95


def plot_periodogram(modes_avg, PX_norm_avg, PY_norm_avg, D1, D2, M):

    # Constants
    markersize = 15

    fig = plt.figure(num=2, clear=0)
    ax = fig.gca()
    plt.figure(num=2)

    ax.clear()
    ax.scatter(modes_avg, PX_norm_avg, marker='o', label='x', s=markersize)
    ax.scatter(modes_avg, PY_norm_avg, marker='x', label='y', s=markersize)
    ax.scatter(modes_avg, PX_norm_avg - PY_norm_avg, marker='x', label='x-y', s=markersize)

    # Add (D1+D2)/2
    xlims = plt.xlim()
    # y_theor = [(1 + D2 / D1) / 2] * 2
    # plt.plot(xlims, y_theor, 'olivedrab')

    plt.xlim([0, M])
    # plt.ylim(ymin=-2)
    plt.xlabel('Mode k')
    plt.ylabel('$P_k / ( D_1\Delta t^2)$')
    plt.legend()
    fig.show()


def plot_link_strength_dependence(seed=None, verbose=False, recalculate=False, dry_run=False):
    """
    The function loads data for the specified parameters and plots the link strength dependence plot.

    If dry_run = True, the calculations are not performed, but instead an arguments file is created to be fed into cluster.
    Note that the means and the confidence intervals are calculated for lg(B), not for B.
    """
    # Neeed to specify parameters to be able to load the right files

    # %% Constants
    D2 = 0.4  # um^2/s
    D1 = 5 * D2   # um^2/s; 0.4
    n1 = 2e3
    n2 = 2e3
    # n12 = 10 * n2  # s^{-1}. Somehting interesting happens between [1e-9; 1e-6]
    Ms = [100, 200]  # required number of points in a trajectory; 100
    # N = 101

    dt = 0.05  # s 0.3
    gamma = 1e-8    # viscous drag, in kg/s
    L = 0.5
    angle = 0
    # trial = 0   # the trial number
    # recalculate = False

    arguments_file = 'arguments.dat'
    # color = [0.1008,    0.4407,    0.7238]

    # Parameters varying in the plot
    trials = 50  # 1000
    n12_range = [1e0, 1e7]
    n12_points = 100
    n12s = np.logspace(log10(n12_range[0]), log10(n12_range[1]), num=n12_points)

    # k1, k2, k12 = np.array([n1, n2, n12]) * gamma

    if dry_run:
        if os.path.exists(arguments_file):
            os.unlink(arguments_file)
        file = open(arguments_file, 'a')

    lg_BF_vals = np.full([len(Ms), n12_points, trials], np.nan)
    for trial in trange(trials, desc='Loading/calculating trial data'):
        if seed is not None:
            trial_seed = seed + trial
        else:
            trial_seed = None

        for ind_M, M in enumerate(Ms):
            T = dt * M  # s
            for ind_n12, n12 in enumerate(n12s):
                args_string = get_cluster_args_string(
                    D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, T=T, dt=dt, angle=angle, L=L, trial=trial, M=M, verbose=verbose, recalculate=recalculate)
                if dry_run:
                    file.write(args_string)

                else:
                    # print(args_string)
                    lg_BF_vals[ind_M, ind_n12, trial], ln_evidence_with_link, ln_evidence_free = simulate_and_calculate_Bayes_factor_terminal(
                        args_string)

            # print('Iteration ', i)
            # true_parameters = {name: val for name, val in zip(
            #     ('D1 D2 n1 n2 n12 gamma T dt angle L trial M'.split()),
            #     (D1, D2, n1, n2, n12, gamma, T, dt, angle, L, trial, M))}
            #
            # t, R, dR, hash = simulate_2_confined_particles_with_fixed_angle_bond(
            #     true_parameters=true_parameters, plot=False, save_figure=False, recalculate=recalculate, seed=trial_seed)

            # Load the Bayes factor

            # calculate_bayes_factor(
            #     t=t, dR=dR, true_parameters=true_parameters, hash=hash, recalculate=recalculate,  plot=False, verbose=verbose)

    if dry_run:
        file.close()
        print('Dry run finished. Arguments file created')
        if verbose:
            print('Warning: verbose was active')
        return np.nan

    # print(lg_BF_vals)

    # %% Calculating means and CI
    median_lg_BFs = np.nanmedian(lg_BF_vals, axis=2)
    # print('mean', median_lg_BFs)

    # BF_vals = 10**lg_BF_vals

    if trials > 1:
        CIs = np.full([len(Ms), n12_points, 2], np.nan)
        CIs[:, :, 0] = np.nanquantile(lg_BF_vals, (1 - confidence_level) / 2, axis=2)  # 0.025
        CIs[:, :, 1] = np.nanquantile(lg_BF_vals, 1 - (1 - confidence_level) / 2, axis=2)  # 0.975
        # print('CIs: ', np.log10(CIs))

    # %% Actual plotting
    fig = plt.figure(num=3, clear=True)

    # Confidence intervals
    # ax = plt.gca()
    xs = n12s / n1
    for ind_M in range(len(Ms)):
        color = color_sequence[ind_M]
        if trials > 1:

            plt.fill_between(xs, CIs[ind_M, :, 0], CIs[ind_M, :, 1],
                             alpha=alpha_shade, color=color)
            # plt.plot(n12s, np.log10(CIs[:, 0]), '-', color='g', alpha=alpha_shade)
            # plt.plot(n12s, np.log10(CIs[:, 1]), '-', color='g', alpha=alpha_shade)

        # Mean
        plt.plot(xs, median_lg_BFs[ind_M, :], color=color, label=f'M={Ms[ind_M]:d}')

    # Significance levels
    xlims = plt.xlim()
    plt.plot(xlims, [-1] * 2, '--', color='k')
    plt.plot(xlims, [1] * 2, '--', color='k')

    plt.xscale('log')
    plt.xlabel('$n_{12}/n_1$')
    plt.ylabel('Median $\mathrm{lg}(B)$')
    plt.title(
        f'trials={trials}, D1={D1:.2f}, D2={D2:.2f}, n1={n1:.1e}, n2={n1:.1e}, dt={dt}, L={L}')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    fig_folder = 'figures'
    figname = f'link_dependence'
    figpath = os.path.join(fig_folder, figname)
    plt.savefig(figpath + '.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(figpath + '.pdf', bbox_inches='tight', pad_inches=0)

    return lg_BF_vals


def plot_diffusivity_dependence(seed=None, verbose=False, recalculate=False, dry_run=False):
    """
    The function loads data for the specified parameters and plots the diffusivity dependence plot.

    If dry_run = True, the calculations are not performed, but instead an arguments file is created to be fed into cluster.
    Note that the means and the confidence intervals are calculated for lg(B), not for B.
    """
    # Neeed to specify parameters to be able to load the right files

    # %% Constants
    D2 = 0.4  # um^2/s
    # D1 = 5 * D2   # um^2/s; 0.4
    n1 = 2e3
    n2 = 2e3
    n12 = n1 * 10
    # n12 = 10 * n2  # s^{-1}. Somehting interesting happens between [1e-9; 1e-6]
    Ms = [100, 200]  # required number of points in a trajectory; 100
    # N = 101

    dt = 0.05  # s 0.3
    gamma = 1e-8    # viscous drag, in kg/s
    L = 0.5
    angle = 0
    # trial = 0   # the trial number
    # recalculate = False

    arguments_file = 'arguments.dat'
    # color = [0.1008,    0.4407,    0.7238]

    # Parameters varying in the plot
    trials = 50  # 1000
    D1_ratio_range = np.array([1e-2, 1e2])
    D1_range = D1_ratio_range * D2
    D1_points = 50
    D1s = np.logspace(log10(D1_range[0]), log10(D1_range[1]), num=D1_points)

    # k1, k2, k12 = np.array([n1, n2, n12]) * gamma

    if dry_run:
        if os.path.exists(arguments_file):
            os.unlink(arguments_file)
        file = open(arguments_file, 'a')

    lg_BF_vals = np.full([len(Ms), D1_points, trials], np.nan)

    for trial in trange(trials, desc='Loading/calculating trial data'):
        # if seed is not None:
        #     trial_seed = seed + trial
        # else:
        #     trial_seed = None
        for ind_M, M in enumerate(Ms):
            T = dt * M  # s
            for ind_D1, D1 in enumerate(D1s):
                args_string = get_cluster_args_string(
                    D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, T=T, dt=dt, angle=angle, L=L, trial=trial, M=M, verbose=verbose, recalculate=recalculate)
                if dry_run:
                    file.write(args_string)

                else:
                    # print(args_string)
                    lg_BF_vals[ind_M, ind_D1, trial], ln_evidence_with_link, ln_evidence_free = simulate_and_calculate_Bayes_factor_terminal(
                        args_string)

    if dry_run:
        file.close()
        print('Dry run finished. Arguments file created')
        if verbose:
            print('Warning: verbose was active')
        return np.nan

    # print(lg_BF_vals)

    # %% Calculating means and CI
    median_lg_BFs = np.nanmedian(lg_BF_vals, axis=2)
    # print('mean', median_lg_BFs)

    # BF_vals = 10**lg_BF_vals

    if trials > 1:
        CIs = np.full([len(Ms), D1_points, 2], np.nan)
        CIs[:, :, 0] = np.nanquantile(lg_BF_vals, (1 - confidence_level) / 2, axis=2)  # 0.025
        CIs[:, :, 1] = np.nanquantile(lg_BF_vals, 1 - (1 - confidence_level) / 2, axis=2)  # 0.975
        # print('CIs: ', np.log10(CIs))

    # %% Actual plotting
    fig = plt.figure(num=3, clear=True)

    # Confidence intervals
    # ax = plt.gca()
    xs = D1s / D2
    for ind_M in range(len(Ms)):
        color = color_sequence[ind_M]
        if trials > 1:

            plt.fill_between(xs, CIs[ind_M, :, 0], CIs[ind_M, :, 1],
                             alpha=alpha_shade, color=color)
            # plt.plot(n12s, np.log10(CIs[:, 0]), '-', color='g', alpha=alpha_shade)
            # plt.plot(n12s, np.log10(CIs[:, 1]), '-', color='g', alpha=alpha_shade)

        # Mean
        plt.plot(xs, median_lg_BFs[ind_M, :], color=color, label=f'M={Ms[ind_M]:d}')

    # Significance levels
    xlims = plt.xlim()
    plt.plot(xlims, [-1] * 2, '--', color='k')
    plt.plot(xlims, [1] * 2, '--', color='k')

    plt.xscale('log')
    plt.xlabel('$D_1/D_2$')
    plt.ylabel('Median $\mathrm{lg}(B)$')
    plt.title(
        f'trials={trials}, D2={D2:.2f}, n1={n1:.1e},\nn2={n1:.1e}, n12={n12:.1e}, dt={dt}, L={L}')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    fig_folder = 'figures'
    figname = f'diffusivity_dependence'
    figpath = os.path.join(fig_folder, figname)
    plt.savefig(figpath + '.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(figpath + '.pdf', bbox_inches='tight', pad_inches=0)

    return lg_BF_vals


def plot_localization_dependence(seed=None, verbose=False, recalculate=False, dry_run=False):
    """
    The function loads data for the specified parameters and makes the plot.

    If dry_run = True, the calculations are not performed, but instead an arguments file is created to be fed into cluster.
    Note that the means and the confidence intervals are calculated for lg(B), not for B.
    """
    # Neeed to specify parameters to be able to load the right files

    # %% Constants
    D2 = 0.4  # um^2/s
    D1 = 5 * D2   # um^2/s; 0.4
    # n1 = 2e3
    n2 = 2e3
    n12 = n2 * 10
    # n12 = 10 * n2  # s^{-1}. Somehting interesting happens between [1e-9; 1e-6]
    Ms = [100, 200]  # required number of points in a trajectory; 100
    # N = 101

    dt = 0.05  # s 0.3
    gamma = 1e-8    # viscous drag, in kg/s
    L = 0.5
    angle = 0
    # trial = 0   # the trial number
    # recalculate = False

    arguments_file = 'arguments.dat'
    # color = [0.1008,    0.4407,    0.7238]

    # Parameters varying in the plot
    trials = 20  # 1000
    n1_ratio_range = np.array([1e-2, 1e2])
    n1_range = n1_ratio_range * n2
    n1_points = 50  # 50
    n1s = np.logspace(log10(n1_range[0]), log10(n1_range[1]), num=n1_points)

    # k1, k2, k12 = np.array([n1, n2, n12]) * gamma

    if dry_run:
        if os.path.exists(arguments_file):
            os.unlink(arguments_file)
        file = open(arguments_file, 'a')

    lg_BF_vals = np.full([len(Ms), n1_points, trials], np.nan)

    for trial in trange(trials, desc='Loading/calculating trial data'):
        # if seed is not None:
        #     trial_seed = seed + trial
        # else:
        #     trial_seed = None
        for ind_M, M in enumerate(Ms):
            T = dt * M  # s
            for ind_n1, n1 in enumerate(n1s):
                args_string = get_cluster_args_string(
                    D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, T=T, dt=dt, angle=angle, L=L, trial=trial, M=M, verbose=verbose, recalculate=recalculate)
                if dry_run:
                    file.write(args_string)

                else:
                    # print(args_string)
                    lg_BF_vals[ind_M, ind_n1, trial], ln_evidence_with_link, ln_evidence_free = simulate_and_calculate_Bayes_factor_terminal(
                        args_string)

    if dry_run:
        file.close()
        print('Dry run finished. Arguments file created')
        if verbose:
            print('Warning: verbose was active')
        return np.nan

    # print(lg_BF_vals)

    # %% Calculating means and CI
    median_lg_BFs = np.nanmedian(lg_BF_vals, axis=2)
    # print('mean', median_lg_BFs)

    # BF_vals = 10**lg_BF_vals

    if trials > 1:
        CIs = np.full([len(Ms), n1_points, 2], np.nan)
        CIs[:, :, 0] = np.nanquantile(lg_BF_vals, (1 - confidence_level) / 2, axis=2)  # 0.025
        CIs[:, :, 1] = np.nanquantile(lg_BF_vals, 1 - (1 - confidence_level) / 2, axis=2)  # 0.975
        # print('CIs: ', np.log10(CIs))

    # %% Actual plotting
    fig = plt.figure(num=3, clear=True)

    # Confidence intervals
    # ax = plt.gca()
    xs = n1s / n2
    for ind_M in range(len(Ms)):
        color = color_sequence[ind_M]
        if trials > 1:

            plt.fill_between(xs, CIs[ind_M, :, 0], CIs[ind_M, :, 1],
                             alpha=alpha_shade, color=color)
            # plt.plot(n12s, np.log10(CIs[:, 0]), '-', color='g', alpha=alpha_shade)
            # plt.plot(n12s, np.log10(CIs[:, 1]), '-', color='g', alpha=alpha_shade)

        # Mean
        plt.plot(xs, median_lg_BFs[ind_M, :], color=color, label=f'M={Ms[ind_M]:d}')

    # Significance levels
    xlims = plt.xlim()
    plt.plot(xlims, [-1] * 2, '--', color='k')
    plt.plot(xlims, [1] * 2, '--', color='k')

    plt.xscale('log')
    plt.xlabel('$n_1/n_2$')
    plt.ylabel('Median $\mathrm{lg}(B)$')
    plt.title(
        f'trials={trials}, D1={D1:.2f}, D2={D2:.2f}, n12/n2={n12/n2:.1e}, n2={n1:.1e}, dt={dt}, L={L}')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    fig_folder = 'figures'
    figname = f'localization_dependence'
    figpath = os.path.join(fig_folder, figname)
    plt.savefig(figpath + '.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(figpath + '.pdf', bbox_inches='tight', pad_inches=0)

    return lg_BF_vals
