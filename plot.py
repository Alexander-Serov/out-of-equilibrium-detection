"""
Contains all plot functions
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import log10
from tqdm import tqdm, trange

from calculate import (calculate_bayes_factor,
                       simulate_and_calculate_Bayes_factor_terminal, simulate_and_calculate_Bayes_factor)
from constants_main import color_sequence
from simulate import simulate_2_confined_particles_with_fixed_angle_bond
from support import get_cluster_args_string, set_figure_size
from stopwatch import stopwatch

# Plot parameters
alpha_shade = 0.25
confidence_level = 0.95
height_factor = 0.8
page_width_frac = 1
rows = 1
lw_theory = 1


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


def plot_link_strength_dependence(trials=20, n12_range=[1e-1, 1e3], verbose=False,
                                  recalculate_trajectory=False, recalculate_BF=False, dry_run=False,
                                  cluster=False, rotation=True):
    """
    The function loads data for the specified parameters and plots the link strength dependence plot.

    If dry_run = True, the calculations are not performed, but instead an arguments file is created to be fed into cluster.
    Note that the means and the confidence intervals are calculated for lg(B), not for B.
    """
    # Neeed to specify parameters to be able to load the right files

    # %% Constants
    # trials = 50  # 20  # 50  # 1000
    D2 = 0.4  # um^2/s
    D1 = 5 * D2  # um^2/s; 0.4
    n1 = 1
    n2 = 1
    # n12 = 10 * n2  # s^{-1}. Somehting interesting happens between [1e-9; 1e-6]
    Ms = [10, 100, 200]  # required number of points in a trajectory; 100
    # N = 101

    dt = 0.05  # s 0.3
    gamma = 1e-8  # viscous drag, in kg/s
    L = 20
    angle = 0
    # trial = 0   # the trial number
    # recalculate = False

    arguments_file = 'arguments.dat'
    # color = [0.1008,    0.4407,    0.7238]

    # Parameters varying in the plot

    # n12_range = [1e-1, 1e3]
    n12_points = 50
    n12s = np.logspace(log10(n12_range[0]), log10(n12_range[1]), num=n12_points)

    # k1, k2, k12 = np.array([n1, n2, n12]) * gamma

    # if cluster:
    # if os.path.exists(arguments_file):
    #     os.unlink(arguments_file)
    with open(arguments_file, 'a') as file:

        lg_BF_vals = np.full([len(Ms), n12_points, trials], np.nan)
        for trial in trange(trials, desc='Loading/scheduling calculations'):

            for ind_M, M in enumerate(Ms):
                for ind_n12, n12 in enumerate(n12s):
                    args_string = get_cluster_args_string(
                        D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, dt=dt, angle=angle, L=L,
                        trial=trial, M=M, verbose=verbose,
                        recalculate_trajectory=recalculate_trajectory,
                        recalculate_BF=recalculate_BF, rotation=rotation)
                    # print('Calculating with parameters: ', args_string)
                    lg_BF_vals[
                        ind_M, ind_n12, trial], ln_evidence_with_link, ln_evidence_free, loaded, dict_data = simulate_and_calculate_Bayes_factor_terminal(
                        args_string, cluster=cluster)
                    if cluster and not loaded:
                        file.write(args_string)
                    # raise RuntimeError('stop')

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

    if cluster and verbose:
        print('Warning: verbose was active')
        # return np.nan

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
    # fig = plt.figure(num=3, clear=True)
    fig = set_figure_size(num=3, rows=rows, page_width_frac=page_width_frac,
                          height_factor=height_factor)

    real_trials = np.min(np.sum(~np.isnan(lg_BF_vals), axis=2))
    # print('real_trials', real_trials)
    # Confidence intervals
    # ax = plt.gca()
    xs = n12s / n1
    for ind_M in range(len(Ms)):
        color = color_sequence[ind_M]
        zorder = len(Ms) - ind_M
        if real_trials > 1:
            plt.fill_between(xs, CIs[ind_M, :, 0], CIs[ind_M, :, 1],
                             alpha=alpha_shade, color=color, zorder=zorder)
            # plt.plot(n12s, np.log10(CIs[:, 0]), '-', color='g', alpha=alpha_shade)
            # plt.plot(n12s, np.log10(CIs[:, 1]), '-', color='g', alpha=alpha_shade)

        # Mean
        plt.plot(xs, median_lg_BFs[ind_M, :], color=color, label=f'M={Ms[ind_M]:d}', zorder=zorder)

    # Significance levels
    xlims = plt.xlim()
    plt.plot(xlims, [-1] * 2, '--', color='k', lw=lw_theory, zorder=0)
    plt.plot(xlims, [1] * 2, '--', color='k', lw=lw_theory, zorder=0)

    plt.xscale('log')
    plt.xlabel('$n_{12}/n_1$')
    plt.ylabel('Median $\mathrm{lg}(B)$')
    plt.title(
        f"trials={real_trials}, D1={D1:.2f}, D2={D2:.2f},n1={n1:.2f}, \nn2={n1:.2f}, dt={dt}, L={L}, rotation={rotation}")

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    fig_folder = 'figures'
    figname = f'link_dependence-weak'
    figpath = os.path.join(fig_folder, figname)
    plt.savefig(figpath + '.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(figpath + '.pdf', bbox_inches='tight', pad_inches=0)

    return lg_BF_vals


def plot_diffusivity_dependence(trials=20, D1_range=[0.01, 10], verbose=False,
                                recalculate_trajectory=False, recalculate_BF=False, dry_run=False,
                                cluster=False, rotation=True):
    """
    The function loads data for the specified parameters and plots the diffusivity dependence plot.

    If dry_run = True, the calculations are not performed, but instead an arguments file is created to be fed into cluster.
    Note that the means and the confidence intervals are calculated for lg(B), not for B.
    """
    # Neeed to specify parameters to be able to load the right files

    # %% Constants
    # trials = 20  # 1000
    D2 = 0.4  # um^2/s
    # D1 = 5 * D2   # um^2/s; 0.4
    n1 = 1
    n2 = 1
    n12 = 30
    # n12 = 10 * n2  # s^{-1}. Somehting interesting happens between [1e-9; 1e-6]
    Ms = [10, 100, 200]  # required number of points in a trajectory; 100
    # N = 101

    dt = 0.05  # s 0.3
    gamma = 1e-8  # viscous drag, in kg/s
    L = 20
    angle = 0
    # trial = 0   # the trial number
    # recalculate = False

    arguments_file = 'arguments.dat'
    # color = [0.1008,    0.4407,    0.7238]

    # Parameters varying in the plot

    # D1_ratio_range = np.array([1e-2, 1e2])
    # D1_range = [0.01, 10]
    D1_points = 50
    D1s = np.logspace(log10(D1_range[0]), log10(D1_range[1]), num=D1_points)

    # k1, k2, k12 = np.array([n1, n2, n12]) * gamma

    # if dry_run:
    #     if os.path.exists(arguments_file):
    #         os.unlink(arguments_file)
    with open(arguments_file, 'a') as file:

        lg_BF_vals = np.full([len(Ms), D1_points, trials], np.nan)

        for trial in trange(trials, desc='Loading/scheduling calculations'):
            # if seed is not None:
            #     trial_seed = seed + trial
            # else:
            #     trial_seed = None
            for ind_M, M in enumerate(Ms):
                for ind_D1, D1 in enumerate(D1s):
                    args_string = get_cluster_args_string(
                        D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, dt=dt, angle=angle, L=L,
                        trial=trial, M=M, verbose=verbose,
                        recalculate_trajectory=recalculate_trajectory,
                        recalculate_BF=recalculate_BF, rotation=rotation)
                    lg_BF_vals[
                        ind_M, ind_D1, trial], ln_evidence_with_link, ln_evidence_free, loaded, dict_data = simulate_and_calculate_Bayes_factor_terminal(
                        args_string, cluster=cluster)
                    if cluster and not loaded:
                        file.write(args_string)

        if cluster and verbose:
            print('Warning: verbose was active')
            # return np.nan

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
    fig = set_figure_size(num=4, rows=rows, page_width_frac=page_width_frac,
                          height_factor=height_factor)

    # Confidence intervals
    real_trials = np.min(np.sum(~np.isnan(lg_BF_vals), axis=2))
    # ax = plt.gca()
    xs = D1s / D2
    for ind_M in range(len(Ms)):
        color = color_sequence[ind_M]
        zorder = len(Ms) - ind_M
        if real_trials > 1:
            plt.fill_between(xs, CIs[ind_M, :, 0], CIs[ind_M, :, 1],
                             alpha=alpha_shade, color=color, zorder=zorder)
            # plt.plot(n12s, np.log10(CIs[:, 0]), '-', color='g', alpha=alpha_shade)
            # plt.plot(n12s, np.log10(CIs[:, 1]), '-', color='g', alpha=alpha_shade)

        # Mean
        plt.plot(xs, median_lg_BFs[ind_M, :], color=color, label=f'M={Ms[ind_M]:d}', zorder=zorder)

    # Significance levels
    xlims = plt.xlim()
    plt.plot(xlims, [-1] * 2, '--', color='k', lw=lw_theory, zorder=0)
    plt.plot(xlims, [1] * 2, '--', color='k', lw=lw_theory, zorder=0)

    plt.xscale('log')
    plt.xlabel('$D_1/D_2$')
    plt.ylabel('Median $\mathrm{lg}(B)$')
    plt.title(
        f'trials={real_trials}, D2={D2:.2f}, n1={n1:.1f},\nn2={n1:.1f}, n12={n12:.1f}, dt={dt}, L={L}, rotation={rotation}')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    fig_folder = 'figures'
    figname = f'diffusivity_dependence-weak'
    figpath = os.path.join(fig_folder, figname)
    plt.savefig(figpath + '.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(figpath + '.pdf', bbox_inches='tight', pad_inches=0)

    return lg_BF_vals


def plot_localization_dependence(trials=20, n_range=[1e-2, 100], particle=1, verbose=False,
                                 recalculate_trajectory=False, recalculate_BF=False, dry_run=False,
                                 cluster=False, rotation=True):
    """
    The function loads data for the specified parameters and makes the plot.

    Parameters:

    If dry_run = True, the calculations are not performed, but instead an arguments file is created to be fed into cluster.
    Note that the means and the confidence intervals are calculated for lg(B), not for B.

    particle - the particle, whose localization is varied
    """
    # Neeed to specify parameters to be able to load the right files

    # %% Constants
    D2 = 0.4  # um^2/s
    D1 = 5 * D2  # um^2/s; 0.4
    # n1 = 2e3
    # n12 = 10 * n2  # s^{-1}. Somehting interesting happens between [1e-9; 1e-6]
    Ms = [10, 100, 200]  # required number of points in a trajectory; 100
    # N = 101

    dt = 0.05  # s 0.3
    gamma = 1e-8  # viscous drag, in kg/s
    L = 20
    angle = 0
    # trial = 0   # the trial number
    # recalculate = False

    # n_range = np.array([1e-2, 100])
    n_other = 1
    n12 = 30
    n_points = 50
    if particle == 1:
        n2 = n_other
        # n_range = n_ratio_range * n_other
        n1s = np.logspace(log10(n_range[0]), log10(n_range[1]), num=n_points)
    elif particle == 2:
        n1 = n_other
        # n_range = n_ratio_range * n_other
        n2s = np.logspace(log10(n_range[0]), log10(n_range[1]), num=n_points)

    arguments_file = 'arguments.dat'

    with open(arguments_file, 'a') as file:

        lg_BF_vals = np.full([len(Ms), n_points, trials], np.nan)

        for trial in trange(trials, desc='Loading/scheduling calculations'):
            # if seed is not None:
            #     trial_seed = seed + trial
            # else:
            #     trial_seed = None
            for ind_M, M in enumerate(Ms):

                # OPTIMIZE:
                if particle == 1:
                    for ind_n, n1 in enumerate(n1s):
                        args_string = get_cluster_args_string(
                            D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, dt=dt, angle=angle,
                            L=L, trial=trial, M=M, verbose=verbose,
                            recalculate_trajectory=recalculate_trajectory,
                            recalculate_BF=recalculate_BF, rotation=rotation)
                        # if dry_run:
                        #     file.write(args_string)
                        #
                        # else:
                        #     # print(args_string)
                        #     lg_BF_vals[ind_M, ind_n, trial], ln_evidence_with_link, ln_evidence_free = simulate_and_calculate_Bayes_factor_terminal(
                        #         args_string)

                        lg_BF_vals[
                            ind_M, ind_n, trial], ln_evidence_with_link, ln_evidence_free, loaded, dict_data = simulate_and_calculate_Bayes_factor_terminal(
                            args_string, cluster=cluster)
                        if cluster and not loaded:
                            file.write(args_string)

                else:
                    for ind_n, n2 in enumerate(n2s):
                        args_string = get_cluster_args_string(
                            D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, dt=dt, angle=angle,
                            L=L, trial=trial, M=M, verbose=verbose, recalculate_trajectory=False,
                            recalculate_BF=False)
                        # if dry_run:
                        #     file.write(args_string)
                        #
                        # else:
                        #     # print(args_string)
                        #     lg_BF_vals[ind_M, ind_n, trial], ln_evidence_with_link, ln_evidence_free = simulate_and_calculate_Bayes_factor_terminal(
                        #         args_string)

                        lg_BF_vals[
                            ind_M, ind_n, trial], ln_evidence_with_link, ln_evidence_free, loaded, dict_data = simulate_and_calculate_Bayes_factor_terminal(
                            args_string)
                        if cluster and not loaded:
                            file.write(args_string)

            # time.sleep(3)

        if cluster and verbose:
            print('Warning: verbose was active')
            # return np.nan

    # %% Calculating means and CI
    real_trials = np.min(np.sum(~np.isnan(lg_BF_vals), axis=2))
    median_lg_BFs = np.nanmedian(lg_BF_vals, axis=2)

    if trials > 1:
        CIs = np.full([len(Ms), n_points, 2], np.nan)
        CIs[:, :, 0] = np.nanquantile(lg_BF_vals, (1 - confidence_level) / 2, axis=2)  # 0.025
        CIs[:, :, 1] = np.nanquantile(lg_BF_vals, 1 - (1 - confidence_level) / 2, axis=2)  # 0.975
        # print('CIs: ', np.log10(CIs))

    # %% Actual plotting
    fig = set_figure_size(num=5, rows=rows, page_width_frac=page_width_frac,
                          height_factor=height_factor)

    # Confidence intervals
    # ax = plt.gca()
    if particle == 1:
        xs = n1s / n12
    else:
        xs = n2s / n12
    for ind_M in range(len(Ms)):
        color = color_sequence[ind_M]
        zorder = len(Ms) - ind_M
        if trials > 1:
            plt.fill_between(xs, CIs[ind_M, :, 0], CIs[ind_M, :, 1],
                             alpha=alpha_shade, color=color, zorder=zorder)

        # Mean
        plt.plot(xs, median_lg_BFs[ind_M, :], color=color, label=f'M={Ms[ind_M]:d}', zorder=zorder)

    # Significance levels
    xlims = plt.xlim()
    plt.plot(xlims, [-1] * 2, '--', color='k', lw=lw_theory, zorder=0)
    plt.plot(xlims, [1] * 2, '--', color='k', lw=lw_theory, zorder=0)

    plt.xscale('log')
    plt.xlabel('$n_1/n_{12}$')  # '$n_1/n_2$' if particle == 1 else '$n_2/n_1$')
    plt.ylabel('Median $\mathrm{lg}(B)$')
    plt.title(
        f'trials={real_trials}, D1={D1:.2f}, D2={D2:.2f}, n12={n12:.1f}, \ndt={dt}, L={L}, rotation={rotation}')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    fig_folder = 'figures'
    figname = f'localization_dependence-weak'
    figpath = os.path.join(fig_folder, figname)
    plt.savefig(figpath + '.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(figpath + '.pdf', bbox_inches='tight', pad_inches=0)

    return lg_BF_vals


def plot_angle_dependence(trials=20, verbose=False, recalculate_trajectory=False,
                          recalculate_BF=False, dry_run=False, cluster=False):
    """
    The function loads data for the specified parameters and plots the diffusivity dependence plot.

    If dry_run = True, the calculations are not performed, but instead an arguments file is created to be fed into cluster.
    Note that the means and the confidence intervals are calculated for lg(B), not for B.
    """
    # Neeed to specify parameters to be able to load the right files

    # %% Constants
    # trials = 20  # 1000
    rotation = True
    D1 = 2  # um^2/s
    D2 = 0.4  # um^2/s
    # D1 = 5 * D2   # um^2/s; 0.4
    n1 = 1
    n2 = 1
    n12 = 30
    # n12 = 10 * n2  # s^{-1}. Somehting interesting happens between [1e-9; 1e-6]
    Ms = [10, 100, 200]  # required number of points in a trajectory; 100
    # Ms = [200]
    # N = 101

    dt = 0.05  # s 0.3
    gamma = 1e-8  # viscous drag, in kg/s
    L = 20
    # angle = 0
    # trial = 0   # the trial number
    # recalculate = False

    arguments_file = 'arguments.dat'
    # color = [0.1008,    0.4407,    0.7238]

    # Parameters varying in the plot

    # D1_ratio_range = np.array([1e-2, 1e2])
    # D1_range = [0.01, 10]
    points = 50
    alphas = np.linspace(-np.pi, np.pi, num=points)
    # a1 = np.pi / 6
    # alphas = np.linspace(a1, a1, num=points)

    # k1, k2, k12 = np.array([n1, n2, n12]) * gamma

    # if dry_run:
    #     if os.path.exists(arguments_file):
    #         os.unlink(arguments_file)
    with open(arguments_file, 'a') as file:

        lg_BF_vals = np.full([len(Ms), points, trials], np.nan)
        MLE_alphas = np.full([len(Ms), points, trials], np.nan)
        MLE_alphas_centered = np.full([len(Ms), points, trials], np.nan)

        for trial in trange(trials, desc='Loading/scheduling calculations'):
            # if seed is not None:
            #     trial_seed = seed + trial
            # else:
            #     trial_seed = None
            for ind_M, M in enumerate(Ms):
                for ind_alpha, alpha in enumerate(alphas):
                    args_string = get_cluster_args_string(
                        D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, dt=dt, angle=alpha, L=L,
                        trial=trial, M=M, verbose=verbose,
                        recalculate_trajectory=recalculate_trajectory,
                        recalculate_BF=recalculate_BF, rotation=rotation)
                    lg_BF_vals[
                        ind_M, ind_alpha, trial], ln_evidence_with_link, ln_evidence_free, loaded, dict_data = simulate_and_calculate_Bayes_factor_terminal(
                        args_string, cluster=cluster)

                    if 'MLE_link' in dict_data:
                        # print(dict_data['MLE_link'])
                        # raise RuntimeError()
                        MLE_alphas[ind_M, ind_alpha,
                                   trial] = dict_data['MLE_link']['alpha'] / np.pi * 180
                        MLE_alphas_centered[ind_M, ind_alpha,
                                            trial] = (dict_data['MLE_link'][
                                                          'alpha'] - alpha) / np.pi * 180

                    # raise RuntimeError('stop')
                    if cluster and not loaded:
                        file.write(args_string)

        if cluster and verbose:
            print('Warning: verbose was active')
            # return np.nan

    # print(lg_BF_vals)

    # %% Calculating means and CI
    real_trials = np.min(np.sum(~np.isnan(lg_BF_vals), axis=2))
    median_lg_BFs = np.nanmedian(lg_BF_vals, axis=2)
    # print('mean', median_lg_BFs)

    # BF_vals = 10**lg_BF_vals

    if trials > 1:
        CIs = np.full([len(Ms), points, 2], np.nan)
        CIs[:, :, 0] = np.nanquantile(lg_BF_vals, (1 - confidence_level) / 2, axis=2)  # 0.025
        CIs[:, :, 1] = np.nanquantile(lg_BF_vals, 1 - (1 - confidence_level) / 2, axis=2)  # 0.975
        # print('CIs: ', np.log10(CIs))

    # %% Plotting Bayes factors
    fig = set_figure_size(num=6, rows=rows, page_width_frac=page_width_frac,
                          height_factor=height_factor)

    # Confidence intervals
    # ax = plt.gca()
    xs = alphas / np.pi * 180
    for ind_M in range(len(Ms)):
        color = color_sequence[ind_M]
        zorder = len(Ms) - ind_M
        if trials > 1:
            plt.fill_between(xs, CIs[ind_M, :, 0], CIs[ind_M, :, 1],
                             alpha=alpha_shade, color=color, zorder=zorder)
            # plt.plot(n12s, np.log10(CIs[:, 0]), '-', color='g', alpha=alpha_shade)
            # plt.plot(n12s, np.log10(CIs[:, 1]), '-', color='g', alpha=alpha_shade)

        # Mean
        plt.plot(xs, median_lg_BFs[ind_M, :], color=color, label=f'M={Ms[ind_M]:d}', zorder=zorder)

    # Significance levels
    xlims = plt.xlim()
    plt.plot(xlims, [-1] * 2, '--', color='k', lw=lw_theory, zorder=0)
    plt.plot(xlims, [1] * 2, '--', color='k', lw=lw_theory, zorder=0)

    # plt.xscale('log')
    plt.xlabel('$\\alpha$, $^\circ$')
    plt.ylabel('Median $\mathrm{lg}(B)$')
    plt.title(
        f'trials={real_trials}, D1={D1:.2f}, D2={D2:.2f}, n1={n1:.1f},\nn2={n1:.1f}, n12={n12:.1f}, dt={dt}, L={L}, rotation={rotation}')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    fig_folder = 'figures'
    figname = f'alpha_dependence-weak'
    figpath = os.path.join(fig_folder, figname)
    plt.savefig(figpath + '.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(figpath + '.pdf', bbox_inches='tight', pad_inches=0)

    # %% Plotting alpha inference
    fig = set_figure_size(num=7, rows=rows, page_width_frac=page_width_frac,
                          height_factor=height_factor)

    # Confidence intervals
    median_alphas_centered = np.nanmedian(MLE_alphas_centered, axis=2)
    # print(median_alphas_centered)
    # print(MLE_alphas_centered)
    if real_trials > 1:
        CIs = np.full([len(Ms), points, 2], np.nan)
        CIs[:, :, 0] = np.nanquantile(
            MLE_alphas_centered, (1 - confidence_level) / 2, axis=2)  # 0.025
        CIs[:, :, 1] = np.nanquantile(MLE_alphas_centered, 1 - (1 - confidence_level) / 2, axis=2)
    # ax = plt.gca()

    for ind_M in range(len(Ms)):
        color = color_sequence[ind_M]
        zorder = 1 + ind_M
        if real_trials > 1:
            plt.fill_between(xs, CIs[ind_M, :, 0], CIs[ind_M, :, 1],
                             alpha=alpha_shade, color=color, zorder=zorder)
            # plt.plot(n12s, np.log10(CIs[:, 0]), '-', color='g', alpha=alpha_shade)
            # plt.plot(n12s, np.log10(CIs[:, 1]), '-', color='g', alpha=alpha_shade)

        # Mean
        plt.plot(xs, median_alphas_centered[ind_M, :],
                 color=color, label=f'M={Ms[ind_M]:d}', zorder=zorder)

    # Significance levels
    xlims = plt.xlim()
    # plt.plot(xlims, [-1] * 2, '--', color='k', lw=lw_theory, zorder=0)
    # plt.plot(xlims, [1] * 2, '--', color='k', lw=lw_theory, zorder=0)

    # plt.xscale('log')
    plt.xlabel('$\\alpha$, $^\circ$')
    plt.ylabel('Median ($\\hat\\alpha - \\alpha$), $^\circ$')
    plt.title(
        f'trials={real_trials}, D1={D1:.2f}, D2={D2:.2f}, n1={n1:.1f},\nn2={n1:.1f}, n12={n12:.1f}, dt={dt}, L={L}, rotation={rotation}')

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    fig_folder = 'figures'
    figname = f'angle_dependence-weak'
    figpath = os.path.join(fig_folder, figname)
    plt.savefig(figpath + '.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(figpath + '.pdf', bbox_inches='tight', pad_inches=0)

    return lg_BF_vals


# def plot_free_hookean_link_strength_dependence(trials=20, n12_range=[1e-1, 1e3], verbose=False,
#                                                recalculate_trajectory=False, recalculate_BF=False,
#                                                dry_run=False,
#                                                cluster=False, rotation=False, model=
#                                                'free_same_D'):
#     """
#     The function loads data for the specified parameters and plots the link strength dependence plot.
#
#     If dry_run = True, the calculations are not performed, but instead an arguments file is created to be fed into cluster.
#     Note that the means and the confidence intervals are calculated for lg(B), not for B.
#     """
#     # Neeed to specify parameters to be able to load the right files
#
#     # %% Constants
#     # trials = 50  # 20  # 50  # 1000
#     D2 = 0.4  # um^2/s
#     D1 = D2  # 5 * D2  # um^2/s; 0.4
#     n1 = 0
#     n2 = 0
#     # n12 = 10 * n2  # s^{-1}. Somehting interesting happens between [1e-9; 1e-6]
#     Ms = [10, 100, 200, 1000]  # required number of points in a trajectory; 100
#     # N = 101
#
#     dt = 0.05  # s 0.3
#     gamma = 1e-8  # viscous drag, in kg/s
#     L = 5
#     angle = 0
#     # trial = 0   # the trial number
#     # recalculate = False
#
#     arguments_file = 'arguments.dat'
#     # color = [0.1008,    0.4407,    0.7238]
#
#     # Parameters varying in the plot
#
#     # n12_range = [1e-1, 1e3]
#     n12_points = 50  # 50
#     n12s = np.logspace(log10(n12_range[0]), log10(n12_range[1]), num=n12_points)
#
#     # k1, k2, k12 = np.array([n1, n2, n12]) * gamma
#
#     # if cluster:
#     # if os.path.exists(arguments_file):
#     #     os.unlink(arguments_file)
#     with open(arguments_file, 'a') as file:
#
#         lg_BF_vals = np.full([len(Ms), n12_points, trials], np.nan)
#         for trial in trange(trials, desc='Loading/scheduling calculations'):
#
#             for ind_M, M in enumerate(Ms):
#                 for ind_n12, n12 in enumerate(n12s):
#                     args_string = get_cluster_args_string(
#                         D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, dt=dt, angle=angle, L=L,
#                         trial=trial, M=M, verbose=verbose, model=model,
#                         recalculate_trajectory=recalculate_trajectory,
#                         recalculate_BF=recalculate_BF, rotation=rotation)
#                     # print('Calculating with parameters: ', args_string)
#                     lg_BF_vals[
#                         ind_M, ind_n12, trial], ln_evidence_with_link, ln_evidence_free, \
#                     loaded, dict_data = simulate_and_calculate_Bayes_factor(
#                         D1=D1, D2=D2, n1=n1,
#                         n2=n2,
#                         n12=n12, dt=dt,
#                         angle=angle, L0=L0,
#                         trial=trial,
#                         M=M, model=model,
#                         recalculate_trajectory=recalculate_trajectory,
#                         recalculate_BF=recalculate_BF,
#                         verbose=verbose,
#                         cluster=cluster,
#                         rotation=rotation)
#
#                     # simulate_and_calculate_Bayes_factor_terminal(
#                     # args_string, cluster=cluster)
#
#                     if cluster and not loaded:
#                         file.write(args_string)
#                     # raise RuntimeError('stop')
#
#             # print('Iteration ', i)
#             # true_parameters = {name: val for name, val in zip(
#             #     ('D1 D2 n1 n2 n12 gamma T dt angle L trial M'.split()),
#             #     (D1, D2, n1, n2, n12, gamma, T, dt, angle, L, trial, M))}
#             #
#             # t, R, dR, hash = simulate_2_confined_particles_with_fixed_angle_bond(
#             #     true_parameters=true_parameters, plot=False, save_figure=False, recalculate=recalculate, seed=trial_seed)
#
#             # Load the Bayes factor
#
#             # calculate_bayes_factor(
#             #     t=t, dR=dR, true_parameters=true_parameters, hash=hash, recalculate=recalculate,  plot=False, verbose=verbose)
#
#     if cluster and verbose:
#         print('Warning: verbose was active')
#         # return np.nan
#
#     # print(lg_BF_vals)
#
#     # %% Calculating means and CI
#     median_lg_BFs = np.nanmedian(lg_BF_vals, axis=2)
#     # print('mean', median_lg_BFs)
#
#     # BF_vals = 10**lg_BF_vals
#
#     if trials > 1:
#         CIs = np.full([len(Ms), n12_points, 2], np.nan)
#         CIs[:, :, 0] = np.nanquantile(lg_BF_vals, (1 - confidence_level) / 2, axis=2)  # 0.025
#         CIs[:, :, 1] = np.nanquantile(lg_BF_vals, 1 - (1 - confidence_level) / 2, axis=2)  # 0.975
#         # print('CIs: ', np.log10(CIs))
#
#     # %% Actual plotting
#     # fig = plt.figure(num=3, clear=True)
#     fig = set_figure_size(num=3, rows=rows, page_width_frac=page_width_frac,
#                           height_factor=height_factor)
#
#     real_trials = np.min(np.sum(~np.isnan(lg_BF_vals), axis=2))
#     # print('real_trials', real_trials)
#     # Confidence intervals
#     # ax = plt.gca()
#     xs = n12s
#     for ind_M in range(len(Ms)):
#         color = color_sequence[ind_M]
#         zorder = len(Ms) - ind_M
#         if real_trials > 1:
#             plt.fill_between(xs, CIs[ind_M, :, 0], CIs[ind_M, :, 1],
#                              alpha=alpha_shade, color=color, zorder=zorder)
#             # plt.plot(n12s, np.log10(CIs[:, 0]), '-', color='g', alpha=alpha_shade)
#             # plt.plot(n12s, np.log10(CIs[:, 1]), '-', color='g', alpha=alpha_shade)
#
#         # Mean
#         plt.plot(xs, median_lg_BFs[ind_M, :], color=color, label=f'M={Ms[ind_M]:d}', zorder=zorder)
#
#     # Significance levels
#     xlims = plt.xlim()
#     plt.plot(xlims, [-1] * 2, '--', color='k', lw=lw_theory, zorder=0)
#     plt.plot(xlims, [1] * 2, '--', color='k', lw=lw_theory, zorder=0)
#
#     plt.xscale('log')
#     plt.xlabel('$n_{12}$')
#     plt.ylabel('Median $\mathrm{lg}(B)$')
#     plt.title(
#         f'trials={real_trials}, D1={D1:.2f}, D2={D2:.2f}, n1={n1:.2f}, \nn2={n1:.2f}, dt={dt}, '
#         f'L={L}, rotation={rotation}')
#
#     plt.legend(loc='upper left')
#     plt.tight_layout()
#     plt.show()
#
#     fig_folder = 'figures'
#     figname = f'link_dependence-weak'
#     figpath = os.path.join(fig_folder, figname)
#     plt.savefig(figpath + '.png', bbox_inches='tight', pad_inches=0)
#     plt.savefig(figpath + '.pdf', bbox_inches='tight', pad_inches=0)
#
#     return lg_BF_vals


def contour_plot_free_dumbbell_same_D(trials=3, verbose=False,
                                      recalculate_trajectory=False,
                                      cluster=False, Ms=[100], l0_range=[1e-1, 1e3],
                                      eta_range=[1e-2, 1e2]):
    """
    Plot a contour plot in 2 dimensionless parameters of the problem.
    :param trials:
    :type trials:
    :param verbose:
    :type verbose:
    :param recalculate_trajectory:
    :type recalculate_trajectory:
    :param recalculate_BF:
    :type recalculate_BF:
    :param dry_run:
    :type dry_run:
    :param cluster:
    :type cluster:
    :param rotation:
    :type rotation:
    :param model:
    :type model:
    :return:
    :rtype:
    """
    # %% Constants
    # trials = 50  # 20  # 50  # 1000
    D1 = D2 = 0.4  # um^2/s
    n1 = 0
    n2 = 0
    # n12 = 10 * n2  # s^{-1}. Somehting interesting happens between [1e-9; 1e-6]
    # Ms = [100]  # , 100, 200, 1000]  # required number of points in a trajectory; 100
    # N = 101

    dt = 0.05  # s 0.3
    # L0 = 5
    model = 'free_same_D'
    # trial = 0   # the trial number
    # recalculate = False

    ## The 3 dimensionless parameters are: eta = n12* dt, l = L0/sqrt(D1 * dt), M

    arguments_file = 'arguments.dat'
    # color = [0.1008,    0.4407,    0.7238]

    mesh_resolution = 2 ** 4 + 1

    # l_range = [1e-1, 1e3]
    l0s = np.logspace(log10(l0_range[0]), log10(l0_range[1]), num=mesh_resolution)
    # L0s = ls * np.sqrt(D1 * dt)

    # eta_range = [1e-2, 1e2]
    etas = np.logspace(log10(eta_range[0]), log10(eta_range[1]), num=mesh_resolution)
    # n12s = etas / dt
    cluster_counter = 0

    with open(arguments_file, 'a') as file:

        lg_BF_vals = np.full([len(Ms), mesh_resolution, mesh_resolution, trials], np.nan)
        for trial in trange(trials, desc='Loading/scheduling calculations'):

            for ind_M, M in enumerate(Ms):
                for ind_eta, eta in enumerate(etas):
                    n12 = eta / dt
                    for ind_l0, l0 in enumerate(l0s):
                        L0 = l0 * np.sqrt(D1 * dt)
                        args_string = get_cluster_args_string(
                            D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, dt=dt, L=L0,
                            trial=trial, M=M, verbose=verbose, model=model,
                            recalculate_trajectory=recalculate_trajectory)
                        # print('Calculating with parameters: ', args_string)
                        # with stopwatch():
                        lg_BF_vals[
                            ind_M, ind_eta, ind_l0, trial], ln_evidence_with_link, \
                        ln_evidence_free, loaded, dict_data = simulate_and_calculate_Bayes_factor(
                            D1=D1, D2=D2, n1=n1,
                            n2=n2,
                            n12=n12, dt=dt,
                            angle=0, L0=L0,
                            trial=trial,
                            M=M, model=model,
                            recalculate_trajectory=recalculate_trajectory,
                            recalculate_BF=False,
                            verbose=verbose,
                            cluster=cluster,
                            rotation=True)

                        # simulate_and_calculate_Bayes_factor_terminal(
                        # args_string, cluster=cluster)

                        if cluster and not loaded:
                            file.write(args_string)
                            cluster_counter += 1
                        # sys.exit(0)

    if cluster and verbose:
        print('Warning: verbose was active')
    if cluster:
        # Reset start position on cluster
        position_file = 'position.dat'
        with open(position_file, 'w') as fp_position:
            fp_position.write('{0:d}'.format(0))
    print(f'{cluster_counter} calculations scheduled for the cluster')

    # %% Calculating means and CI
    median_lg_BFs = np.nanmedian(lg_BF_vals, axis=3)

    if trials > 1:
        CIs = np.full([len(Ms), mesh_resolution, mesh_resolution, 2], np.nan)
        CIs[:, :, :, 0] = np.nanquantile(lg_BF_vals, (1 - confidence_level) / 2, axis=3)  # 0.025
        CIs[:, :, :, 1] = np.nanquantile(lg_BF_vals, 1 - (1 - confidence_level) / 2,
                                         axis=3)  # 0.975

    # %% Actual plotting
    fig = set_figure_size(num=3, rows=rows, page_width_frac=page_width_frac,
                          height_factor=height_factor)

    real_trials = np.min(np.sum(~np.isnan(lg_BF_vals), axis=3))

    X, Y = np.meshgrid(etas, l0s)
    # xs = n12s
    # for ind_M in range(len(Ms)):
    #
    #
    #     if real_trials > 1:
    #         plt.fill_between(xs, CIs[ind_M, :, 0], CIs[ind_M, :, 1],
    #                          alpha=alpha_shade, color=color, zorder=zorder)
    #         # plt.plot(n12s, np.log10(CIs[:, 0]), '-', color='g', alpha=alpha_shade)
    #         # plt.plot(n12s, np.log10(CIs[:, 1]), '-', color='g', alpha=alpha_shade)

    # Mean
    ind_M = 0
    color = color_sequence[ind_M]
    zorder = len(Ms) - ind_M
    # plt.plot(xs, median_lg_BFs[ind_M, :], color=color, label=f'M={Ms[ind_M]:d}', zorder=zorder)
    plt.contourf(X, Y, median_lg_BFs[ind_M, :, :], # label=f'M={Ms[ind_M]:d}',
                 zorder=zorder, cmap='inferno', )
                 # levels = range(-10,10))
    cb = plt.colorbar()
    cb.set_label('Median $\mathrm{lg}B$')

    # plot special levels
    lgB_levels = [-1, 1]
    c_major = plt.contour(X, Y, median_lg_BFs[ind_M, :, :], levels=lgB_levels, colors='k',
                          linewidths=1)

    # # Significance levels
    # xlims = plt.xlim()
    # plt.plot(xlims, [-1] * 2, '--', color='k', lw=lw_theory, zorder=0)
    # plt.plot(xlims, [1] * 2, '--', color='k', lw=lw_theory, zorder=0)
    #
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\eta\equiv n_{12}\Delta t$')
    plt.ylabel(r'$\ell_0\equiv L_0/\sqrt{D\Delta t}$')

    # plt.ylabel('Median $\mathrm{lg}(B)$')

    plt.title(
        f'trials={real_trials}, M={Ms[ind_M]:d}, D1={D1:.2f},\nD2={D2:.2f}, dt={dt}')

    # plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

    fig_folder = 'figures'
    figname = f'free_hookean_M={Ms[ind_M]:d}'
    figpath = os.path.join(fig_folder, figname)
    plt.savefig(figpath + '.png', bbox_inches='tight', pad_inches=0)
    plt.savefig(figpath + '.pdf', bbox_inches='tight', pad_inches=0)

    return lg_BF_vals
