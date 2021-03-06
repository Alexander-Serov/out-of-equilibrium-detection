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
                       simulate_and_calculate_Bayes_factor_terminal,
                       simulate_and_calculate_Bayes_factor)
from constants_main import color_sequence
from simulate import simulate_2_confined_particles_with_fixed_angle_bond
from support import get_cluster_args_string, set_figure_size, delete_data
from stopwatch import stopwatch
import logging

# Plot parameters
alpha_shade = 0.25
confidence_level = 0.95
height_factor = 0.8
page_width_frac = 0.5  # 0.5
rows = 1
lw_theory = 1
fig_folder = 'figures'
arguments_file = 'arguments.dat'
mesh_resolution = 2 ** 3 + 1

# Default values
eta_default = 0.05
eta12_default = 1.5
gamma_default = 5
M_default = 1000
dt = 0.05  # s
D2 = 0.1  # um^2/s
L0 = 20


def plot_1d(xs, ys, CIs,
            xlabel=None, xscale='linear',
            ylabel=None, title=None, fig_num=1,
            figname=None, labels=None,
            y_levels=[-1, 1],
            legend_loc='best',
            clear=True,
            style=None,
            ):
    fig = set_figure_size(num=fig_num, rows=rows, page_width_frac=page_width_frac,
                          height_factor=height_factor, clear=clear)

    # if len(ys.shape)>2:
    #     multiple_curves = True
    # else:
    #     ys = [ys]
    #     CIs = [CIs]

    # for i_curve in len(ys.shape):

    len_Ms = np.shape(ys)[0]
    for ind_M in range(len_Ms):
        color = color_sequence[ind_M]
        zorder = len_Ms - ind_M
        # Confidence intervals
        if CIs is not None:
            plt.fill_between(xs, CIs[ind_M, :, 0], CIs[ind_M, :, 1],
                             alpha=alpha_shade, color=color, zorder=zorder)
        # Mean
        if style is None:
            style = '-'
        if labels:
            plt.plot(xs, ys[ind_M, :], style, color=color, label=f'M={labels[ind_M]:d}', \
                     zorder=zorder)
        else:
            plt.plot(xs, ys[ind_M, :], style, color=color, zorder=zorder)

    # Significance levels
    if y_levels:
        xlims = plt.xlim()
        for lvl in y_levels:
            plt.plot(xlims, [lvl] * 2, '--', color='k', lw=lw_theory, zorder=0)
            plt.plot(xlims, [lvl] * 2, '--', color='k', lw=lw_theory, zorder=0)

    if xscale == 'log':
        plt.xscale('log')

    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    if labels:
        plt.legend(loc=legend_loc)
    plt.tight_layout()

    if figname is not None:
        figpath = os.path.join(fig_folder, figname)
        plt.savefig(figpath + '.png', bbox_inches='tight', pad_inches=0)
        plt.savefig(figpath + '.pdf', bbox_inches='tight', pad_inches=0)
        plt.show()


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


def contour_plot(X, Y, Z, Z_lgB=None, fig_num=1, clims=None,
                 levels=None, cb_label=None, xscale='log', yscale='log',
                 xlabel=None, ylabel=None, lgB_levels=None,
                 title=None,
                 figname=None,
                 clear=True,
                 cmap='bwr',
                 ):
    set_figure_size(num=fig_num, rows=rows, page_width_frac=page_width_frac,
                    height_factor=height_factor, clear=clear)
    # print(clims)

    if clims is not None:
        im = plt.contourf(X, Y, Z, cmap=cmap, vmin=clims[0], vmax=clims[1], levels=levels)
    else:
        im = plt.contourf(X, Y, Z, cmap=cmap, levels=levels)

    if not np.all(np.isnan(Z)):
        cb = plt.colorbar()
        # todo: temporarily disabled the colorbar label
        # cb.set_label(cb_label)

    # Plot special levels
    if lgB_levels is not None:
        if Z_lgB is None:
            Z_lgB = Z
        plt.contour(X, Y, Z_lgB, levels=lgB_levels, colors='k', linewidths=1,
                    linestyles=['-', '--', '-'])

    # Labels
    if xscale is 'log':
        plt.xscale('log')
    if yscale is 'log':
        plt.yscale('log')
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # Save
    if figname and len(figname):
        try:
            plt.tight_layout()
            figpath = os.path.join(fig_folder, figname)
            plt.savefig(figpath + '.png', bbox_inches='tight', pad_inches=0)
            plt.savefig(figpath + '.pdf', bbox_inches='tight', pad_inches=0)
        except AttributeError as e:
            logging.warning('Unable to save figure.\n')  # str(e))

    plt.show()


def calculate_and_plot_contour_plot(
        args_dict,
        x_update_func,
        y_update_func,
        trials=3,
        Ms=(10,),
        x_step=None,
        mesh_resolution_x=2 ** 3 + 1,
        mesh_resolution_y=2 ** 3 + 1,
        xlabel='x',
        ylabel='y',
        xscale='log',
        yscale='log',
        title=None,
        x_range=(1e-1, 1e1),
        y_range=(1e-1, 1e1),
        cluster=False,
        verbose=False,
        figname_base=str(),
        put_M_in_title=True,
        plot_simulation_time=False,
):
    if len(x_range) == 2:
        Xs = np.logspace(log10(x_range[0]), log10(x_range[1]), num=mesh_resolution_x)
    elif len(x_range) == 3:
        Xs = np.arange(x_range[0], x_range[1] + x_range[2], x_range[2])
    else:
        Xs = x_range
    # print(Xs)

    if len(y_range) == 2:
        Ys = np.logspace(log10(y_range[0]), log10(y_range[1]), num=mesh_resolution_y)
    elif len(y_range) == 3:
        Ys = np.arange(y_range[0], y_range[1] + y_range[2], y_range[2])
    else:
        Ys = y_range

    cluster_counter = 0
    with open(arguments_file, 'a') as file:

        lg_BF_vals = np.full([len(Ms), len(Xs), len(Ys), trials], np.nan)
        simulation_time = np.empty_like(lg_BF_vals)
        full_time = np.empty_like(lg_BF_vals)

        for trial in trange(trials, desc='Loading/scheduling calculations'):
            args_dict.update({'trial': trial})

            for ind_M, M in enumerate(Ms):
                args_dict.update({'M': M})

                for ind_y, y in enumerate(Ys):
                    y_update_func(args_dict, y)

                    for ind_x, x in enumerate(Xs):
                        x_update_func(args_dict, x)

                        lg_BF_vals[ind_M, ind_x, ind_y, trial], ln_evidence_with_link, \
                        ln_evidence_free, loaded, _hash, simulation_time[
                            ind_M, ind_x, ind_y, trial], traj = \
                            simulate_and_calculate_Bayes_factor(**args_dict)

                        times = {'simulation_time': traj.simulation_time,
                                 'calculation_time_link': traj.calculation_time_link,
                                 'calculation_time_no_link': traj.calculation_time_no_link}
                        full_time[ind_M, ind_x, ind_y, trial] = np.sum(list(times.values()))

                        if cluster and not loaded:
                            file.write(get_cluster_args_string(**args_dict))
                            cluster_counter += 1

    if cluster and verbose:
        print('Warning: verbose was active')
    if cluster:
        # Reset start position on cluster
        position_file = 'position.dat'
        with open(position_file, 'w') as fp_position:
            fp_position.write('{0:d}'.format(0))

    # %% Calculating means and CI over trials
    median_lg_BFs = np.nanmedian(lg_BF_vals, axis=3)
    median_simulation_time_hours = np.nanmedian(simulation_time, axis=3)
    avg_time = np.nanmean(full_time)
    count = np.sum(np.logical_not(np.isnan(full_time)))
    sum_hours = np.nansum(full_time) / 3600 / count * np.prod(full_time.shape)
    around = 'around ' if count < np.prod(full_time.shape) else ''

    if not np.isnan(avg_time):
        print(f'On average, it took {avg_time / 60:.1f} min to calculate each of the {count:d} '
              f'recorded points.')
        print(f'Calculation of the whole plot takes {around}{sum_hours:.1f} hours of CPU time or '
              f'{around}{sum_hours/2000:.1f} hours on 2000 CPUs.')
    print(f'{cluster_counter} calculations scheduled for the cluster')

    if trials > 1:
        CIs = np.full([len(Ms), len(Xs), len(Ys), 2], np.nan)
        CIs[:, :, :, 0] = np.nanquantile(lg_BF_vals, (1 - confidence_level) / 2, axis=3)  # 0.025
        CIs[:, :, :, 1] = np.nanquantile(lg_BF_vals, 1 - (1 - confidence_level) / 2,
                                         axis=3)  # 0.975
        CI_widths = CIs[:, :, :, 1] - CIs[:, :, :, 0]
        min_CI_width = np.floor(np.nanmin(CI_widths))
        max_CI_width = np.ceil(np.nanmax(CI_widths))

    # %% Actual plotting
    for ind_M, M in enumerate(Ms):
        real_trials = np.min(np.sum(~np.isnan(lg_BF_vals), axis=3), axis=(1, 2))
        max_real_trials = np.max(np.sum(~np.isnan(lg_BF_vals), axis=3), axis=(1, 2))

        Xs_plot, Ys_plot = np.meshgrid(Xs, Ys)
        full_title = f'trials={real_trials[ind_M]} ({max_real_trials[ind_M]:d}), '
        if put_M_in_title:
            full_title += f'M={Ms[ind_M]:d}, '
        full_title += title
        lgB_levels = [-1, 0, 1]

        # LgB values
        clims = np.array([-1, 1]) * np.ceil(np.abs(np.nanmax(median_lg_BFs)))
        if np.all(np.isnan(clims)):
            levels = None
        else:
            levels = np.arange(clims[0], clims[1] + 1, 1)

        if put_M_in_title:
            figname = figname_base + f'_M={Ms[ind_M]:d}'
        else:
            figname = figname_base
        contour_plot(fig_num=3 + 3 * ind_M, X=Xs_plot, Y=Ys_plot,
                     Z=median_lg_BFs[ind_M, :, :].T,
                     clims=clims,
                     levels=levels,
                     lgB_levels=lgB_levels,
                     cb_label='Median $\mathrm{lg}B$',
                     xscale=xscale, yscale=yscale,
                     xlabel=xlabel, ylabel=ylabel,
                     title='Bayes factor\n' + full_title,
                     figname=figname,
                     cmap='bwr'
                     )

        # Confidence intervals
        if np.all(np.isnan([min_CI_width, max_CI_width])):
            levels = None
        else:
            levels = np.arange(min_CI_width, max_CI_width + 1, 1)

        contour_plot(fig_num=3 + 1 + 3 * ind_M, X=Xs_plot, Y=Ys_plot,
                     Z=CI_widths[ind_M, :, :].T,
                     levels=levels,
                     cb_label='$\mathrm{lg}B$, 95 \% CI width',
                     xscale=xscale, yscale=yscale,
                     xlabel=xlabel, ylabel=ylabel,
                     title='Confidence intervals\n' + full_title,
                     figname=figname + '_CI',
                     cmap='Reds'
                     )

        # Simulation time
        if plot_simulation_time:
            contour_plot(fig_num=3 + 2 + 3 * ind_M, X=Xs_plot, Y=Ys_plot,
                         Z=median_simulation_time_hours[ind_M, :, :].T,
                         Z_lgB=median_lg_BFs[ind_M, :, :].T,
                         lgB_levels=lgB_levels,
                         cb_label='Simulation time, secs',
                         xscale=xscale, yscale=yscale,
                         xlabel=xlabel, ylabel=ylabel,
                         title='Simulation time\n' + full_title,
                         figname=None,
                         cmap='bwr'
                         )

    return lg_BF_vals


def contour_plot_localized_eta12_v_eta(
        trials=3,
        verbose=False,
        recalculate_trajectory=False,
        cluster=False,
        Ms=(100,),
        eta12_range=(1e-1, 1e1),
        eta_range=(1e-1, 1e1),
        dt=0.05,
        angle=0,
        var='eta1',
):
    D1 = D2 * gamma_default  # um^2/s
    recalculate_BF = False
    model = 'localized_different_D_detect_angle'

    args_dict = {'D1': D1, 'D2': D2,
                 'dt': dt,
                 'angle': angle, 'L0': L0,
                 'verbose': verbose,
                 'recalculate_trajectory': recalculate_trajectory,
                 'recalculate_BF': recalculate_BF,
                 'rotation': True,
                 'cluster': cluster,
                 'model': model}

    # args_dict = args_dict_base.copy()

    def update_y(args_dict, y):
        # Y = eta12 = n12 * dt
        args_dict.update({'n12': y / dt})

    if var == 'eta1':
        # Change eta1
        eta2 = eta_default
        args_dict.update({'n2': eta2 / dt})

        def update_x(args_dict, x):
            # X = eta = n * dt
            args_dict.update({'n1': x / dt})

        def update_y(args_dict, y):
            # Y = eta12 = n12 * dt
            args_dict.update({'n12': y / dt})

        xlabel = r'$\eta_1$'
        title = f'D1={D1:.2f},\nD2={D2:.2f}, dt={dt}, L0={L0:.2f}, eta2={eta2:.2f}'
        figname_base = 'localized_eta12_v_eta1'

    elif var == 'eta2':
        # Change eta1
        eta1 = eta_default
        args_dict.update({'n1': eta1 / dt})

        def update_x(args_dict, x):
            # X = eta = n * dt
            args_dict.update({'n2': x / dt})

        xlabel = r'$\eta_2$'
        title = f'D1={D1:.2f},\nD2={D2:.2f}, dt={dt}, L0={L0:.2f}, eta1={eta1:.2f}'
        figname_base = 'localized_eta12_v_eta2'
    else:
        raise RuntimeError('var should be one of ["eta1", "eta2"]')

    calculate_and_plot_contour_plot(
        args_dict,
        x_update_func=update_x,
        y_update_func=update_y,
        trials=trials,
        Ms=Ms,
        mesh_resolution_x=mesh_resolution,
        mesh_resolution_y=mesh_resolution,
        xlabel=xlabel,
        ylabel=r'$\eta_{12}$',
        title=title,
        x_range=eta_range,
        y_range=eta12_range,
        cluster=cluster,
        verbose=verbose,
        figname_base=figname_base,
    )


def contour_plot_localized_eta12_v_eta1_eta2(
        trials=3,
        verbose=False,
        recalculate_trajectory=False,
        cluster=False,
        Ms=(100,),
        eta12_range=(1e-1, 1e1),
        eta1_range=(1e-1, 1e1),
        dt=0.05,
        angle=0,
):
    D1 = D2 * gamma_default  # um^2/s
    recalculate_BF = False
    model = 'localized_different_D_detect_angle'

    args_dict = {'D1': D1, 'D2': D2,
                 'dt': dt,
                 'angle': angle, 'L0': L0,
                 'verbose': verbose,
                 'recalculate_trajectory': recalculate_trajectory,
                 'recalculate_BF': recalculate_BF,
                 'rotation': True,
                 'cluster': cluster,
                 'model': model}

    def update_x(args_dict, x):
        # X = eta1 = eta2 = n1 * dt = n2 * dt
        n = x / dt
        args_dict.update({'n1': n, 'n2': n})

    def update_y(args_dict, y):
        # Y = eta12 = n12 * dt
        args_dict.update({'n12': y / dt})

    calculate_and_plot_contour_plot(
        args_dict,
        x_update_func=update_x,
        y_update_func=update_y,
        trials=trials,
        Ms=Ms,
        mesh_resolution_x=mesh_resolution,
        mesh_resolution_y=mesh_resolution,
        xlabel=r'$\eta_1 = \eta_2$',
        ylabel=r'$\eta_{12}$',
        title=f'D1={D1:.2f},\nD2={D2:.2f}, dt={dt}, L0={L0:.2f}',
        x_range=eta1_range,
        y_range=eta12_range,
        cluster=cluster,
        verbose=verbose,
        figname_base='localized_eta12_v_eta1=eta2',
    )


def contour_plot_localized_eta12_v_gamma(
        trials=3,
        verbose=False,
        recalculate_trajectory=False,
        cluster=False,
        Ms=(100,),
        eta12_range=(1e-1, 1e1),
        gamma_range=(1e-1, 1e1),
        dt=0.05,
        angle=0,
):
    n1 = eta_default / dt
    n2 = eta_default / dt
    recalculate_BF = False
    model = 'localized_different_D_detect_angle'

    args_dict = {'n1': n1, 'n2': n2,
                 'D2': D2,
                 'dt': dt,
                 'angle': angle, 'L0': L0,
                 'verbose': verbose,
                 'recalculate_trajectory': recalculate_trajectory,
                 'recalculate_BF': recalculate_BF,
                 'rotation': True,
                 'cluster': cluster,
                 'model': model}

    def update_x(args_dict, x):
        # x = gamma = D1/D2
        D1 = D2 * x
        args_dict.update({'D1': D1})

    def update_y(args_dict, y):
        # Y = eta12 = n12 * dt
        args_dict.update({'n12': y / dt})

    xlabel = r'$\gamma \equiv D_1/D_2$'
    ylabel = r'$\eta_{12}$'
    title = f'n1={n1:.2f}, n2={n2:.2f}, D2={D2:.2f},\ndt={dt}, L0={L0:.2f}'

    calculate_and_plot_contour_plot(
        args_dict,
        x_update_func=update_x,
        y_update_func=update_y,
        trials=trials,
        Ms=Ms,
        mesh_resolution_x=mesh_resolution,
        mesh_resolution_y=mesh_resolution,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        x_range=gamma_range,
        y_range=eta12_range,
        cluster=cluster,
        verbose=verbose,
        figname_base='localized_eta12_v_gamma',
    )

    # Flip the same plot
    calculate_and_plot_contour_plot(
        args_dict,
        x_update_func=update_y,
        y_update_func=update_x,
        trials=trials,
        Ms=Ms,
        mesh_resolution_x=mesh_resolution,
        mesh_resolution_y=mesh_resolution,
        xlabel=ylabel,
        ylabel=xlabel,
        title=title,
        x_range=eta12_range,
        y_range=gamma_range,
        cluster=cluster,
        verbose=verbose,
        figname_base='localized_gamma_v_eta12',
    )


def contour_plot_localized_eta12_v_eta_ratio(
        trials=3,
        verbose=False,
        recalculate_trajectory=False,
        cluster=False,
        Ms=(100,),
        eta12_range=(1e-1, 1e1),
        eta_ratio_range=(1e-1, 1e1),
        dt=0.05,
        angle=0,
):
    D1 = gamma_default * D2
    n2 = eta_default / dt
    recalculate_BF = False
    model = 'localized_different_D_detect_angle'

    args_dict = {'D1': D1, 'D2': D2,
                 'n2': n2,
                 'dt': dt,
                 'angle': angle, 'L0': L0,
                 'verbose': verbose,
                 'recalculate_trajectory': recalculate_trajectory,
                 'recalculate_BF': recalculate_BF,
                 'rotation': True,
                 'cluster': cluster,
                 'model': model}

    def update_x(args_dict, x):
        # x = eta1 / eta2 = n1 / n2
        n1 = x * n2
        args_dict.update({'n1': n1})

    def update_y(args_dict, y):
        # y = eta12 = n12 * dt
        args_dict.update({'n12': y / dt})

    calculate_and_plot_contour_plot(
        args_dict,
        x_update_func=update_x,
        y_update_func=update_y,
        trials=trials,
        Ms=Ms,
        mesh_resolution_x=mesh_resolution,
        mesh_resolution_y=mesh_resolution,
        xlabel=r'$\eta_1 / \eta_2 \equiv n_1 / n_2$',
        ylabel=r'$\eta_{12}$',
        title=f'D1={D1:.2f}, n2={n2:.2f},\nD2={D2:.2f}, dt={dt}, L0={L0:.2f}',
        x_range=eta_ratio_range,
        y_range=eta12_range,
        cluster=cluster,
        verbose=verbose,
        figname_base='localized_eta12_v_eta_ratio',
    )


def contour_plot_localized_gamma_v_eta(
        trials=3,
        verbose=False,
        recalculate_trajectory=False,
        cluster=False,
        Ms=(100,),
        gamma_range=(1e-1, 1e1),
        eta_range=(1e-1, 1e1),
        dt=0.05,
        angle=0,
        var='eta1'
):
    n12 = eta12_default / dt
    recalculate_BF = False
    model = 'localized_different_D_detect_angle'
    args_dict = {'D2': D2,
                 'n12': n12,
                 'dt': dt,
                 'angle': angle, 'L0': L0,
                 'verbose': verbose,
                 'recalculate_trajectory': recalculate_trajectory,
                 'recalculate_BF': recalculate_BF,
                 'rotation': True,
                 'cluster': cluster,
                 'model': model}

    def update_y(args_dict, y):
        # y = gamma = D1 / D2
        D1 = D2 * y
        args_dict.update({'D1': D1})

    if var == 'eta1':
        # Change eta1
        eta2 = eta_default
        args_dict.update({'n2': eta2 / dt})

        def update_x(args_dict, x):
            args_dict.update({'n1': x / dt})

        xlabel = r'$\eta_1$'
        title = f'D2={D2:.2f},\nn12={n12:.2f}, dt={dt}, L0={L0:.2f}, eta2={eta2:.2f}'
        figname_base = 'localized_gamma_v_eta1'
    elif var == 'eta2':
        # Change eta2
        eta1 = eta_default
        args_dict.update({'n1': eta1 / dt})

        def update_x(args_dict, x):
            args_dict.update({'n2': x / dt})

        xlabel = r'$\eta_2$'
        title = f'D2={D2:.2f},\nn12={n12:.2f}, dt={dt}, L0={L0:.2f}, eta1={eta1:.2f}'
        figname_base = 'localized_gamma_v_eta2'

    else:
        raise RuntimeError('var should be one of ["eta1", "eta2"]')

    calculate_and_plot_contour_plot(
        args_dict,
        x_update_func=update_x,
        y_update_func=update_y,
        trials=trials,
        Ms=Ms,
        mesh_resolution_x=mesh_resolution,
        mesh_resolution_y=mesh_resolution,
        xlabel=xlabel,
        ylabel=r'$\gamma \equiv D_1/D_2$',
        title=title,
        x_range=eta_range,
        y_range=gamma_range,
        cluster=cluster,
        verbose=verbose,
        figname_base=figname_base,
    )


def contour_plot_localized_gamma_v_eta1_eta2(
        trials=3,
        verbose=False,
        recalculate_trajectory=False,
        cluster=False,
        Ms=(100,),
        gamma_range=(1e-1, 1e1),
        eta_range=(1e-1, 1e1),
        dt=0.05,
        angle=0,
):
    n12 = eta12_default / dt
    recalculate_BF = False
    model = 'localized_different_D_detect_angle'

    args_dict = {'D2': D2,
                 'n12': n12,
                 'dt': dt,
                 'angle': angle, 'L0': L0,
                 'verbose': verbose,
                 'recalculate_trajectory': recalculate_trajectory,
                 'recalculate_BF': recalculate_BF,
                 'rotation': True,
                 'cluster': cluster,
                 'model': model}

    def update_x(args_dict, x):
        # x = eta1 = eta2 = n1 * dt = n2 * dt
        n = x / dt
        args_dict.update({'n1': n, 'n2': n})

    def update_y(args_dict, y):
        # y = gamma = D1 / D2
        D1 = D2 * y
        args_dict.update({'D1': D1})

    calculate_and_plot_contour_plot(
        args_dict,
        x_update_func=update_x,
        y_update_func=update_y,
        trials=trials,
        Ms=Ms,
        mesh_resolution_x=mesh_resolution,
        mesh_resolution_y=mesh_resolution,
        xlabel=r'$\eta_1 = \eta_2$',
        ylabel=r'$\gamma \equiv D_1/D_2$',
        title=f'D2={D2:.2f},\nn12={n12:.2f}, dt={dt}, L0={L0:.2f}',
        x_range=eta_range,
        y_range=gamma_range,
        cluster=cluster,
        verbose=verbose,
        figname_base='localized_gamma_v_eta',
    )


def contour_plot_localized_gamma_v_eta_ratio(
        trials=10,
        verbose=False,
        recalculate_trajectory=False,
        cluster=False,
        Ms=(100,),
        gamma_range=(1e-1, 1e1),
        eta_ratio_range=(1e-1, 1e1),
        dt=0.05,
        angle=0,
):
    n2 = eta_default / dt
    n12 = eta12_default / dt
    recalculate_BF = False
    model = 'localized_different_D_detect_angle'

    args_dict = {'D2': D2,
                 'n2': n2,
                 'n12': n12,
                 'dt': dt,
                 'L0': L0,
                 'angle': angle,
                 'verbose': verbose,
                 'recalculate_trajectory': recalculate_trajectory,
                 'recalculate_BF': recalculate_BF,
                 'rotation': True,
                 'cluster': cluster,
                 'model': model}

    def update_x(args_dict, x):
        # x = eta1 / eta2 = n1 / n2
        n1 = x * n2
        args_dict.update({'n1': n1})

    def update_y(args_dict, y):
        # y = gamma = D1 / D2
        D1 = D2 * y
        args_dict.update({'D1': D1})

    calculate_and_plot_contour_plot(
        args_dict,
        x_update_func=update_x,
        y_update_func=update_y,
        trials=trials,
        Ms=Ms,
        mesh_resolution_x=mesh_resolution,
        mesh_resolution_y=mesh_resolution,
        xlabel=r'$\eta_1 / \eta_2 \equiv n_1 / n_2$',
        ylabel=r'$\gamma \equiv D_1/D_2$',
        title=f'D2={D2:.2f},\nn2={n2:.2f}, n12={n12:.2f}, dt={dt}, L0={L0:.2f}',
        x_range=eta_ratio_range,
        y_range=gamma_range,
        cluster=cluster,
        verbose=verbose,
        figname_base='localized_gamma_v_eta_ratio',
    )


def contour_plot_localized_eta12_v_M(
        trials=10,
        verbose=False,
        recalculate_trajectory=False,
        cluster=False,
        # Ms=(100,),
        eta12_range=(1e-1, 1e1),
        M_range=(100, 1000, 100),
        # M_step = 25,
        dt=0.05,
        angle=0,
):
    D1 = gamma_default * D2
    n1 = eta_default / dt
    n2 = eta_default / dt
    recalculate_BF = False
    model = 'localized_different_D_detect_angle'

    args_dict = {'D1': D1,
                 'D2': D2,
                 'n1': n1,
                 'n2': n2,
                 'dt': dt,
                 'L0': L0,
                 'angle': angle,
                 'verbose': verbose,
                 'recalculate_trajectory': recalculate_trajectory,
                 'recalculate_BF': recalculate_BF,
                 'rotation': True,
                 'cluster': cluster,
                 'model': model}

    def update_x(args_dict, x):
        # x = M
        M = x
        args_dict.update({'M': M})

    def update_y(args_dict, y):
        # y = eta12 = n12 * dt
        n12 = y / dt
        args_dict.update({'n12': n12})

    calculate_and_plot_contour_plot(
        args_dict,
        x_update_func=update_x,
        y_update_func=update_y,
        trials=trials,
        Ms=(1000,),
        mesh_resolution_y=mesh_resolution,
        xlabel=r'$M$',
        ylabel=r'$\eta_{12}$',
        title=f'D1={D1:.2f}, D2={D2:.2f},\nn1={n1:.2f}, n2={n2:.2f}, dt={dt}, L0={L0:.2f}',
        x_range=M_range,
        y_range=eta12_range,
        xscale='linear',
        cluster=cluster,
        verbose=verbose,
        figname_base='localized_eta12_v_M',
        put_M_in_title=False,
    )


def contour_plot_localized_gamma_v_M(
        trials=10,
        verbose=False,
        recalculate_trajectory=False,
        cluster=False,
        # Ms=(100,),
        gamma_range=(1e-2, 1e1),
        M_range=(100, 1000, 100),
        # M_step = 25,
        dt=0.05,
        angle=0,
):
    n1 = eta_default / dt
    n2 = eta_default / dt
    n12 = eta12_default / dt
    recalculate_BF = False
    model = 'localized_different_D_detect_angle'

    args_dict = {'D2': D2,
                 'n1': n1,
                 'n2': n2,
                 'n12': n12,
                 'dt': dt,
                 'L0': L0,
                 'angle': angle,
                 'verbose': verbose,
                 'recalculate_trajectory': recalculate_trajectory,
                 'recalculate_BF': recalculate_BF,
                 'rotation': True,
                 'cluster': cluster,
                 'model': model}

    def update_x(args_dict, x):
        # x = M
        M = x
        args_dict.update({'M': M})

    def update_y(args_dict, y):
        # y = gamma = D1/D2
        D1 = y * D2
        args_dict.update({'D1': D1})

    calculate_and_plot_contour_plot(
        args_dict,
        x_update_func=update_x,
        y_update_func=update_y,
        trials=trials,
        Ms=(1000,),
        mesh_resolution_y=mesh_resolution,
        xlabel=r'$M$',
        ylabel=r'$\gamma$',
        title=f'D2={D2:.2f},\nn1={n1:.2f}, n2={n2:.2f}, n12={n12:.2f}, dt={dt}, L0={L0:.2f}',
        x_range=M_range,
        y_range=gamma_range,
        xscale='linear',
        cluster=cluster,
        verbose=verbose,
        figname_base='localized_gamma_v_M',
        put_M_in_title=False,
    )


def plot_link_strength_dependence(trials=20, points=2 ** 5 + 1, n12_range=[1e-1, 1e3],
                                  verbose=False,
                                  recalculate_trajectory=False, recalculate_BF=False, dry_run=False,
                                  cluster=False, rotation=True):
    """
    The function loads data for the specified parameters and plots the link strength dependence plot.

    If dry_run = True, the calculations are not performed, but instead an arguments file is created
    to be fed into cluster.
    Note that the means and the confidence intervals are calculated for lg(B), not for B.
    """
    # Constants
    D1 = gamma_default * D2
    n1 = eta_default / dt
    n2 = eta_default / dt
    Ms = [10, 100, 200]  # required number of points in a trajectory; 100
    gamma = 1e-8  # viscous drag, in kg/s
    angle = 0
    model = 'localized_same_D_detect_angle'
    arguments_file = 'arguments.dat'

    # Parameters for the x axis
    n12s = np.logspace(log10(n12_range[0]), log10(n12_range[1]), num=points)

    cluster_counter = 0
    args_dict = {'D1': D1, 'D2': D2, 'n1': n1, 'n2': n2, 'gamma': gamma,
                 'dt': dt,
                 'angle': angle, 'L0': L0,
                 'verbose': verbose,
                 'recalculate_trajectory': recalculate_trajectory,
                 'recalculate_BF': recalculate_BF, 'rotation': rotation,
                 'cluster': cluster}

    with open(arguments_file, 'a') as file:
        # Calculate with different D
        args_dict.update({'model': 'localized_different_D_detect_angle'})
        lg_BF_vals_different_D = np.full([len(Ms), points, trials], np.nan)

        for trial in trange(trials, desc='Loading/scheduling calculations'):
            args_dict.update({'trial': trial})

            for ind_M, M in enumerate(Ms):
                args_dict.update({'M': M})

                for ind_n12, n12 in enumerate(n12s):
                    args_dict.update({'n12': n12})
                    lg_BF_vals_different_D[
                        ind_M, ind_n12, trial], ln_evidence_with_link, ln_evidence_free, loaded, \
                    _hash, _, trajectory = simulate_and_calculate_Bayes_factor(**args_dict)
                    if cluster and not loaded:
                        file.write(get_cluster_args_string(**args_dict))
                        cluster_counter += 1

        # Calculate with different D and seeing both particles
        args_dict.update({'model': 'localized_different_D_detect_angle_see_both'})
        lg_BF_vals_different_D_see_both = np.full([len(Ms), points, trials], np.nan)

        for trial in trange(trials, desc='Loading/scheduling calculations'):
            args_dict.update({'trial': trial})

            for ind_M, M in enumerate(Ms):
                args_dict.update({'M': M})

                for ind_n12, n12 in enumerate(n12s):
                    args_dict.update({'n12': n12})
                    lg_BF_vals_different_D_see_both[
                        ind_M, ind_n12, trial], ln_evidence_with_link, ln_evidence_free, loaded, \
                    _hash, _, trajectory = simulate_and_calculate_Bayes_factor(**args_dict)
                    if cluster and not loaded:
                        file.write(get_cluster_args_string(**args_dict))
                        cluster_counter += 1

        # Calculate with same D
        args_dict.update({'model': 'localized_same_D_detect_angle'})
        lg_BF_vals_same_D = np.full([len(Ms), points, trials], np.nan)

        for trial in trange(trials, desc='Loading/scheduling calculations'):
            args_dict.update({'trial': trial})

            for ind_M, M in enumerate(Ms):
                args_dict.update({'M': M})

                for ind_n12, n12 in enumerate(n12s):
                    args_dict.update({'n12': n12})
                    lg_BF_vals_same_D[
                        ind_M, ind_n12, trial], ln_evidence_with_link, ln_evidence_free, loaded, \
                    _hash, _, trajectory = simulate_and_calculate_Bayes_factor(**args_dict)
                    if cluster and not loaded:
                        file.write(get_cluster_args_string(**args_dict))
                        cluster_counter += 1

    if cluster and verbose:
        print('Warning: verbose was active')
    if cluster:
        # Reset start position on cluster
        position_file = 'position.dat'
        with open(position_file, 'w') as fp_position:
            fp_position.write('{0:d}'.format(0))
    print(f'{cluster_counter} calculations scheduled for the cluster')

    # Different D
    median_lg_BFs_different_D = np.nanmedian(lg_BF_vals_different_D, axis=2)
    median_lg_BFs_different_D_see_both = np.nanmedian(lg_BF_vals_different_D_see_both, axis=2)

    if trials > 1:
        CIs_different_D = np.full([len(Ms), points, 2], np.nan)
        CIs_different_D[:, :, 0] = np.nanquantile(lg_BF_vals_different_D,
                                                  (1 - confidence_level) / 2, axis=2)  # 0.025
        CIs_different_D[:, :, 1] = np.nanquantile(lg_BF_vals_different_D,
                                                  1 - (1 - confidence_level) / 2, axis=2)  # 0.975

        CIs_different_D_see_both = np.full([len(Ms), points, 2], np.nan)
        CIs_different_D_see_both[:, :, 0] = np.nanquantile(lg_BF_vals_different_D_see_both,
                                                           (1 - confidence_level) / 2,
                                                           axis=2)  # 0.025
        CIs_different_D_see_both[:, :, 1] = np.nanquantile(lg_BF_vals_different_D_see_both,
                                                           1 - (1 - confidence_level) / 2,
                                                           axis=2)  # 0.975

    xs = n12s / n1
    real_trials = np.min(
        [np.min(np.sum(~np.isnan(lg_BF_vals_different_D), axis=2)),
         np.min(np.sum(~np.isnan(lg_BF_vals_different_D), axis=2))]
    )

    # Plot lgB for different D partner
    plot_1d(xs=xs, ys=median_lg_BFs_different_D, CIs=CIs_different_D,
            fig_num=3,
            xlabel='$n_{12}/n_1$', xscale='log',
            ylabel='$\mathrm{lg}B$, diff. D partner v. no link',
            title=f"See particle 1,\ntrials={real_trials}, D1={D1:.2f}, D2={D2:.2f},n1={n1:.2f}, "
            f"\nn2={n1:.2f}, dt={dt}, L0={L0}, rotation={rotation}",
            labels=Ms,
            figname=f'link_dependence-link-or-no-link-see-one')

    # Plot same if both particles can be seen
    plot_1d(xs=xs, ys=median_lg_BFs_different_D_see_both, CIs=CIs_different_D_see_both,
            fig_num=4, clear=True,
            xlabel='$n_{12}/n_1$', xscale='log',
            ylabel='$\mathrm{lg}B$, diff. D partner v. no link',
            title=f"See both particles,\ntrials={real_trials}, D1={D1:.2f}, D2={D2:.2f},"
            f"n1={n1:.2f},\nn2={n1:.2f}, dt={dt}, L0={L0}, rotation={rotation}",
            labels=Ms,
            style='--',
            figname=f'link_dependence-link-or-no-link-see-both',
            )

    # Energy transfer v. identical partner
    lg_BF_vals_energy_transfer = lg_BF_vals_different_D - lg_BF_vals_same_D
    median_lg_BFs_energy_transfer = np.nanmedian(lg_BF_vals_energy_transfer, axis=2)

    if trials > 1:
        CIs_energy_transfer = np.full([len(Ms), points, 2], np.nan)
    CIs_energy_transfer[:, :, 0] = np.nanquantile(lg_BF_vals_energy_transfer,
                                                  (1 - confidence_level) / 2, axis=2)  # 0.025
    CIs_energy_transfer[:, :, 1] = np.nanquantile(lg_BF_vals_energy_transfer,
                                                  1 - (1 - confidence_level) / 2,
                                                  axis=2)  # 0.975
    real_trials = np.min(np.sum(~np.isnan(lg_BF_vals_energy_transfer), axis=2))

    plot_1d(xs=xs, ys=median_lg_BFs_energy_transfer, CIs=CIs_energy_transfer,
            fig_num=5,
            xlabel='$n_{12}/n_1$', xscale='log',
            ylabel='$\mathrm{lg}B_\mathrm{en}$, diff. D partner v. same D partner',
            title=f"See particle 1,\ntrials={real_trials}, D1={D1:.2f}, D2={D2:.2f},n1={n1:.2f}, "
            f"\nn2={n1:.2f}, dt={dt}, "
            f"L0={L0}, rotation={rotation}",
            figname=f'link_dependence-energy-transfer',
            labels=Ms,
            legend_loc='upper left',
            )


def plot_diffusivity_dependence(trials=20, points=2 ** 5 + 1, D1_range=[0.01, 10], verbose=False,
                                recalculate_trajectory=False, recalculate_BF=False, dry_run=False,
                                cluster=False, rotation=True):
    """
    The function loads data for the specified parameters and plots the diffusivity dependence plot.

    If dry_run = True, the calculations are not performed, but instead an arguments file is created to be fed into cluster.
    Note that the means and the confidence intervals are calculated for lg(B), not for B.
    """
    # Neeed to specify parameters to be able to load the right files

    # %% Constants
    n1 = eta_default / dt
    n2 = eta_default / dt
    n12 = eta12_default / dt
    Ms = [10, 100, 200]  # required number of points in a trajectory; 100

    gamma = 1e-8  # viscous drag, in kg/s
    angle = 0

    arguments_file = 'arguments.dat'

    D1s = np.logspace(log10(D1_range[0]), log10(D1_range[1]), num=points)

    cluster_counter = 0
    args_dict = {'D2': D2, 'n1': n1, 'n2': n2, 'n12': n12, 'gamma': gamma,
                 'dt': dt,
                 'angle': angle, 'L0': L0,
                 'verbose': verbose,
                 'recalculate_trajectory': recalculate_trajectory,
                 'recalculate_BF': recalculate_BF, 'rotation': rotation,
                 'cluster': cluster}

    with open(arguments_file, 'a') as file:
        # Calculate with same D
        args_dict.update({'model': 'localized_same_D_detect_angle'})
        lg_BF_vals_same_D = np.full([len(Ms), points, trials], np.nan)
        for trial in trange(trials, desc='Loading/scheduling calculations'):
            args_dict.update({'trial': trial})

            for ind_M, M in enumerate(Ms):
                args_dict.update({'M': M})

                for ind_D1, D1 in enumerate(D1s):
                    args_dict.update({'D1': D1})
                    # args_string = get_cluster_args_string(
                    #     D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, dt=dt, angle=angle, L=L,
                    #     trial=trial, M=M, verbose=verbose,
                    #     recalculate_trajectory=recalculate_trajectory,
                    #     recalculate_BF=recalculate_BF, rotation=rotation)
                    lg_BF_vals_same_D[
                        ind_M, ind_D1, trial], ln_evidence_with_link, ln_evidence_free, loaded, \
                    _hash, _, trajectory = simulate_and_calculate_Bayes_factor(
                        **args_dict)
                    if cluster and not loaded:
                        file.write(get_cluster_args_string(**args_dict))
                        cluster_counter += 1

        # Calculate with different D
        args_dict.update({'model': 'localized_different_D_detect_angle'})
        lg_BF_vals_different_D = np.full([len(Ms), points, trials], np.nan)
        for trial in trange(trials, desc='Loading/scheduling calculations'):
            args_dict.update({'trial': trial})

            for ind_M, M in enumerate(Ms):
                args_dict.update({'M': M})

                for ind_D1, D1 in enumerate(D1s):
                    args_dict.update({'D1': D1})
                    # args_string = get_cluster_args_string(
                    #     D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, dt=dt, angle=angle, L=L,
                    #     trial=trial, M=M, verbose=verbose,
                    #     recalculate_trajectory=recalculate_trajectory,
                    #     recalculate_BF=recalculate_BF, rotation=rotation)
                    lg_BF_vals_different_D[
                        ind_M, ind_D1, trial], ln_evidence_with_link, ln_evidence_free, loaded, \
                    _hash, _, trajectory = simulate_and_calculate_Bayes_factor(
                        **args_dict)
                    if cluster and not loaded:
                        file.write(get_cluster_args_string(**args_dict))
                        cluster_counter += 1

    if cluster and verbose:
        print('Warning: verbose was active')
    if cluster:
        # Reset start position on cluster
        position_file = 'position.dat'
        with open(position_file, 'w') as fp_position:
            fp_position.write('{0:d}'.format(0))
    print(f'{cluster_counter} calculations scheduled for the cluster')

    # Same D
    median_lg_BFs_same_D = np.nanmedian(lg_BF_vals_same_D, axis=2)

    if trials > 1:
        CIs_same_D = np.full([len(Ms), points, 2], np.nan)
        CIs_same_D[:, :, 0] = np.nanquantile(lg_BF_vals_same_D,
                                             (1 - confidence_level) / 2, axis=2)  # 0.025
        CIs_same_D[:, :, 1] = np.nanquantile(lg_BF_vals_same_D,
                                             1 - (1 - confidence_level) / 2, axis=2)  # 0.975
    xs = D1s / D2
    real_trials = np.min(np.sum(~np.isnan(lg_BF_vals_same_D), axis=2))

    # Actual plotting
    plot_1d(xs=xs, ys=median_lg_BFs_same_D, CIs=CIs_same_D,
            fig_num=4,
            xlabel='$D_1/D_2$', xscale='log',
            ylabel='$\mathrm{lg}B$, same D partner v. no link',
            title=f'trials={real_trials}, D2={D2:.2f}, n1={n1:.1f},\nn2={n1:.1f}, n12={n12:.1f}, '
            f'dt={dt}, L={L0}, rotation={rotation}',
            figname=f'diffusivity_dependence-link-or-no-link',
            labels=Ms)

    # Energy transfer v. identical partner
    lg_BF_vals_energy_transfer = lg_BF_vals_different_D - lg_BF_vals_same_D
    median_lg_BFs_energy_transfer = np.nanmedian(lg_BF_vals_energy_transfer, axis=2)

    if trials > 1:
        CIs_energy_transfer = np.full([len(Ms), points, 2], np.nan)
        CIs_energy_transfer[:, :, 0] = np.nanquantile(lg_BF_vals_energy_transfer,
                                                      (1 - confidence_level) / 2, axis=2)  # 0.025
        CIs_energy_transfer[:, :, 1] = np.nanquantile(lg_BF_vals_energy_transfer,
                                                      1 - (1 - confidence_level) / 2,
                                                      axis=2)  # 0.975
    real_trials = np.min(np.sum(~np.isnan(lg_BF_vals_energy_transfer), axis=2))

    plot_1d(xs=xs, ys=median_lg_BFs_energy_transfer, CIs=CIs_energy_transfer,
            fig_num=5,
            xlabel='$D_1/D_2$', xscale='log',
            ylabel='$\mathrm{lg}B_\mathrm{en}$, energy transfer v. same D partner',
            title=f'trials={real_trials}, D2={D2:.2f}, n1={n1:.1f},\nn2={n1:.1f}, n12={n12:.1f}, '
            f'dt={dt}, L={L0}, rotation={rotation}',
            figname=f'diffusivity_dependence-energy-transfer',
            labels=Ms)

    #
    #
    # # %% Calculating means and CI
    # median_lg_BFs = np.nanmedian(lg_BF_vals, axis=2)
    # # print('mean', median_lg_BFs)
    #
    # # BF_vals = 10**lg_BF_vals
    #
    # if trials > 1:
    #     CIs = np.full([len(Ms), D1_points, 2], np.nan)
    #     CIs[:, :, 0] = np.nanquantile(lg_BF_vals, (1 - confidence_level) / 2, axis=2)  # 0.025
    #     CIs[:, :, 1] = np.nanquantile(lg_BF_vals, 1 - (1 - confidence_level) / 2, axis=2)  # 0.975
    #     # print('CIs: ', np.log10(CIs))
    #
    #
    # real_trials = np.min(np.sum(~np.isnan(lg_BF_vals), axis=2))
    # # %% Actual plotting
    # plot_1d(xs=xs, ys=median_lg_BFs, CIs=CIs,
    #         fig_num=4,
    #         xlabel='$D_1/D_2$', xscale='log',
    #         ylabel='$\mathrm{lg}B$, same D partner v. no link',
    #         title=f'trials={real_trials}, D2={D2:.2f}, n1={n1:.1f},\nn2={n1:.1f}, n12={n12:.1f}, '
    #         f'dt={dt}, L={L0}, rotation={rotation}',
    #         figname=f'diffusivity_dependence-weak',
    #         labels=Ms)


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
    D1 = gamma_default * D2
    Ms = [10, 100, 200]  # required number of points in a trajectory; 100
    gamma = 1e-8  # viscous drag, in kg/s
    angle = 0
    n_other = eta_default / dt
    n12 = eta12_default / dt
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


def plot_angle_dependence(trials=20, points=2 ** 5 + 1, verbose=False, recalculate_trajectory=False,
                          recalculate_BF=False, dry_run=False, cluster=False):
    """
    The function loads data for the specified parameters and plots the diffusivity dependence plot.

    If dry_run = True, the calculations are not performed, but instead an arguments file is created to be fed into cluster.
    Note that the means and the confidence intervals are calculated for lg(B), not for B.
    """
    # Neeed to specify parameters to be able to load the right files

    # %% Constants
    rotation = True
    D1 = gamma_default * D2
    n1 = eta_default / dt
    n2 = eta_default / dt
    n12 = eta12_default / dt
    Ms = [10, 100, 200]  # required number of points in a trajectory; 100
    model = 'localized_different_D_detect_angle'

    arguments_file = 'arguments.dat'

    alphas = np.linspace(-np.pi, np.pi, num=points)

    cluster_counter = 0
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
                        D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, dt=dt, angle=alpha, L=L0,
                        trial=trial, M=M, verbose=verbose,
                        recalculate_trajectory=recalculate_trajectory,
                        recalculate_BF=recalculate_BF, rotation=rotation, model=model)
                    lg_BF_vals[
                        ind_M, ind_alpha, trial], ln_evidence_with_link, ln_evidence_free, \
                    loaded, _hash, _, trajectory = \
                        simulate_and_calculate_Bayes_factor(
                            D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, dt=dt, angle=alpha, L0=L0,
                            trial=trial, M=M, verbose=verbose,
                            recalculate_trajectory=recalculate_trajectory,
                            recalculate_BF=recalculate_BF, rotation=rotation, model=model,
                            cluster=cluster)

                    if trajectory.MLE_link:
                        # print(trajectory.MLE_link)
                        MLE_alphas[ind_M, ind_alpha,
                                   trial] = trajectory.MLE_link['alpha'] / np.pi * 180
                        MLE_alphas_centered[ind_M, ind_alpha,
                                            trial] = (trajectory.MLE_link[
                                                          'alpha'] - alpha) / np.pi * 180

                    # raise RuntimeError('stop')
                    if cluster and not loaded:
                        file.write(args_string)
                        cluster_counter += 1

    if cluster and verbose:
        print('Warning: verbose was active')
    if cluster:
        # Reset start position on cluster
        position_file = 'position.dat'
        with open(position_file, 'w') as fp_position:
            fp_position.write('{0:d}'.format(0))
    print(f'{cluster_counter} calculations scheduled for the cluster')

    # %% Calculating means and CI
    real_trials = np.min(np.sum(~np.isnan(lg_BF_vals), axis=2))
    median_lg_BFs = np.nanmedian(lg_BF_vals, axis=2)

    if trials > 1:
        CIs = np.full([len(Ms), points, 2], np.nan)
        CIs[:, :, 0] = np.nanquantile(lg_BF_vals, (1 - confidence_level) / 2, axis=2)  # 0.025
        CIs[:, :, 1] = np.nanquantile(lg_BF_vals, 1 - (1 - confidence_level) / 2, axis=2)  # 0.975
    xs = alphas / np.pi * 180

    # %% Plotting Bayes factors
    plot_1d(xs=xs, ys=median_lg_BFs, CIs=CIs,
            fig_num=6,
            xlabel='$\\alpha$, $^\circ$',
            ylabel='Median $\mathrm{lg}(B)$',
            title=f'trials={real_trials}, D1={D1:.2f}, D2={D2:.2f}, n1={n1:.1f},\nn2={n1:.1f}, '
            f'n12={n12:.1f}, dt={dt}, L={L0}, rotation={rotation}',
            figname=f'alpha_dependence',
            labels=Ms)

    # %% Plotting alpha inference
    median_alphas_centered = np.nanmedian(MLE_alphas_centered, axis=2)
    if trials > 1:
        CIs = np.full([len(Ms), points, 2], np.nan)
        CIs[:, :, 0] = np.nanquantile(
            MLE_alphas_centered, (1 - confidence_level) / 2, axis=2)  # 0.025
        CIs[:, :, 1] = np.nanquantile(MLE_alphas_centered, 1 - (1 - confidence_level) / 2, axis=2)

    plot_1d(xs=xs, ys=median_alphas_centered, CIs=CIs,
            fig_num=7,
            xlabel='$\\alpha$, $^\circ$',
            ylabel='$\\hat\\alpha - \\alpha$), $^\circ$',
            title=f'trials={real_trials}, D1={D1:.2f}, D2={D2:.2f}, n1={n1:.1f},\nn2={n1:.1f}, '
            f'n12={n12:.1f}, dt={dt}, L={L0}, rotation={rotation}',
            figname=f'angle_inference',
            labels=Ms, y_levels=None)


# def plot_free_hookean_length_dependence(trials=3, verbose=False,
#                                         recalculate_trajectory=False,
#                                         cluster=False, Ms=[100], l0_range=[1e-1, 1e4], delete_old
#                                         =False,
#                                         model='free_same_D'):
#     """
#     The function loads data for the specified parameters and plots the link strength dependence plot.
#
#     If dry_run = True, the calculations are not performed, but instead an arguments file is created to be fed into cluster.
#     Note that the means and the confidence intervals are calculated for lg(B), not for B.
#     """
#     # Neeed to specify parameters to be able to load the right files
#
#     # %% Constants
#     D1 = gamma_default * D2
#     n1 = 0
#     n2 = 0
#     n12 = 0  # s^{-1}. Somehting interesting happens between [1e-9; 1e-6]
#     Ms = [1000]  # [10, 100, 200, 1000]  # required number of points in a trajectory; 100
#     # N = 101
#
#     dt = 0.05  # s 0.3
#     gamma = 1e-8  # viscous drag, in kg/s
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
#     mesh_points = 2 ** 3 + 1  # 50
#     l0s = np.logspace(log10(l0_range[0]), log10(l0_range[1]), num=mesh_points)
#
#     with open(arguments_file, 'a') as file:
#         lg_BF_vals = np.full([len(Ms), mesh_points, trials], np.nan)
#         for trial in trange(trials, desc='Loading/scheduling calculations'):
#
#             for ind_M, M in enumerate(Ms):
#                 for ind, l0 in enumerate(l0s):
#                     L0 = l0 * np.sqrt(D1 * dt)
#                     args_string = get_cluster_args_string(
#                         D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, dt=dt, angle=angle, L=L0,
#                         trial=trial, M=M, verbose=verbose, model=model,
#                         recalculate_trajectory=recalculate_trajectory)
#                     # print('Calculating with parameters: ', args_string)
#                     lg_BF_vals[
#                         ind_M, ind, trial], ln_evidence_with_link, ln_evidence_free, \
#                     loaded, dict_data, _hash = simulate_and_calculate_Bayes_factor(
#                         D1=D1, D2=D2, n1=n1,
#                         n2=n2,
#                         n12=n12, dt=dt,
#                         angle=angle, L0=L0,
#                         trial=trial,
#                         M=M, model=model,
#                         recalculate_trajectory=recalculate_trajectory,
#                         verbose=verbose,
#                         cluster=cluster)
#
#                     # simulate_and_calculate_Bayes_factor_terminal(
#                     # args_string, cluster=cluster)
#
#                     if cluster and not loaded:
#                         file.write(args_string)
#
#                     # raise RuntimeError('stop')
#
#     if cluster and verbose:
#         print('Warning: verbose was active')
#         # return np.nan
#
#     # %% Calculating means and CI
#     median_lg_BFs = np.nanmedian(lg_BF_vals, axis=2)
#     # print('mean', median_lg_BFs)
#
#     # BF_vals = 10**lg_BF_vals
#
#     if trials > 1:
#         CIs = np.full([len(Ms), mesh_points, 2], np.nan)
#         CIs[:, :, 0] = np.nanquantile(lg_BF_vals, (1 - confidence_level) / 2, axis=2)  # 0.025
#         CIs[:, :, 1] = np.nanquantile(lg_BF_vals, 1 - (1 - confidence_level) / 2, axis=2)  # 0.975
#         # print('CIs: ', np.log10(CIs))
#
#     # %% Actual plotting
#     # fig = plt.figure(num=3, clear=True)
#     fig = set_figure_size(num=4, rows=rows, page_width_frac=page_width_frac,
#                           height_factor=height_factor)
#
#     real_trials = np.min(np.sum(~np.isnan(lg_BF_vals), axis=2))
#     # print('real_trials', real_trials)
#     # Confidence intervals
#     # ax = plt.gca()
#     xs = l0s
#     for ind_M in range(len(Ms)):
#         color = color_sequence[ind_M]
#         zorder = len(Ms) - ind_M
#         if real_trials > 1:
#             plt.fill_between(xs, CIs[ind_M, :, 0], CIs[ind_M, :, 1],
#                              alpha=alpha_shade, color=color, zorder=zorder)
#             # plt.plot(xs, np.log10(CIs[:, 0]), '-', color='g', alpha=alpha_shade)
#             # plt.plot(xs, np.log10(CIs[:, 1]), '-', color='g', alpha=alpha_shade)
#
#         # Mean
#         plt.plot(xs, median_lg_BFs[ind_M, :], color=color, label=f'M={Ms[ind_M]:d}', zorder=zorder)
#
#     # print(CIs)
#     # Significance levels
#     xlims = plt.xlim()
#     plt.plot(xlims, [-1] * 2, '--', color='k', lw=lw_theory, zorder=0)
#     plt.plot(xlims, [1] * 2, '--', color='k', lw=lw_theory, zorder=0)
#
#     plt.xscale('log')
#     plt.xlabel('$\ell_0$')
#     plt.ylabel('Median $\mathrm{lg}(B)$')
#     plt.title(
#         f'trials={real_trials}, D1={D1:.2f}, D2={D2:.2f}, n1={n1:.2f}, \nn2={n1:.2f}, '
#         f'dt={dt}')
#
#     plt.legend(loc='upper left')
#     plt.tight_layout()
#     plt.show()
#
#     # fig_folder = 'figures'
#     # figname = f'link_dependence-weak'
#     # figpath = os.path.join(fig_folder, figname)
#     # plt.savefig(figpath + '.png', bbox_inches='tight', pad_inches=0)
#     # plt.savefig(figpath + '.pdf', bbox_inches='tight', pad_inches=0)
#
#     return lg_BF_vals


# def plot_free_hookean_link_strength_dependence(trials=3, n12_range=[1e-1, 1e3], Ms=[100,
#                                                                                     200, 1000],
#                                                verbose=False,
#                                                recalculate_trajectory=False,
#                                                cluster=False, rotation=True, model=
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
#     # Ms = [10, 100, 200, 1000]  # required number of points in a trajectory; 100
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
#     n12_points = 2 ** 4 + 1  # 50
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
#                         recalculate_trajectory=recalculate_trajectory, rotation=rotation)
#                     # print('Calculating with parameters: ', args_string)
#                     lg_BF_vals[
#                         ind_M, ind_n12, trial], ln_evidence_with_link, ln_evidence_free, \
#                     loaded, dict_data, _hash = simulate_and_calculate_Bayes_factor(
#                         D1=D1, D2=D2, n1=n1,
#                         n2=n2,
#                         n12=n12, dt=dt,
#                         angle=angle, L0=L,
#                         trial=trial,
#                         M=M, model=model,
#                         recalculate_trajectory=recalculate_trajectory,
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
#     fig = set_figure_size(num=4, rows=rows, page_width_frac=page_width_frac,
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
                                      cluster=False, Ms=(100,),
                                      l0_range=(1e-1, 1e3),
                                      eta_range=(1e-2, 1e2),
                                      dt=0.05):
    D1 = D2 = 0.4  # um^2/s
    n1 = 0
    n2 = 0
    recalculate_BF = False
    model = 'free_same_D'
    mesh_resolution = 2 ** 4 + 1

    args_dict = {'D1': D1, 'D2': D2,
                 'n1': n1, 'n2': n2,
                 'dt': dt,
                 'angle': 0,
                 'verbose': verbose,
                 'recalculate_trajectory': recalculate_trajectory,
                 'recalculate_BF': recalculate_BF,
                 'rotation': True,
                 'cluster': cluster,
                 'model': model}

    def update_x(args_dict, x):
        # x = eta = n12 * dt
        n12 = x / dt
        args_dict.update({'n12': n12})

    def update_y(args_dict, y):
        # y = l0 = L0 / \sqrt{4 * D * dt}
        L0 = y * np.sqrt(4 * D1 * dt)
        args_dict.update({'L0': L0})

    calculate_and_plot_contour_plot(
        args_dict,
        x_update_func=update_x,
        y_update_func=update_y,
        trials=trials,
        Ms=Ms,
        mesh_resolution_x=mesh_resolution,
        mesh_resolution_y=mesh_resolution,
        xlabel=r'$\eta_{12}$',
        ylabel=r'$\ell_0\equiv L_0/\sqrt{4D\Delta t}$',
        title=f'D1={D1:.2f},\nD2={D2:.2f}, dt={dt}',
        x_range=eta_range,
        y_range=l0_range,
        cluster=cluster,
        verbose=verbose,
        figname_base='localized_eta12_v_eta_ratio',
        plot_simulation_time=True,
    )

    # # ===
    #
    #
    # # %% Constants
    # D1 = D2 = 0.4  # um^2/s
    # n1 = 0
    # n2 = 0
    # model = 'free_same_D'
    #
    # mesh_resolution = 2 ** 4 + 1
    # mesh_resolution_l0 = mesh_resolution
    # mesh_resolution_eta = mesh_resolution
    #
    # # l_range = [1e-1, 1e3]
    # l0s = np.logspace(log10(l0_range[0]), log10(l0_range[1]), num=mesh_resolution_l0)
    # # L0s = ls * np.sqrt(D1 * dt)
    #
    # # eta_range = [1e-2, 1e2]
    # etas = np.logspace(log10(eta_range[0]), log10(eta_range[1]), num=mesh_resolution_eta)
    # # n12s = etas / dt
    # cluster_counter = 0
    #
    # with open(arguments_file, 'a') as file:
    #
    #     lg_BF_vals = np.full([len(Ms), mesh_resolution_eta, mesh_resolution_l0, trials], np.nan)
    #     simulation_time = np.empty_like(lg_BF_vals)
    #     for trial in trange(trials, desc='Loading/scheduling calculations'):
    #
    #         for ind_M, M in enumerate(Ms):
    #             for ind_eta, eta in enumerate(etas):
    #                 n12 = eta / dt
    #                 for ind_l0, l0 in enumerate(l0s):
    #                     L0 = l0 * np.sqrt(4 * D1 * dt)
    #                     args_string = get_cluster_args_string(
    #                         D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, dt=dt, L0=L0,
    #                         trial=trial, M=M, verbose=verbose, model=model,
    #                         recalculate_trajectory=recalculate_trajectory)
    #                     # print('Calculating with parameters: ', args_string)
    #                     # with stopwatch():
    #                     lg_BF_vals[
    #                         ind_M, ind_eta, ind_l0, trial], ln_evidence_with_link, \
    #                     ln_evidence_free, loaded, _hash, simulation_time[
    #                         ind_M, ind_eta, ind_l0, trial], dict_data = \
    #                         simulate_and_calculate_Bayes_factor(
    #                             D1=D1, D2=D2, n1=n1,
    #                             n2=n2,
    #                             n12=n12, dt=dt,
    #                             angle=0, L0=L0,
    #                             trial=trial,
    #                             M=M, model=model,
    #                             recalculate_trajectory=recalculate_trajectory,
    #                             recalculate_BF=False,
    #                             verbose=verbose,
    #                             cluster=cluster,
    #                             rotation=True)
    #                     # todo: why is rotation here, but not above?
    #
    #                     # simulate_and_calculate_Bayes_factor_terminal(
    #                     # args_string, cluster=cluster)
    #
    #                     if cluster and not loaded:
    #                         file.write(args_string)
    #                         cluster_counter += 1
    #
    #                     # delete_data(_hash)
    #
    # if cluster and verbose:
    #     print('Warning: verbose was active')
    # if cluster:
    #     # Reset start position on cluster
    #     position_file = 'position.dat'
    #     with open(position_file, 'w') as fp_position:
    #         fp_position.write('{0:d}'.format(0))
    # print(f'{cluster_counter} calculations scheduled for the cluster')
    #
    # # %% Calculating means and CI
    # median_lg_BFs = np.nanmedian(lg_BF_vals, axis=3)
    # median_simulation_time_hours = np.nanmedian(simulation_time, axis=3) / 3600
    #
    # if trials > 1:
    #     CIs = np.full([len(Ms), mesh_resolution_eta, mesh_resolution_l0, 2], np.nan)
    #     CIs[:, :, :, 0] = np.nanquantile(lg_BF_vals, (1 - confidence_level) / 2, axis=3)  # 0.025
    #     CIs[:, :, :, 1] = np.nanquantile(lg_BF_vals, 1 - (1 - confidence_level) / 2,
    #                                      axis=3)  # 0.975
    #     CI_widths = CIs[:, :, :, 1] - CIs[:, :, :, 0]
    #     min_CI_width = np.floor(np.nanmin(CI_widths))
    #     max_CI_width = np.ceil(np.nanmax(CI_widths))
    #
    # # %% Actual plotting
    #
    # for ind_M, M in enumerate(Ms):
    #     real_trials = np.min(np.sum(~np.isnan(lg_BF_vals), axis=3), axis=(1, 2))
    #     max_real_trials = np.max(np.sum(~np.isnan(lg_BF_vals), axis=3), axis=(1, 2))
    #
    #     X, Y = np.meshgrid(etas, l0s)
    #     str_title = f'trials={real_trials[ind_M]} ({max_real_trials[ind_M]:d}), ' \
    #         f'M={Ms[ind_M]:d}, D1={D1:.2f},\nD2={D2:.2f}, dt={dt}'
    #     lgB_levels = [-1, 0, 1]
    #     xlabel = r'$\eta\equiv n_{12}\Delta t$'
    #     ylabel = r'$\ell_0\equiv L_0/\sqrt{4D\Delta t}$'
    #
    #     # LgB values
    #     clims = np.array([-1, 1]) * np.ceil(np.abs(np.nanmax(median_lg_BFs)))
    #     levels = np.arange(clims[0], clims[1] + 1, 1)
    #
    #     contour_plot(fig_num=3 + 3 * ind_M, X=X, Y=Y,
    #                  Z=median_lg_BFs[ind_M, :, :].T,
    #                  clims=clims,
    #                  levels=levels,
    #                  lgB_levels=lgB_levels,
    #                  cb_label='Median $\mathrm{lg}B$',
    #                  xscale='log', yscale='log',
    #                  xlabel=xlabel, ylabel=ylabel,
    #                  title=str_title,
    #                  figname=f'free_hookean_M={Ms[ind_M]:d}',
    #                  cmap='bwr'
    #                  )
    #
    #     # Confidence intervals
    #     levels = np.arange(min_CI_width, max_CI_width + 1, 1)
    #     contour_plot(fig_num=3 + 1 + 3 * ind_M, X=X, Y=Y,
    #                  Z=CI_widths[ind_M, :, :].T,
    #                  levels=levels,
    #                  cb_label='$\mathrm{lg}B$, 95 \% CI width',
    #                  xscale='log', yscale='log',
    #                  xlabel=xlabel, ylabel=ylabel,
    #                  title=str_title,
    #                  figname=f'free_hookean_CI_M={Ms[ind_M]:d}',
    #                  cmap='Reds'
    #                  )
    #
    #     # Simulation time
    #     contour_plot(fig_num=3 + 2 + 3 * ind_M, X=X, Y=Y,
    #                  Z=median_simulation_time_hours[ind_M, :, :].T,
    #                  Z_lgB=median_lg_BFs[ind_M, :, :].T,
    #                  lgB_levels=lgB_levels,
    #                  cb_label='Simulation time, hours',
    #                  xscale='log', yscale='log',
    #                  xlabel=xlabel, ylabel=ylabel,
    #                  title=str_title,
    #                  figname=None,
    #                  cmap='bwr'
    #                  )

    # return lg_BF_vals
