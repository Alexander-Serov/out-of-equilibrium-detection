"""
This file contains code of periodogram calculation and other analysis.
The part that concerns periodogram likelihoods calculation and fitting parameters can be found in `likelihood.py`
"""
import ast

import numpy as np
from numpy import log
from scipy.fftpack import fft

from likelihood import get_MLE
from support import (
    delete_data,
    hash_from_dictionary,
    load_data,
    save_data,
    save_mle_guess,
    stopwatch,
)
from trajectory import Trajectory

max_abs_lg_B_per_M = 2


def calculate_periodogram(dX1, dY1, dt, D1, D2, dk=10, plot=True):
    """Calcualte the periodogram

    Parameters:
    dk - mode grouping and averaging window (for plotting only)

    """
    M = len(dX1)
    dX1_image = dt * fft(dX1)
    dY1_image = dt * fft(dY1)
    df = 1 / dt / M
    modes = np.arange(M)
    PX = np.abs(dX1_image) ** 2 / (M * dt)
    PY = np.abs(dY1_image) ** 2 / (M * dt)

    PX_norm = PX / (D1 * dt ** 2)
    PY_norm = PY / (D1 * dt ** 2)

    # # Plot
    # if plot:
    #     # Normalize to D1 * dt
    #
    #     # The individual components are too noisy, so for plotting it's better to locally average them
    #     # The averaging is only used for plotting, not for fitting
    #     dk = 10  # grouping window
    #     PX_norm_avg, modes_avg = average_over_modes(PX_norm, dk)
    #     PY_norm_avg, modes_avg = average_over_modes(PY_norm, dk)
    #
    #     plot_periodogram(modes_avg=modes_avg,
    #                      PX_norm_avg=PX_norm_avg,
    #                      PY_norm_avg=PY_norm_avg,
    #                      D1=D1, D2=D2, M=M)
    return PX, PY, PX_norm, PY_norm, modes


def calculate_bayes_factor(
    t,
    dR,
    true_parameters,
    hash,
    dim=2,
    recalculate=False,
    plot=False,
    verbose=False,
    true_args=None,
    cluster=False,
    rotation=True,
):
    """
    Calculate log10 Bayes factor for the presence of the link between 2 particles.
    """

    dX1 = dR[0, :]
    dX2 = dR[2, :]
    dY1 = dR[1, :]
    dt = t[1] - t[0]
    M = len(dX1)
    D1, D2 = true_parameters["D1"], true_parameters["D2"]

    # Load Bayes factor if previously calculated
    hash, hash_no_trial = hash_from_dictionary(dim=dim, parameters=true_parameters)
    dict_data, loaded = load_data(hash)
    # print(loaded, dict_data)
    # return
    # print(recalculate)
    if not recalculate and "lg_B" in dict_data.keys():
        if not np.isnan(dict_data["lg_B"]):
            lg_bayes_factor, ln_evidence_with_link, ln_evidence_free = [
                dict_data[key] for key in "lg_B ln_evid_link ln_evid_no_link".split()
            ]

            # print('Bayes factor values reloaded.')

            return (
                lg_bayes_factor,
                ln_evidence_with_link,
                ln_evidence_free,
                loaded,
                dict_data,
            )
        else:
            print("Loaded lg_B = NaN. Recalculating")

    # else:
    #     print('lg_B not found in saved file, or recalculation was requested')
    #     print('Details:', dict_data)

    if cluster:
        return np.nan, np.nan, np.nan, False, {}
    print("\nPerforming calculations with parameters: ", true_args)

    # Calculate the periodogram
    PX, PY, PX_norm, PY_norm, modes = calculate_periodogram(
        dX1=dX1, dY1=dY1, dt=dt, D1=D1, D2=D2, dk=10, plot=plot
    )
    # short because it takes only the first 2 components X1, Y1
    dRk_short = dt * fft(dR[:2, :])

    # Choose the k numbers that will be used to construct the likelihood
    #
    # Note: for even M, one must be careful with fitting k=0, k=M/2 because they are real,
    # for odd M, k=0 is real
    # For the moment, I just do not take these frequencies into account

    if not M % 2:  # even M
        fit_indices = np.arange(1, M / 2, dtype=np.int)
    else:  # odd M
        fit_indices = np.arange(1, (M - 1) / 2 + 1, dtype=np.int)
    ks_fit = fit_indices
    PX_fit = PX[fit_indices]
    PY_fit = PY[fit_indices]
    dRks_fit = dRk_short[:, fit_indices]

    # print('ks and dRks', ks_fit, list(dRks_fit))
    # raise RuntimeError('Stop requested')

    # # %% MLE and evidence for the model without a link
    print("\nCalculating no-link evidence...")
    with stopwatch("No-link evidence calculation"):
        MLE_free, ln_evidence_free, max_free, success_free = get_MLE(
            ks=ks_fit,
            zs_x=PX_fit,
            zs_y=PY_fit,
            dRks=dRks_fit,
            hash_no_trial=hash_no_trial,
            M=M,
            dt=dt,
            link=False,
            verbose=verbose,
            rotation=False,
            true_parameters=true_parameters,
        )

    if success_free:
        print("Done!", ln_evidence_free)
    else:
        print("Calculation failed!")

    # %% Infer the MLE for the model with link
    if success_free:
        print("\nCalculating evidence with link...")
        with stopwatch("Evidence calculation with link"):
            MLE_link, ln_evidence_with_link, max_link, success_link = get_MLE(
                ks=ks_fit,
                zs_x=PX_fit,
                zs_y=PY_fit,
                dRks=dRks_fit,
                hash_no_trial=hash_no_trial,
                M=M,
                dt=dt,
                link=True,
                verbose=verbose,
                rotation=rotation,
                true_parameters=true_parameters,
            )  # , start_point=true_parameters)
        if success_link:
            print("Done!", ln_evidence_with_link)
        else:
            print("Calculation failed!")
    else:
        MLE_link, ln_evidence_with_link, max_link = [np.nan] * 3
        success_link = False
    # link_hess = np.linalg.inv(max_link.hess_inv)

    # Bayes factor
    lg_bayes_factor = (ln_evidence_with_link - ln_evidence_free) / log(10)
    # print('ln evidence with and without link', ln_evidence_with_link, ln_evidence_free)
    print("lg Bayes factor for the presence of the link ", lg_bayes_factor)

    # raise RuntimeError()

    # Save data to disk
    dict_data["lg_B"] = lg_bayes_factor
    dict_data["ln_evid_link"] = ln_evidence_with_link
    dict_data["ln_evid_no_link"] = ln_evidence_free
    dict_data["MLE_free"] = MLE_free
    dict_data["MLE_link"] = MLE_link

    if success_free and success_link:
        save_data(dict_data, hash)
    else:
        delete_data(hash)

    return lg_bayes_factor, ln_evidence_with_link, ln_evidence_free, True, dict_data


def average_over_modes(input, dk):
    """Makes averages of the 1D array in over the interval of mode indices dk"""
    M = len(input)
    if dk == 1:
        k_new = np.arange(M)
        return input, k_new

    M_new = np.floor(M / dk).astype(int)
    out = np.zeros(M_new) * np.nan
    for i in range(M_new):
        out[i] = np.mean(input[dk * i : dk * (i + 1)])
    k_new = dk / 2 + dk * np.arange(M_new)

    return out, k_new


def simulate_and_calculate_Bayes_factor(
    D1,
    D2,
    n1,
    n2,
    n12,
    dt,
    angle,
    L0,
    trial,
    M,
    model,
    seed=None,
    recalculate_trajectory=False,
    recalculate_BF=False,
    verbose=False,
    true_args=None,
    cluster=False,
    rotation=True,
    collect_mle=False,
    **kwargs
):
    """
    The function combines trajectory simulation and bayes factor calculation to be able to delegate
    the task to a computation cluster.

    Parameters
    ----------
    collect_mle
        If True, do not perform calculations, but just collect the MLE estimates for
        points, for which they are available.
        If dry_run, the collection will be performed on the cluster.
        Otherwise - locally.
    """
    traj = Trajectory(
        D1=D1,
        D2=D2,
        n1=n1,
        n2=n2,
        n12=n12,
        M=M,
        dt=dt,
        L0=L0,
        trial=trial,
        angle=angle,
        recalculate=recalculate_trajectory,
        dry_run=cluster,
        model=model,
        collect_mle=collect_mle,
    )
    loaded = False if np.isnan(traj.lgB) else True

    if loaded and collect_mle:
        true_params = dict(
            D1=D1,
            D2=D2,
            n1=n1,
            n2=n2,
            n12=n12,
            M=M,
            dt=dt,
            L0=L0,
            angle=angle,
            model=model,
        )
        for link, mle in [
            (True, traj.MLE_link),
            (False, traj.MLE_no_link),
        ]:
            true_params["link"] = link
            fun = traj.get_ln_posterior(link)(**mle)

            save_mle_guess(
                true_params=true_params,
                mle=mle,
                value=-fun,
            )

    return (
        traj.lgB,
        traj.ln_model_evidence_with_link,
        traj.ln_model_evidence_no_link,
        loaded,
        traj.hash,
        traj.simulation_time,
        traj,
    )


def simulate_and_calculate_Bayes_factor_terminal(args_str: str, cluster: bool = False):
    """Same as `simulate_and_calculate_Bayes_factor`, but takes input arguments as
    `repr` of the arguments dictionary.

    Parameters
    ----------
    args_str : str
        `repr` of the arguments dictionary defining the calculation.
    cluster : bool
        If `True`, the calculations will not be performed now, but scheduled for the cluster in
        the arguments file.

    Returns
    -------
    tuple
        Simulation results. Same as `simulate_and_calculate_Bayes_factor`.

    """
    args_dict = ast.literal_eval(args_str)
    args_dict["cluster"] = cluster

    return simulate_and_calculate_Bayes_factor(**args_dict)
