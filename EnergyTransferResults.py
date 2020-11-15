import copy
import json
import warnings
from pathlib import Path
from typing import Any, Iterable, Tuple

import numpy as np
from numpy import log10
from tqdm.auto import trange

from calculate import simulate_and_calculate_Bayes_factor
from plot import D2 as D2_default
from plot import (
    L0,
    arguments_file,
    calculate_and_plot_contour_plot_for_class,
    contour_plot,
    eta12_default,
    eta_default,
    gamma_default,
)

JSON_INDENT = 2

M_DEFAULT = 1000  # Default trajectory length


class EnergyTransferResults:
    cache_folder = Path(".cache")
    cache_folder.mkdir(exist_ok=True)

    def __init__(
        self,
        trials,
        cluster,
        Ms,
        x_label,
        x_range,
        update_x,
        y_label,
        y_range,
        update_y,
        dt,
        angle,
        rotation,
        print_MLE=False,
        resolution=3,
        statistic="mean",
        title="",
        figname_base="figure",
        clip=10,
        verbose=False,
        recalculate_BF=False,
        recalculate_trajectory=False,
        xscale="log",
        yscale="log",
        default_values_dot: Tuple = None,
        plot_3_color_version: bool = True,
    ):
        """Store simulation parameters.

        Parameters
        ----------
        resolution: int
            Mesh resolution z that. Each integer value of z corresponds to
            (2**z + 1) points along each axis in the produced figure.
        update_x : Callable
            Function of (dict, float) that takes a dictionary and updates its
            values to correspond the given float value.
        update_y : Callable
            Same as `update_x` for the `y` axis.
        x_range
            Can be borders (a tuple of 2) or a borders with step (tuple of 3). #todo
        plot_3_color_version
            If True, also plot a 3-color version, where only "link", "no link" and
            "inconclusive" are shown, i.e. "energy transfer" and "no energy transfer"
            are merged into "link".

        """
        self.trials = trials
        self.verbose = verbose
        self.recalculate_trajectory = recalculate_trajectory
        self.cluster = cluster
        self.Ms = Ms
        self.dt = dt
        self.angle = angle
        self.print_MLE = print_MLE
        self.mesh_resolution_x = 2 ** resolution + 1
        self.mesh_resolution_y = 2 ** resolution + 1
        self.rotation = rotation
        self.x_label = x_label
        self.x_range = x_range
        self.update_x = update_x
        self.x_scale = xscale
        self.y_scale = yscale
        self.y_label = y_label
        self.y_range = y_range
        self.update_y = update_y
        self.figname_base = figname_base
        self.statistic = statistic
        self.clip = clip
        self.plot_3_color_version = plot_3_color_version
        self.cache_filename = (self.cache_folder / self.figname_base).with_suffix(
            ".json"
        )
        self.vars_to_store = [
            "lg_BF_vals",
            "simulation_time",
            "full_time",
            "ln_evidence_with_links",
            "ln_evidence_frees",
            "MLE_links",
            "MLE_no_links",
        ]

        self.recalculate_BF = recalculate_BF
        self.models = {
            "with_eng_transfer": "localized_different_D_detect_angle",
            "no_eng_transfer": "localized_same_D_detect_angle",
        }

        self.default_args_dict = {
            "n1": eta_default / self.dt,
            "n2": eta_default / self.dt,
            "n12": eta12_default / self.dt,
            "D1": gamma_default * D2_default,
            "D2": D2_default,
            "dt": self.dt,
            "angle": angle,
            "L0": L0,
            "verbose": self.verbose,
            "recalculate_trajectory": self.recalculate_trajectory,
            "recalculate_BF": self.recalculate_BF,
            "rotation": self.rotation,
            "cluster": self.cluster,
        }

        # Create a dictionary that will be used for expanding plot titles
        self.default_values_dict_for_plot = copy.deepcopy(self.default_args_dict)
        self.default_values_dict_for_plot.update(
            {
                "eta1": eta_default,
                "eta2": eta_default,
                "eta12": eta12_default,
                "gamma": gamma_default,
                "M": M_DEFAULT,
            }
        )

        # Substitute the placeholders in title with the default parameter values
        self.title = self.substitute_default_parameters(title)

        # Calculate ranges
        if len(self.x_range) == 2:
            self.Xs = np.logspace(
                log10(self.x_range[0]),
                log10(self.x_range[1]),
                num=self.mesh_resolution_x,
            )
        elif len(self.x_range) == 3:
            self.Xs = np.arange(
                self.x_range[0], self.x_range[1] + self.x_range[2], self.x_range[2]
            )
        else:
            self.Xs = self.x_range

        if len(self.y_range) == 2:
            self.Ys = np.logspace(
                log10(self.y_range[0]),
                log10(self.y_range[1]),
                num=self.mesh_resolution_y,
            )
        elif len(self.y_range) == 3:
            self.Ys = np.arange(
                self.y_range[0], self.y_range[1] + self.y_range[2], self.y_range[2]
            )
        else:
            self.Ys = self.y_range

        # Substitute the default values to be added to the plot
        self.default_values_dot = default_values_dot
        if self.default_values_dot is not None:
            self.default_values_dot = tuple(
                [
                    self.default_values_dict_for_plot[el]
                    for el in self.default_values_dot
                ]
            )

        # Expand models
        if self.models is None:
            if "model" not in self.args_dict:
                raise ValueError("`model` must be provided.")
            else:
                warnings.warn(
                    "Consider using the new interface for supplying model information."
                )
                self.models = {
                    self.default_args_dict["model"]: self.default_args_dict["model"]
                }

        # Initialize results arrays
        self.results_shape = [
            len(self.models),
            len(self.Ms),
            len(self.Xs),
            len(self.Ys),
            self.trials,
        ]
        self.lg_BF_vals = np.full(self.results_shape, np.nan)
        self.simulation_time = np.empty_like(self.lg_BF_vals)
        self.full_time = np.empty_like(self.lg_BF_vals)
        self.ln_evidence_with_links = np.full_like(self.lg_BF_vals, np.nan)
        self.ln_evidence_frees = np.full_like(self.lg_BF_vals, np.nan)
        self.MLE_links = {}
        self.MLE_no_links = {}

        self.load_cache()

    def run(self, **kwargs):
        """Perform/schedule calculations with and without a link.

        Returns
        -------

        """

        args_dict = copy.deepcopy(self.default_args_dict)

        cluster_counter = 0
        loaded_new = False
        with open(arguments_file, "a") as file:

            for trial in trange(self.trials, desc="Loading/scheduling calculations"):
                args_dict.update({"trial": trial})

                for ind_M, M in enumerate(self.Ms):
                    args_dict.update({"M": M})

                    for ind_model, model in enumerate(self.models.values()):
                        args_dict.update({"model": model})

                        for ind_y, y in enumerate(self.Ys):
                            self.update_y(args_dict, y)

                            for ind_x, x in enumerate(self.Xs):
                                self.update_x(args_dict, x)

                                # Only try to load from disk if has not been loaded
                                # before (including results found in the cache)
                                if np.isnan(
                                    self.lg_BF_vals[
                                        ind_model, ind_M, ind_x, ind_y, trial
                                    ]
                                ):
                                    (
                                        lg_bf,
                                        ln_evidence_with_link,
                                        ln_evidence_free,
                                        loaded,
                                        _hash,
                                        self.simulation_time[
                                            ind_model, ind_M, ind_x, ind_y, trial
                                        ],
                                        traj,
                                    ) = simulate_and_calculate_Bayes_factor(**args_dict)

                                    loaded_new = True

                                    self.lg_BF_vals[
                                        ind_model, ind_M, ind_x, ind_y, trial
                                    ] = lg_bf

                                    times = {
                                        "simulation_time": traj.simulation_time,
                                        "calculation_time_link": traj.calculation_time_link,
                                        "calculation_time_no_link": traj.calculation_time_no_link,
                                    }
                                    self.full_time[
                                        ind_model, ind_M, ind_x, ind_y, trial
                                    ] = np.sum(list(times.values()))

                                    self.ln_evidence_with_links[
                                        ind_model, ind_M, ind_x, ind_y, trial
                                    ] = ln_evidence_with_link
                                    self.ln_evidence_frees[
                                        ind_model, ind_M, ind_x, ind_y, trial
                                    ] = ln_evidence_free

                                    # Get the MLE estimates
                                    if loaded:
                                        self.MLE_links[
                                            (ind_model, M, x, y, trial)
                                        ] = traj.MLE_link
                                        self.MLE_no_links[
                                            (ind_model, M, x, y, trial)
                                        ] = traj.MLE_no_link

                                    if self.cluster and not loaded:
                                        file.write(str(args_dict) + "\n")
                                        cluster_counter += 1

                                if self.print_MLE and trial == 0:
                                    print(
                                        "\nPrinting MLEs.\nArguments:\n",
                                        repr(args_dict),
                                    )
                                    print("MLE with link:\t", traj.MLE_link)
                                    print("MLE no link:\t", traj.MLE_link)
                                    print("lgB for link:\t", traj.lgB)

        if loaded_new and np.any(~np.isnan(self.lg_BF_vals)):
            # Save if something new was loaded and at least one value is not nan
            self.save_cache()

        if self.cluster and self.verbose:
            print("Warning: verbose was active")

        calculate_and_plot_contour_plot_for_class(
            self.lg_BF_vals,
            trials=self.trials,
            Ms=self.Ms,
            xlabel=self.x_label,
            ylabel=self.y_label,
            xscale=self.x_scale,
            yscale=self.y_scale,
            title=self.title,
            figname_base=self.figname_base,
            models=self.models,
            statistic=self.statistic,
            simulation_time=self.simulation_time,
            full_time=self.full_time,
            cluster_counter=cluster_counter,
            Xs=self.Xs,
            Ys=self.Ys,
            default_values_dot=self.default_values_dot,
            **kwargs,
        )

    def collect_mle_guesses(self, **kwargs):
        """Extract mle guesses from already calculated files.

        Returns
        -------

        """
        args_dict = copy.deepcopy(self.default_args_dict)
        args_dict["collect_mle"] = True

        cluster_counter = 0
        with open(arguments_file, "a") as file:

            for trial in trange(self.trials, desc="Loading/scheduling calculations"):
                args_dict.update({"trial": trial})

                for ind_M, M in enumerate(self.Ms):
                    args_dict.update({"M": M})

                    for ind_model, model in enumerate(self.models.values()):
                        args_dict.update({"model": model})

                        for ind_y, y in enumerate(self.Ys):
                            self.update_y(args_dict, y)

                            for ind_x, x in enumerate(self.Xs):
                                self.update_x(args_dict, x)

                                (
                                    lg_bf,
                                    ln_evidence_with_link,
                                    ln_evidence_free,
                                    loaded,
                                    _hash,
                                    self.simulation_time[
                                        ind_model, ind_M, ind_x, ind_y, trial
                                    ],
                                    traj,
                                ) = simulate_and_calculate_Bayes_factor(**args_dict)

                                self.ln_evidence_with_links[
                                    ind_model, ind_M, ind_x, ind_y, trial
                                ] = ln_evidence_with_link
                                self.ln_evidence_frees[
                                    ind_model, ind_M, ind_x, ind_y, trial
                                ] = ln_evidence_free

                                # Get the MLE estimates
                                if loaded:
                                    self.MLE_links[
                                        (ind_model, M, x, y, trial)
                                    ] = traj.MLE_link
                                    self.MLE_no_links[
                                        (ind_model, M, x, y, trial)
                                    ] = traj.MLE_no_link

                                # Store arguments for cluster evaluation if loaded
                                if self.cluster and loaded:
                                    file.write(str(args_dict) + "\n")
                                    cluster_counter += 1

    def save_cache(self):
        """Caches the results for the given figure.
        The cache file will be loaded before trying to load individual results.
        """
        results = {}
        for var_name in self.vars_to_store:
            var = getattr(self, var_name)
            if isinstance(var, np.ndarray):
                results[var_name] = var.tolist()
            elif isinstance(var, dict):
                converted = [{"key": key, "value": value} for key, value in var.items()]
                results[var_name] = converted
            else:
                results[var_name] = var

        with open(self.cache_filename, "w") as f:
            json.dump(results, f, indent=JSON_INDENT)
        print(f"Figure data cache updated successfully (`{self.cache_filename}`).\n")

    def load_cache(self):
        """Load the cache file."""
        n_loaded = 0
        try:
            with open(self.cache_filename, "r") as f:
                results = json.load(f)
        except FileNotFoundError:
            print(
                f"Cache file not found at the expected location "
                f"`{self.cache_filename}`.\n"
                f"The results will be reloaded."
            )
            return

        def backup_old_cache():
            # Backup the old file
            backup = str(self.cache_filename) + ".bak"
            self.cache_filename.replace(backup)
            print(f"Old cache file backed up to `{backup}`.")

        # Use the loaded results only if they have the expected size
        if not all([var in results for var in self.vars_to_store]):
            print(
                f"Cache found at `{self.cache_filename}`, but some variables were "
                "missing from the cache.\n"
                "The results will be reloaded."
            )
            backup_old_cache()
            return

        if not all(
            [
                np.array(results[var]).shape == getattr(self, var).shape
                for var in self.vars_to_store
                if isinstance(getattr(self, var), np.ndarray)
            ]
        ):
            print(
                f"Cache found `{self.cache_filename}`, but some of the arrays have "
                "different shape.\n"
                "The results will be reloaded."
            )
            backup_old_cache()
            return

        for var_name in self.vars_to_store:
            # Process loaded values based on the expected type
            if isinstance(getattr(self, var_name), np.ndarray):
                loaded = np.array(results[var_name])
            elif isinstance(getattr(self, var_name), dict):
                loaded = {}
                for el in results[var_name]:
                    if not isinstance(  # Convert keys to tuples if necessary
                        el["key"], str
                    ) and isinstance(el["key"], Iterable):
                        loaded[tuple(el["key"])] = el["value"]
                    else:
                        loaded[el["key"]] = el["value"]
            else:
                loaded = results[var_name]
            setattr(self, var_name, loaded)

        print(f"Figure cache loaded successfully. ")

    def substitute_default_parameters(self, str_in: str) -> str:
        """Substitute default parameter values in the input string if found.

        Parameters
        ----------
        str_in
            Input formatted string containing parameters to substitute in braces.

        Returns
        -------
        str
            Same string with format placeholders substituted with the default
            parameter values.

        """
        # Create a subset of the substitutions that we need
        substitutions = {
            k: v for k, v in self.default_values_dict_for_plot.items() if k in str_in
        }

        str_out = str_in.format(**substitutions)

        return str_out

    def plot_mle(self, link: bool, parameter: str, model: Any, clip: float = None):
        """Plot a 2D map of the median of the maximum-likelihood estimates of the
        posterior distribution.

        Parameters
        ----------
        link
            If True, plot the results from the link model.
            Otherwise, use the no-link model.
        parameter
            Name of the parameter to plot.
        model
            Must be on of the keys of the models dictionary of the instance.
        clip
            If provided, clip the values at a given level.

        Returns
        -------
        Nothing, but plots the figure.
        """
        for i, key in enumerate(self.models.keys()):
            if key == model:
                model_ind = i
                break
        else:
            raise ValueError(f"Model `{model}` not found in this instance.")

        # Prepare the data
        data_dict = self.MLE_links if link else self.MLE_no_links
        data_array = np.full(
            (
                len(self.Xs),
                len(self.Ys),
                self.trials,
            ),
            np.nan,
        )
        # print(list(data_dict.keys())[0])
        for i_x, x in enumerate(self.Xs):
            for i_y, y in enumerate(self.Ys):
                for trial in range(self.trials):
                    try:
                        key = (model_ind, self.Ms[0], x, y, trial)
                        data_array[i_x, i_y, trial] = data_dict[
                            (model_ind, self.Ms[0], x, y, trial)
                        ][parameter]
                    except KeyError:
                        # print(f'Key `{key}` not found.')
                        pass
        data_median = np.nanmedian(data_array, axis=2)

        contour_plot(
            X=self.Xs,
            Y=self.Ys,
            Z=data_median.T,
            cb_label=f"{parameter}",
            xscale=self.x_scale,
            yscale=self.y_scale,
            xlabel=self.x_label,
            ylabel=self.y_label,
            title=f"Median MLE for {parameter}, link={link}, model="
            f"{model.replace('_', '-')}",
            cmap="bwr",
            clip=clip,
        )

    def plot_evidence(self, link: bool, model: Any, clip: float = None):
        """Plot a 2D map of the mean of the evidence for the given
        link status and model.

        Parameters
        ----------
        link
            If True, plot the results from the link model.
            Otherwise, use the no-link model.
        model
            Must be on of the keys of the models dictionary of the instance.
        clip
            If provided, clip the values at a given level.

        Returns
        -------
        Nothing, but plots the figure.
        """
        for i, key in enumerate(self.models.keys()):
            if key == model:
                model_ind = i
                break
        else:
            raise ValueError(f"Model `{model}` not found in this instance.")

        # Prepare the data
        data_array = self.ln_evidence_with_links if link else self.ln_evidence_frees
        data_mean = np.nanmean(data_array[model_ind, 0, :, :, :], axis=2)

        contour_plot(
            X=self.Xs,
            Y=self.Ys,
            Z=data_mean.T,
            cb_label=f"Evidence",
            xscale=self.x_scale,
            yscale=self.y_scale,
            xlabel=self.x_label,
            ylabel=self.y_label,
            title=f"Mean evidence for link={link}, model={model.replace('_', '-')}",
            cmap="bwr",
            clip=clip,
        )
