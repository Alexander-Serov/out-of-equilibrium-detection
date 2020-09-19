import copy
import json
import warnings
from pathlib import Path

import numpy as np
from numpy import log10
from tqdm.auto import trange

from calculate import simulate_and_calculate_Bayes_factor
from plot import D2 as D2_default
from plot import (
    L0,
    arguments_file,
    calculate_and_plot_contour_plot_for_class,
    eta12_default,
    eta_default,
    gamma_default,
)

JSON_INDENT = 2


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
        self.cache_filename = (self.cache_folder / self.figname_base).with_suffix(
            ".json"
        )
        self.vars_to_store = ["lg_BF_vals", "simulation_time", "full_time"]

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
        self.plot_title_values_dict = copy.deepcopy(self.default_args_dict)
        self.plot_title_values_dict.update({"eta1": eta_default, "eta2": eta_default})

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
                                        self.lg_BF_vals[
                                            ind_model, ind_M, ind_x, ind_y, trial
                                        ],
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
                                    times = {
                                        "simulation_time": traj.simulation_time,
                                        "calculation_time_link": traj.calculation_time_link,
                                        "calculation_time_no_link": traj.calculation_time_no_link,
                                    }
                                    self.full_time[
                                        ind_model, ind_M, ind_x, ind_y, trial
                                    ] = np.sum(list(times.values()))

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
            self.save_cache()

        if self.cluster and self.verbose:
            print("Warning: verbose was active")

        self.lg_BF_vals = calculate_and_plot_contour_plot_for_class(
            self.default_args_dict,
            x_update_func=self.update_x,
            y_update_func=self.update_y,
            trials=self.trials,
            Ms=self.Ms,
            mesh_resolution_x=self.mesh_resolution_x,
            mesh_resolution_y=self.mesh_resolution_y,
            xlabel=self.x_label,
            ylabel=self.y_label,
            x_range=self.x_range,
            y_range=self.y_range,
            xscale=self.x_scale,
            yscale=self.y_scale,
            title=self.title,
            cluster=self.cluster,
            verbose=self.verbose,
            figname_base=self.figname_base,
            models=self.models,
            clip=self.clip,
            print_MLE=self.print_MLE,
            statistic=self.statistic,
            lg_BF_vals=self.lg_BF_vals,
            simulation_time=self.simulation_time,
            full_time=self.full_time,
            cluster_counter=cluster_counter,
            Xs=self.Xs,
            Ys=self.Ys,
            **kwargs,
        )

    def save_cache(self):
        """Caches the results for the given figure.
        The cache file will be loaded before trying to load individual results.
        """
        results = {}
        for var_name in self.vars_to_store:
            results[var_name] = getattr(self, var_name).tolist()

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

        # Use the loaded results only if they have the expected size
        if not all([var in results for var in self.vars_to_store]):
            print(
                "Cache found at `{self.cache_filename}`, but some variables were "
                "missing from the cache.\n"
                "The results will be reloaded."
            )
            return

        if not all(
            [
                np.array(results[var]).shape == getattr(self, var).shape
                for var in self.vars_to_store
            ]
        ):
            print(
                "Cache found `{self.cache_filename}`, but some of the variables have "
                "different shape.\n"
                "The results will be reloaded."
            )
            return

        for var_name in self.vars_to_store:
            loaded = np.array(results[var_name])
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
            k: v for k, v in self.plot_title_values_dict.items() if k in str_in
        }

        str_out = str_in.format(**substitutions)

        return str_out
