from plot import D2 as D2_default
from plot import (
    L0,
    calculate_and_plot_contour_plot,
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

        """
        self.trials = trials
        self.verbose = verbose
        self.recalculate_trajectory = recalculate_trajectory
        self.cluster = cluster
        self.Ms = Ms
        self.dt = dt
        self.angle = angle
        self.print_MLE = print_MLE
        self.mesh_resolution = 2 ** resolution + 1
        self.rotation = rotation
        self.x_label = x_label
        self.x_range = x_range
        self.update_x = update_x
        self.x_scale = xscale
        self.y_scale = yscale
        self.y_label = y_label
        self.y_range = y_range
        self.update_y = update_y
        self.title = title
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
        self.lg_BF_vals = None

    def run(self, **kwargs):
        """Perform/schedule calculations with and without a link.

        Returns
        -------

        """
        self.lg_BF_vals = calculate_and_plot_contour_plot(
            self.default_args_dict,
            x_update_func=self.update_x,
            y_update_func=self.update_y,
            trials=self.trials,
            Ms=self.Ms,
            mesh_resolution_x=self.mesh_resolution,
            mesh_resolution_y=self.mesh_resolution,
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
