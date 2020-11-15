"""
Define 'trajectory' class that will manage simulation and fitting of the trajectories.
"""
import logging
import time
import warnings

import numpy as np
from scipy.fftpack import fft

from likelihood import (
    get_ln_likelihood_func_2_particles_x_link,
    get_ln_likelihood_func_free_hookean_no_link_same_D,
    get_ln_likelihood_func_free_hookean_with_link_same_D,
    get_ln_likelihood_func_no_link,
    get_ln_prior_func,
    get_MLE,
    max_expected_D,
    max_expected_eta,
    max_expected_eta12,
)
from simulate import (
    simulate_2_confined_particles_with_fixed_angle_bond,
    simulate_a_free_hookean_dumbbell,
)
from support import hash_from_dictionary, load_data, save_data


class Trajectory:
    def __init__(
        self,
        D1=2.0,
        D2=0.4,
        n1=1.0,
        n2=1.0,
        n12=30.0,
        M=1000,
        dt=0.05,
        L0=10.0,
        trial=0,
        angle=-np.pi / 3,
        recalculate=False,
        recalculate_BF=False,
        dry_run=False,
        plot=False,
        model="localized_different_D",
        dim=2,
        verbose=True,
    ):

        # Parameters for hash
        self.D1 = D1
        self.D2 = D2
        self.n1 = n1
        self.n2 = n2
        self.n12 = n12
        self.M = M
        self.dt = dt
        self.L0 = L0
        self.trial = trial
        self.angle = angle
        self.model = model
        self.dim = dim
        self.parameters = {
            "D1": D1,
            "D2": D2,
            "n1": n1,
            "n2": n2,
            "n12": n12,
            "M": M,
            "dt": dt,
            "L0": L0,
            "model": model,
            "angle": angle,
            "trial": trial,
        }
        if trial == 0:
            self.check_parameter_values()

        if self.n1 == 0 and self.n2 == 0:
            self.simulation_function = simulate_a_free_hookean_dumbbell
        else:
            self.simulation_function = (
                simulate_2_confined_particles_with_fixed_angle_bond
            )

        self.recalculate = recalculate
        self.recalculate_BF = recalculate_BF
        self.dry_run = dry_run
        self.plot = plot
        self.verbose = verbose
        (
            self._ln_prior,
            self._sample_from_the_prior,
            self.prior_hash,
        ) = get_ln_prior_func(self.dt)
        self.parameters["prior_hash"] = self.prior_hash
        # print('Prior hash:', self.prior_hash)

        # Variables calculated later
        (
            self._t,
            self._R,
            self._dR,
            self._dRk,
            self._ks,
            self._lgB,
            self._MLE_link,
            self._MLE_no_link,
            self._ln_model_evidence_with_link,
            self._ln_model_evidence_no_link,
        ) = [None] * 10
        self._simulation_time = np.nan
        self._calculation_time_link = np.nan
        self._calculation_time_no_link = np.nan
        self._both = False

        # Calculate hash and load
        self._dict_data = {}
        self._hash, self._hash_no_trial = self.calculate_hash()
        if not self.recalculate:
            self._load_data()

            if self.recalculate_BF:
                # Clean everything except the trajectory
                new_dict_data = {**self.parameters}
                for key in "t R dR _simulation_time".split():
                    if key in self._dict_data:
                        new_dict_data.update({key: self._dict_data[key]})
                self._dict_data = new_dict_data
                self._save_data()
        else:
            self._dict_data = {**self.parameters}
            self._save_data()

        # Choose a model to fit
        if not self.dry_run:
            if model == "free_same_D":
                self._ln_likelihood_with_link = (
                    get_ln_likelihood_func_free_hookean_with_link_same_D(
                        ks=self.ks, M=self.M, dt=self.dt, dRks=self.dRks
                    )
                )
                self._ln_likelihood_no_link = (
                    get_ln_likelihood_func_free_hookean_no_link_same_D(
                        ks=self.ks, M=self.M, dt=self.dt, dRks=self.dRks
                    )
                )
                self._names_with_link = ("D1", "n12", "L0")
                self._names_no_link = ("D1",)

            elif model == "free_different_D":
                raise NotImplemented()

            elif model == "localized_same_D_detect_angle":
                self._ln_likelihood_with_link = (
                    get_ln_likelihood_func_2_particles_x_link(
                        ks=self.ks,
                        M=self.M,
                        dt=self.dt,
                        dRks=self.dRks,
                        rotation=True,
                        same_D=True,
                    )
                )
                self._ln_likelihood_no_link = get_ln_likelihood_func_no_link(
                    ks=self.ks, M=self.M, dt=self.dt, dRks=self.dRks
                )
                self._names_with_link = ("D1", "n1", "n2", "n12", "alpha")
                self._names_no_link = ("D1", "n1")

            elif model == "localized_different_D_detect_angle":
                self._ln_likelihood_with_link = (
                    get_ln_likelihood_func_2_particles_x_link(
                        ks=self.ks, M=self.M, dt=self.dt, dRks=self.dRks, rotation=True
                    )
                )
                self._ln_likelihood_no_link = get_ln_likelihood_func_no_link(
                    ks=self.ks, M=self.M, dt=self.dt, dRks=self.dRks
                )
                self._names_with_link = ("D1", "D2", "n1", "n2", "n12", "alpha")
                self._names_no_link = ("D1", "n1")

            elif model == "localized_same_D_detect_angle_see_both":
                self._both = True
                self._ln_likelihood_with_link = (
                    get_ln_likelihood_func_2_particles_x_link(
                        ks=self.ks,
                        M=self.M,
                        dt=self.dt,
                        dRks=self.dRks,
                        rotation=True,
                        same_D=True,
                        both=True,
                    )
                )
                self._ln_likelihood_no_link = get_ln_likelihood_func_no_link(
                    ks=self.ks,
                    M=self.M,
                    dt=self.dt,
                    dRks=self.dRks,
                    same_D=True,
                    both=True,
                )
                self._names_with_link = ("D1", "n1", "n2", "n12", "alpha")
                self._names_no_link = ("D1", "n1", "n2")

            elif model == "localized_different_D_detect_angle_see_both":
                self._both = True
                self._ln_likelihood_with_link = (
                    get_ln_likelihood_func_2_particles_x_link(
                        ks=self.ks,
                        M=self.M,
                        dt=self.dt,
                        dRks=self.dRks,
                        rotation=True,
                        both=True,
                    )
                )
                self._ln_likelihood_no_link = get_ln_likelihood_func_no_link(
                    ks=self.ks, M=self.M, dt=self.dt, dRks=self.dRks, both=True
                )
                self._names_with_link = ("D1", "D2", "n1", "n2", "n12", "alpha")
                self._names_no_link = ("D1", "D2", "n1", "n2")
            else:
                raise NotImplementedError()

    # Another constructor
    @classmethod
    def from_parameter_dictionary(cls, parameters):
        return cls(**parameters)

    @property
    def t(self):
        if self._t is None:
            # if self.dry_run:
            #     return np.array([np.nan])
            self._simulate()
        return self._t

    @property
    def R(self):
        if self._R is None:
            self._simulate()
        return self._R

    @property
    def dR(self):
        if self._dR is None:
            self._simulate()
        return self._dR

    @property
    def dRks(self):
        # raise RuntimeError('Test')
        # print('Test 2: ', self._dRk)
        if self._dRk is None:
            self._calculate_DFT()
        return self._dRk

    @property
    def ks(self):
        if self._ks is None:
            self._calculate_DFT()
        return self._ks

    @property
    def hash(self):
        return self._hash

    @property
    def hash_no_trial(self):
        return self._hash_no_trial

    @property
    def ln_model_evidence_with_link(self):
        if self._ln_model_evidence_with_link is None or not np.isfinite(
            self._ln_model_evidence_with_link
        ):
            # check loaded
            if (
                self.model in self._dict_data
                and "_ln_model_evidence_with_link" in self._dict_data[self.model]
                and np.isfinite(
                    self._dict_data[self.model]["_ln_model_evidence_with_link"]
                )
            ):
                self._ln_model_evidence_with_link = self._dict_data[self.model][
                    "_ln_model_evidence_with_link"
                ]
                return self._ln_model_evidence_with_link

            elif "_ln_model_evidence_with_link" in self._dict_data:
                # This clause is kept for compatibility. Remove later
                warnings.warn(
                    "Saving data without a model name will be deprecated in the future",
                    PendingDeprecationWarning,
                )
                self._ln_model_evidence_with_link = self._dict_data[
                    "_ln_model_evidence_with_link"
                ]
                return self._ln_model_evidence_with_link

            # calculate
            if self.dry_run:
                self._ln_model_evidence_with_link = np.nan
            else:
                # print(self.model)

                # start = time.time()
                # self._t, self._R, self._dR, _ = self.simulation_function(
                #     parameters=self.parameters, plot=self.plot, show=False)
                # self._simulation_time = time.time() - start
                #
                # # Save
                # self._dict_data = {'t': self._t, 'R': self._R, 'dR': self._dR, '_simulation_time':
                #     self._simulation_time, **self.parameters}

                start = time.time()
                (
                    self._MLE_link,
                    self._ln_model_evidence_with_link,
                    min,
                    success,
                ) = self._calculate_MLE(link=True, names=self._names_with_link)
                self._calculation_time_link = time.time() - start
                model_results = {
                    "_MLE_link": self._MLE_link,
                    "_ln_model_evidence_with_link": self._ln_model_evidence_with_link,
                    "_calculation_time_link": self._calculation_time_link,
                }

                # Save
                if np.isfinite(self._ln_model_evidence_with_link):
                    if self.model in self._dict_data:
                        self._dict_data[self.model].update(model_results)
                    else:
                        self._dict_data[self.model] = model_results
                    self._save_data()
        return self._ln_model_evidence_with_link

    @property
    def MLE_link(self):
        if self._MLE_link is None:
            # check loaded
            if (
                self.model in self._dict_data
                and "_MLE_link" in self._dict_data[self.model]
            ):
                self._MLE_link = self._dict_data[self.model]["_MLE_link"]
                return self._MLE_link

            elif "_MLE_link" in self._dict_data:
                # This clause is kept for compatibility. Remove later
                warnings.warn(
                    "Saving data without a model name will be deprecated in the future",
                    PendingDeprecationWarning,
                )
                self._MLE_link = self._dict_data["_MLE_link"]
                return self._MLE_link

            # calculate
            if self.dry_run:
                self._MLE_link = {}
            else:
                self.ln_model_evidence_with_link
        return self._MLE_link

    @property
    def ln_model_evidence_no_link(self):
        if self._ln_model_evidence_no_link is None or not np.isfinite(
            self._ln_model_evidence_no_link
        ):
            # if self.model in self._dict_data and '_ln_model_evidence_no_link' in \
            #         self._dict_data[self.model]:
            #     self._ln_model_evidence_no_link = self._dict_data[self.model][
            #         '_ln_model_evidence_no_link']
            #     return self._ln_model_evidence_no_link
            label_evidence = "_ln_model_evidence_no_link"
            label_MLE = "_MLE_no_link"
            if self._both:
                label_evidence += "_both"
                label_MLE += "_both"

            if label_evidence in self._dict_data and np.isfinite(
                self._dict_data[label_evidence]
            ):
                # The no link result does not depend on the model, but depends on seeing both
                # particles
                self._ln_model_evidence_no_link = self._dict_data[label_evidence]
                return self._ln_model_evidence_no_link

            # calculate
            if self.dry_run:
                self._ln_model_evidence_no_link = np.nan
            else:
                # start = time.time()
                # self._t, self._R, self._dR, _ = self.simulation_function(
                #     parameters=self.parameters, plot=self.plot, show=False)
                # self._simulation_time = time.time() - start
                #
                # # Save
                # self._dict_data = {'t': self._t, 'R': self._R, 'dR': self._dR, '_simulation_time':
                #     self._simulation_time, **self.parameters}
                start = time.time()
                (
                    self._MLE_no_link,
                    self._ln_model_evidence_no_link,
                    min,
                    success,
                ) = self._calculate_MLE(link=False, names=self._names_no_link)
                self._calculation_time_no_link = time.time() - start

                # Save
                if np.isfinite(self._ln_model_evidence_no_link):
                    model_results = {
                        label_MLE: self._MLE_no_link,
                        label_evidence: self._ln_model_evidence_no_link,
                        "_calculation_time_no_link": self._calculation_time_no_link,
                    }
                    # if self.model in self._dict_data:
                    self._dict_data.update(model_results)
                    # else:
                    #     self._dict_data[self.model] = model_results

                    self._save_data()
        return self._ln_model_evidence_no_link

    @property
    def MLE_no_link(self):
        if self._MLE_no_link is None:
            label_no_link = "_MLE_no_link"
            if self._both:
                label_no_link += "_both"

            if label_no_link in self._dict_data:
                # The no link result does not depend on the model, but depends on seeing both
                self._MLE_no_link = self._dict_data[label_no_link]
                return self._MLE_no_link

            # calculate
            if self.dry_run:
                self._MLE_no_link = {}
            else:
                self.ln_model_evidence_no_link
        return self._MLE_no_link

    @property
    def lgB(self):
        if self._lgB is None or not np.isfinite(self._lgB):
            # check loaded
            if (
                self.model in self._dict_data
                and "_lgB" in self._dict_data[self.model]
                and np.isfinite(self._dict_data[self.model]["_lgB"])
            ):
                self._lgB = self._dict_data[self.model]["_lgB"]
                return self._lgB

            elif "_lgB" in self._dict_data:
                # todo This clause is kept for compatibility. Remove later
                warnings.warn(
                    "Saving _lgB without a model name will be deprecated in the future",
                    PendingDeprecationWarning,
                )
                self._lgB = self._dict_data["_lgB"]
                return self._lgB

            # calculate
            self._calculate_bayes_factor()
            # Save
            if np.isfinite(self._lgB):
                model_results = {"_lgB": self._lgB}
                if self.model in self._dict_data:
                    self._dict_data[self.model].update(model_results)
                else:
                    self._dict_data[self.model] = model_results
                self._save_data()

        return self._lgB

    @property
    def simulation_time(self):
        if np.isnan(self._simulation_time) and "_simulation_time" in self._dict_data:
            self._simulation_time = self._dict_data["_simulation_time"]

        return self._simulation_time

    @property
    def calculation_time_link(self):
        if (
            np.isnan(self._calculation_time_link)
            and self.model in self._dict_data
            and "_calculation_time_link" in self._dict_data[self.model]
        ):
            self._calculation_time_link = self._dict_data[self.model][
                "_calculation_time_link"
            ]

        return self._calculation_time_link

    @property
    def calculation_time_no_link(self):
        if (
            np.isnan(self._calculation_time_no_link)
            and "_calculation_time_no_link" in self._dict_data
        ):
            self._calculation_time_no_link = self._dict_data[
                "_calculation_time_no_link"
            ]

        return self._calculation_time_no_link

    def calculate_hash(self, old_hash=False):
        return hash_from_dictionary(
            parameters=self.parameters, dim=self.dim, use_model=old_hash
        )

    def _load_data(self):
        self._dict_data, _ = load_data(self._hash)
        # If not loaded, try with the old hash
        if "t" not in self._dict_data:
            self._dict_data, _ = load_data(self.calculate_hash(old_hash=True)[0])

    def _save_data(self):
        # self._dict_data, _ = load_data(self._hash)
        if not self.dry_run:
            save_data(dict_data=self._dict_data, hash=self._hash)

    def _simulate(self):
        # Check if can load
        if not self.recalculate:
            self._load_data()
            if np.all([key in self._dict_data for key in "t R dR".split()]):
                self._t, self._R, self._dR = [
                    self._dict_data[key] for key in "t R dR".split()
                ]
                # if '_simulation_time'in self._dict_data:
                #     self._simulation_time = self._dict_data['_simulation_time']
                return

        # Calculate
        start = time.time()
        self._t, self._R, self._dR, _ = self.simulation_function(
            parameters=self.parameters, plot=self.plot, show=False
        )
        self._simulation_time = time.time() - start

        # Save
        self._dict_data = {
            "t": self._t,
            "R": self._R,
            "dR": self._dR,
            "_simulation_time": self._simulation_time,
            **self.parameters,
        }
        self._save_data()
        print("Trajectory simulated successfully")
        # save_data(dict_data=dict_data, hash=self._hash)

    def _calculate_DFT(self):
        # print('Test 1: ', self._both, self.dR.shape, self.model)
        # Calculate
        if self._both:
            self._dRk = self.dt * fft(self.dR)
        else:
            self._dRk = self.dt * fft(self.dR[:2, :])

        # Choose the frequencies k that will be used to construct the likelihood.
        # Note: for even M, one must be careful with fitting k=0, k=M/2 because they are real,
        # for odd M, k=0 is real.
        # For the moment, I just do not take these frequencies into account.

        if not self.M % 2:  # even M
            fit_indices = np.arange(1, self.M / 2, dtype=np.int)
        else:  # odd M
            fit_indices = np.arange(1, (self.M - 1) / 2 + 1, dtype=np.int)
        self._ks = fit_indices
        self._dRk = self._dRk[:, fit_indices]

    def get_ln_posterior(self, link):
        if not link:

            def _ln_posterior(**kwargs):
                return self._ln_prior(**kwargs) + self._ln_likelihood_no_link(**kwargs)

        else:

            def _ln_posterior(**kwargs):
                return self._ln_prior(**kwargs) + self._ln_likelihood_with_link(
                    **kwargs
                )

        return _ln_posterior

    def _calculate_MLE(self, link, names):

        # names = self._names_with_link if link else self._names_no_link
        if self.verbose:
            print(
                "\nMLE search started for a simulation with true parameters:\n",
                self.parameters,
            )
            print(f"Hash: {self.hash}\nHash no trial: {self.hash_no_trial}")

        if link:
            log_lklh = self._ln_likelihood_with_link
        else:
            log_lklh = self._ln_likelihood_no_link
        # print('A', log_lklh)

        return get_MLE(
            ln_posterior=self.get_ln_posterior(link),
            names=names,
            sample_from_the_prior=self._sample_from_the_prior,
            hash_no_trial=self._hash_no_trial,
            link=link,
            log_lklh=log_lklh,
        )

    def _calculate_bayes_factor(self):
        self._lgB = (
            self.ln_model_evidence_with_link - self.ln_model_evidence_no_link
        ) / np.log(10)
        if not self.dry_run:
            print(f"Bayes factor: {self._lgB}")

    def check_parameter_values(self):
        # Check if the prior will allow inference for this parameters
        dct = {
            "D1": (self.D1, max_expected_D),
            "D2": (self.D2, max_expected_D),
            "n1": (self.n1, max_expected_eta / self.dt),
            "n2": (self.n2, max_expected_eta / self.dt),
            "n12": (self.n12, max_expected_eta12 / self.dt),
        }

        for key, value in dct.items():
            # print(key, value)
            if value[0] > 2 * value[1]:
                logging.warning(
                    f"The simulated value of {key} is outside the expected region: "
                    f"{value[0]:.4g} > {value[1]:.4g}."
                    f"\nIf the two values are far apart, the inference will likely "
                    f"fail."
                )


# Tests
if __name__ == "__main__":
    # test_traj = Trajectory(n1=0, n2=0, M=100)
    # test_traj = Trajectory.from_parameter_dictionary(
    #     {'D1': 0.4,
    #      'D2': 0.4,
    #      'n1': 1,
    #      'n2': 1,
    #      'n12': 2e-4 / 0.05,
    #      'L0': 100 * np.sqrt(4 * 0.4 * 0.05),
    #      'M': 500,
    #      # 'model': 'localized_same_D_detect_angle',
    #      # 'model': 'localized_different_D_detect_angle',
    #      'model':'localized_different_D_detect_angle_see_both',
    #      'recalculate': 0,
    #      'plot': 1,
    #      'dry_run': 0,
    #      'trial': 1})

    # test_traj = Trajectory.from_parameter_dictionary(
    #     {'D1': 0.0224937, 'D2': 0.4, 'n1': 1.0, 'n2': 1.0, 'n12': 30.0, 'M': 101, 'dt': 0.05,
    #      'L0': 20.0, 'model': 'localized_different_D_detect_angle', 'angle': 0.0, 'trial': 10,
    #      'recalculate': 0,
    #      'plot': 1,
    #      'dry_run': 0,
    #      }
    # )

    test_traj = Trajectory.from_parameter_dictionary(
        {
            "D2": 0.1,
            "n1": 1.0,
            "n2": 1.0,
            "n12": 30.0,
            "dt": 0.05,
            "L0": 20,
            "angle": 0,
            "model": "localized_different_D_detect_angle",
            "trial": 47,
            "M": 8,
            "D1": 0.001,
            "recalculate_BF": 0,
            "verbose": 1,
            "recalculate": 0,
        }
    )

    # print(test_traj.dR.shape)
    # print(test_traj.dRks.shape)
    # print(test_traj.ks)
    # print('link = True: ', test_traj.ln_posterior(link=True, D1=1, n12=10, L0=1))
    # print('link = False: ', test_traj.ln_posterior(link=False, D1=1))

    # print('\nEvidence with link: ', test_traj.ln_model_evidence_with_link, 'MLE: ',
    #       test_traj.MLE_link)
    # print('\nEvidence without link: ', test_traj.ln_model_evidence_no_link, 'MLE: ',
    #       test_traj.MLE_no_link)
    print("Hash: ", test_traj.hash)
    print("Hash no trial: ", test_traj.hash_no_trial)
    print("Lg Bayes factor for link: ", test_traj.lgB)
    print("Calc time link: ", test_traj.calculation_time_link, "s")
    print("Calc time no link: ", test_traj.calculation_time_no_link, "s")

    # print('\nSimulation time: ', test_traj.simulation_time)
