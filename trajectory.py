"""
Define 'trajectory' class that will manage simulation and fitting of the trajectories.
"""
from support import hash_from_dictionary, load_data, save_data, stopwatch
import numpy as np
from simulate import simulate_2_confined_particles_with_fixed_angle_bond, \
    simulate_a_free_hookean_dumbbell
from scipy.fftpack import fft, ifft
from likelihood import get_ln_prior_func, get_ln_likelihood_func_free_hookean_with_link_same_D, \
    get_ln_likelihood_func_free_hookean_no_link_same_D, get_MLE, \
    get_ln_likelihood_func_2_particles_x_link, get_ln_likelihood_func_no_link
import time


class Trajectory:
    def __init__(self, D1=2.0, D2=0.4, n1=1.0, n2=1.0, n12=30.0, M=1000, dt=0.05, L0=10.0,
                 trial=0, angle=-np.pi / 3, recalculate=False, dry_run=False, plot=False,
                 model='localized_different_D', dim=2):
        """
        :param D1:
        :param D2:
        :param n1: Localization constant (k1/gamma) of particle 1. Both n12 and n2 must be set to
        0 for a free dumbbell simulation.

        :param n2: Localization constant (k2/gamma) of particle 2. Both n12 and n2 must be set to
        0 for a free dumbbell simulation.

        :param n12:
        :param M:
        :param dt:
        :param L0:
        :param trial:

        :param model: {'free_same_D', 'free_different_D', 'localized_same_D',
        'localized_different_D'}    Define the model to fit to the simulated data.

        :param dim: Dimensionality of the problem. For the moment, only 2D is supported.
        """

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
        self.parameters = {'D1': D1, 'D2': D2, 'n1': n1, 'n2': n2, 'n12': n12, 'M': M,
                           'dt': dt, 'L0': L0, 'model': model, 'angle': angle, 'trial': trial}

        if self.n1 == 0 and self.n2 == 0:
            self.simulation_function = simulate_a_free_hookean_dumbbell
        else:
            self.simulation_function = simulate_2_confined_particles_with_fixed_angle_bond

        self.recalculate = recalculate
        self.dry_run = dry_run
        self.plot = plot
        self._ln_prior, self._sample_from_the_prior = get_ln_prior_func()

        # Variables calculated later
        self._t, self._R, self._dR, self._dRk, self._ks, self._lgB, self._MLE_link, \
        self._MLE_no_link, self._ln_model_evidence_with_link, self._ln_model_evidence_no_link, \
            = [None] * 10
        self._simulation_time = np.nan

        # Calculate hash and load
        self._dict_data = {}
        self._hash, self._hash_no_trial = self.calculate_hash()
        if not self.recalculate:
            self._load_data()

        # Choose a model to fit
        if not self.dry_run:
            if model == 'free_same_D':
                self._ln_likelihood_with_link = get_ln_likelihood_func_free_hookean_with_link_same_D(
                    ks=self.ks, M=self.M, dt=self.dt, dRks=self.dRks)
                self._ln_likelihood_no_link = get_ln_likelihood_func_free_hookean_no_link_same_D(
                    ks=self.ks, M=self.M, dt=self.dt, dRks=self.dRks)
                self._names_with_link = ('D1', 'n12', 'L0')
                self._names_no_link = ('D1',)
            elif model == 'free_different_D':
                1
            elif model == 'localized_same_D_detect_angle':
                1
            elif model == 'localized_different_D_detect_angle':
                self._ln_likelihood_with_link = get_ln_likelihood_func_2_particles_x_link(
                    ks=self.ks, M=self.M, dt=self.dt, dRks=self.dRks, rotation=True)
                self._ln_likelihood_no_link = get_ln_likelihood_func_no_link(
                    ks=self.ks, M=self.M, dt=self.dt, dRks=self.dRks)
                self._names_with_link = ('D1', 'D2', 'n1', 'n2', 'n12', 'alpha')
                self._names_no_link = ('D1', 'n1')

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
        if self._ln_model_evidence_with_link is None:
            # check loaded
            if '_ln_model_evidence_with_link' in self._dict_data:
                self._ln_model_evidence_with_link = self._dict_data['_ln_model_evidence_with_link']
                return self._ln_model_evidence_with_link

            # calculate
            if self.dry_run:
                self._ln_model_evidence_with_link = np.nan
            else:
                # print(self.model)
                self._MLE_link, self._ln_model_evidence_with_link, min, success = \
                    self._calculate_MLE(link=True, names=self._names_with_link)
                # model_results =
                self._dict_data.update({'_MLE_link': self._MLE_link,
                                        '_ln_model_evidence_with_link':
                                            self._ln_model_evidence_with_link})
                self._save_data()
        return self._ln_model_evidence_with_link

    @property
    def MLE_link(self):
        if self._MLE_link is None:
            # check loaded
            if '_MLE_link' in self._dict_data:
                self._MLE_link = self._dict_data['_MLE_link']
                return self._MLE_link

            # calculate
            if self.dry_run:
                self._MLE_link = {}
            else:
                self.ln_model_evidence_with_link
        return self._MLE_link

    @property
    def ln_model_evidence_no_link(self):
        if self._ln_model_evidence_no_link is None:
            # check loaded
            if '_ln_model_evidence_no_link' in self._dict_data:
                self._ln_model_evidence_no_link = self._dict_data['_ln_model_evidence_no_link']
                return self._ln_model_evidence_no_link

            # calculate
            if self.dry_run:
                self._ln_model_evidence_no_link = np.nan
            else:
                self._MLE_no_link, self._ln_model_evidence_no_link, min, success = \
                    self._calculate_MLE(link=False, names=self._names_no_link)
                self._dict_data.update({'_MLE_no_link': self._MLE_no_link,
                                        '_ln_model_evidence_no_link':
                                            self._ln_model_evidence_no_link})
                self._save_data()
        return self._ln_model_evidence_no_link

    @property
    def MLE_no_link(self):
        if self._MLE_no_link is None:
            # check loaded
            if '_MLE_no_link' in self._dict_data:
                self._MLE_no_link = self._dict_data['_MLE_no_link']
                return self._MLE_no_link

            # calculate
            if self.dry_run:
                self._MLE_no_link = {}
            else:
                self.ln_model_evidence_no_link
        return self._MLE_no_link

    @property
    def lgB(self):
        if self._lgB is None:
            # check loaded
            if '_lgB' in self._dict_data:
                self._lgB = self._dict_data['_lgB']
                return self._lgB

            # calculate
            self._calculate_bayes_factor()
            self._dict_data.update({'_lgB': self._lgB})
            self._save_data()

        return self._lgB

    @property
    def simulation_time(self):
        if np.isnan(self._simulation_time) and '_simulation_time' in self._dict_data:
            self._simulation_time = self._dict_data['_simulation_time']

        return self._simulation_time

    def calculate_hash(self):
        return hash_from_dictionary(parameters=self.parameters, dim=self.dim)

    def _load_data(self):
        self._dict_data, _ = load_data(self._hash)

    def _save_data(self):
        # self._dict_data, _ = load_data(self._hash)
        if not self.dry_run:
            save_data(dict_data=self._dict_data, hash=self._hash)

    def _simulate(self):
        # Check if can load
        if not self.recalculate:
            self._load_data()
            if np.all([key in self._dict_data for key in 't R dR'.split()]):
                self._t, self._R, self._dR = [self._dict_data[key] for key in 't R dR'.split()]
                # if '_simulation_time'in self._dict_data:
                #     self._simulation_time = self._dict_data['_simulation_time']
                return

        # Calculate
        start = time.time()
        self._t, self._R, self._dR, _ = self.simulation_function(
            parameters=self.parameters, plot=self.plot, show=False)
        self._simulation_time = time.time() - start

        # Save
        self._dict_data = {'t': self._t, 'R': self._R, 'dR': self._dR, '_simulation_time':
            self._simulation_time, **self.parameters}
        self._save_data()
        # save_data(dict_data=dict_data, hash=self._hash)

    def _calculate_DFT(self):
        # Calculate
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
                return self._ln_prior(**kwargs) + self._ln_likelihood_with_link(**kwargs)
        return _ln_posterior

    def _calculate_MLE(self, link, names):
        # names = self._names_with_link if link else self._names_no_link
        return get_MLE(
            ln_posterior=self.get_ln_posterior(link), names=names,
            sample_from_the_prior=self._sample_from_the_prior, hash_no_trial=self._hash_no_trial,
            link=link)

    def _calculate_bayes_factor(self):
        self._lgB = (self.ln_model_evidence_with_link - self.ln_model_evidence_no_link) / np.log(10)
        if not self.dry_run:
            print(f'Bayes factor: {self._lgB}')




# Tests
if __name__ == '__main__':
    # test_traj = Trajectory(n1=0, n2=0, M=100)
    test_traj = Trajectory.from_parameter_dictionary({'D1': 0.4,
                                                      'D2': 0.4,
                                                      'n1': 0,
                                                      'n2': 0,
                                                      'n12': 2e-4 / 0.05,
                                                      'L0': 100 * np.sqrt(4 * 0.4 * 0.05),
                                                      'M': 1000,
                                                      'model': 'free_same_D',
                                                      'recalculate': 1,
                                                      'plot': 1,
                                                      'dry_run': 0,
                                                      'trial': 0})
    # print(test_traj.dRks)
    # print(test_traj.ks)
    # print('link = True: ', test_traj.ln_posterior(link=True, D1=1, n12=10, L0=1))
    # print('link = False: ', test_traj.ln_posterior(link=False, D1=1))

    # print('\nEvidence without link: ', test_traj.ln_model_evidence_no_link, 'MLE: ',
    #       test_traj.MLE_no_link)
    # print('\nEvidence with link: ', test_traj.ln_model_evidence_with_link, 'MLE: ',
    #       test_traj.MLE_link)
    # print('\nLg Bayes factor for link: ', test_traj.lgB)
    print('\nSimulation time: ', test_traj.simulation_time)

