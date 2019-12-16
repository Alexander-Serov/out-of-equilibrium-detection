"""
Define 'trajectory' class that will manage simulation and fitting of the trajectories.
"""
from support import hash_from_dictionary, load_data, save_data
import numpy as np
from simulate import simulate_2_confined_particles_with_fixed_angle_bond, \
    simulate_a_free_hookean_dumbbell
from scipy.fftpack import fft, ifft


class Trajectory:
    def __init__(self, D1=2.0, D2=0.4, n1=1.0, n2=1.0, n12=30.0, M=1000, dt=0.05, L0=10.0,
                 trial=0, angle=-np.pi / 3,
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
        self._hash, self._hash_no_trial = self.calculate_hash()

        if self.n1 == 0 and self.n2 == 0:
            self.simulation_function = simulate_a_free_hookean_dumbbell
        else:
            self.simulation_function = simulate_2_confined_particles_with_fixed_angle_bond

        self.recalculate = False

        # Calculated variables
        self._t, self._R, self._dR, self._dRk, self._ks, self._lgB = [None] * 6

        # Choose a model to fit
        if model == 'free_same_D':
            1
        elif model == 'free_different_D':
            1
        elif model == 'localized_same_D':
            self._prior = 1
            self._likelihood = 1
        elif model == 'localized_different_D':
            1

    # Another constructor
    @classmethod
    def from_parameter_dictionary(cls, parameters):
        return cls(**parameters)

    @property
    def t(self):
        if self._t is None:
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
    def dRk(self):
        if self._dRk is None:
            self._calculate_DFT()
        return self._dRk

    @property
    def ks(self):
        if self._ks is None:
            self._calculate_DFT()
        return self._ks

    @property
    def lgB(self):
        if self._lgB is None:
            self._calculate_bayes_factor()
        return self._lgB


    def calculate_hash(self):
        return hash_from_dictionary(parameters=self.parameters, dim=self.dim)

    def _simulate(self):
        # Check if can load
        if not self.recalculate:
            dict_data, loaded = load_data(self._hash)
            if loaded:
                self._t, self._R, self._dR = [dict_data[key] for key in 't R dR'.split()]
                return
        # Calculate
        self._t, self._R, self._dR, _ = self.simulation_function(
            parameters=self.parameters)
        # Save
        dict_data = {'t': self._t, 'R': self._R, 'dR': self._dR, **self.parameters}
        save_data(dict_data=dict_data, hash=self._hash)

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

    def _calculate_bayes_factor(self):
        # Check load

        # Calculate

        # Save results





# Tests
if __name__ == '__main__':
    test_traj = Trajectory(n1=0, n2=0, M=100)
    test_traj = Trajectory.from_parameter_dictionary({'n1': 0, 'n2': 0, 'M': 100})
    print(test_traj.dRk)
    print(test_traj.ks)

# class lazyproperty:
#     def __init__(self, func):
#         self.func=func
#
#     def __get__(self, instance, cls):
#         if instance is None:
#             return self
#         else:
#             value = self.func(instance)
#             setattr(instance, self.func.__name__,value)
#             return value
