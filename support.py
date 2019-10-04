"""
Support functions for the calculations that do not rely on a certain structure of the data
"""

import copy
import hashlib
import logging
import os
import pickle
import socket
import time

import numpy as np
from filelock import FileLock

# from calculate import max_abs_lg_B_per_M

hostname = socket.gethostname()
if hostname == 'onsager-dbc':
    data_folder = r'D:\calculated_data\out-of-equilibrium_detection'
else:
    data_folder = 'data'

MLE_guess_file = 'MLE_guesses.pyc'


def get_rotation_matrix(dr):
    """Return a rotation matrix that rotates the given vector to be aligned with the positive direction of x axis"""
    tan_theta = - dr[1] / dr[0] if dr[0] != 0 else np.inf * (-dr[1])
    if dr[0] >= 0:
        theta = arctan(tan_theta)
    else:
        theta = arctan(tan_theta) + pi
    # print(theta / pi)
    # theta = 0
    R = np.array([[cos(theta), - sin(theta)], [sin(theta), cos(theta)]])
    return R


def locally_rotate_a_vector(dR, lag):
    """Create a vector, such that its component at position j+lag were defined by a rotation matrix for position j, which aligns the vector at position j with the axis x"""
    N = dR.shape[1]

    rotated = np.full_like(dR, np.nan)
    for j in range(N - lag):
        dr = dR[:, j]
        RM = get_rotation_matrix(dr)
        rotated[:, j] = RM @  dR[:, j + lag]
    return rotated


def _hash_me(*args):
    """
    Calculate a hash of the parameters to use as a unique key for saving or loading data.

    Only accepts single-value parameters. Does not accept arrays or lists
    For external use, please call hash_from_dictionary()
    """
    hash_str = ''
    for i, arg in enumerate(args):
        # print(i, arg)
        if i > 0:
            hash_str += '_'
        if isinstance(arg, str):
            hash_str += arg
        else:
            hash_str += f'{arg:e}'

    hash = hashlib.md5(hash_str.encode('utf-8'))
    return hash.hexdigest()


def hash_from_dictionary(true_parameters, dim=2):
    args = [true_parameters[key] for key in 'D1 D2 n1 n2 n12 T dt angle L M'.split()]
    args_no_trial = copy.deepcopy(args)

    # Allow to have multiple hashes for the same parameters
    if 'trial' in true_parameters.keys():
        trial = true_parameters['trial']
    else:
        trial = 0
    args.append(trial)
    args_no_trial.append(0)

    str = f'{dim:d}D'
    return _hash_me(str, *args), _hash_me(str, *args_no_trial)


def load_data(hash):
    """Load a data dictionary from a pickle file"""
    filename = os.path.join(data_folder, 'data_' + hash + '.pyc')
    dict_data = {}
    loaded = False
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            try:
                dict_data = pickle.load(file)
                if isinstance(dict_data, dict):
                    loaded = True
            except:
                pass

    return dict_data, loaded


def delete_data(hash):
    """Delete a pickle file"""
    filename = os.path.join(data_folder, 'data_' + hash + '.pyc')
    if os.path.exists(filename):
        os.unlink(filename)
    return


def save_data(dict_data, hash):
    """Write dict_data to a pickle file"""
    # Check if folder exists
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # Save to file
    filename = os.path.join(data_folder, 'data_' + hash + '.pyc')
    success = False
    try:
        with open(filename, 'wb') as file:
            pickle.dump(dict_data, file, pickle.HIGHEST_PROTOCOL)
        success = True
    except:
        pass

    return success


class stopwatch:
    """
    A class for measuring execution time.

    Use it by launching a function of interest in the block of
        with stopwatch():
            function()

    """

    def __init__(self, name="Execution", verbose=True):
        self.name = name
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        delta = self.end - self.start
        if self.verbose:
            print(f'\n{self.name} completed in {round(delta, 2)} s.\n')


def stopwatch_dec(func):
    """An alternative decorator for measuring the elapsed time."""

    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        delta = time.time() - start
        print(f'\n{func.__name__} completed in {round(delta, 1)} s.\n')
        return results
    return wrapper


def get_cluster_args_string(D1, D2, n1, n2, n12, gamma, T, dt, angle, L, M, trial=0, recalculate=False, verbose=False):
    args_string = '--D1={D1:g} --D2={D2:g} --n1={n1:g} --n2={n2:g} --n12={n12:g} --gamma={gamma:g} --T={T:g} --dt={dt:g} --angle={angle:f} --L={L:f} --trial={trial:d} --M={M}'.format(
        D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, T=T, dt=dt, angle=angle, L=L, trial=trial, M=M)
    if recalculate:
        args_string += ' --recalculate'
    if verbose:
        args_string += ' --verbose'
    args_string += '\n'
    return args_string


def save_MLE_guess(hash_no_trial, MLE_guess, ln_posterior_value, link, force_update=False):
    """
    For a faster MLE search on the next occasion, save the previous result in a separate file. Naturally, this guess is the same across all trials.

    Unless force_update, update the guess only if the found maximum is higher

    Parameters:
    ln_posterior_value - natural logarithm of the postive-sign posterior
    link {bool} - whether the system has a link
    """
    success = True
    filename = MLE_guess_file
    lock_file = filename + '.lock'
    lock = FileLock(lock_file, timeout=1)

    hash_no_trial += f'{link:b}'

    # Load current guesses
    MLE_guesses, success_load = load_all_MLE_guesses()
    if success_load is 'file_not_found':
        MLE_guesses = {}
    elif success_load == False:
        logging.warn('Unable to load MLE guesses for an unknown reason. Aborting save operation')
        return False

    if hash_no_trial in MLE_guesses.keys():
        old_ln_value = MLE_guesses[hash_no_trial][1]
    else:
        old_ln_value = -np.inf

    if old_ln_value < ln_posterior_value or force_update:
        MLE_guesses[hash_no_trial] = (MLE_guess, ln_posterior_value)

        # Save to file
        try:
            with lock:
                with open(filename, 'wb') as file:
                    pickle.dump(MLE_guesses, file, pickle.HIGHEST_PROTOCOL)
        except:
            logging.warn('Unable to save MLE guess')
            success = False
            pass
    else:
        success = False

    if success:
        print('Saved MLE guess updated: ', MLE_guess)
        print(
            f'Log max value of the MLE guess increased from {old_ln_value:.3g} to {ln_posterior_value:.3g}')

    return success


def load_all_MLE_guesses():
    """
    Load all MLE guesses
    """
    success = False
    MLE_guesses = {}
    filename = MLE_guess_file
    lock_file = filename + '.lock'
    lock = FileLock(lock_file, timeout=1)

    if os.path.exists(filename):
        try:
            with lock:
                with open(filename, 'rb') as file:
                    MLE_guesses = pickle.load(file)
                    if isinstance(MLE_guesses, dict):
                        success = True
        except:
            logging.warn('The MLE guess file exists, but unable to load the MLE guesses')
            pass
    else:
        success = 'file_not_found'

    return MLE_guesses, success


def load_MLE_guess(hash_no_trial, link):
    """
    Load an MLE guess for the given hash.

    The hash must be generated with trial = 0.
    """
    MLE_guess = np.nan
    old_ln_value = np.nan
    hash_no_trial += f'{link:b}'

    MLE_guesses, success = load_all_MLE_guesses()

    if success:
        if hash_no_trial in MLE_guesses.keys():
            MLE_guess, old_ln_value = MLE_guesses[hash_no_trial]
        else:
            success = False

    # if success:
    #     print('MLE guess loaded successfully: ', MLE_guess)

    return MLE_guess, old_ln_value, success
