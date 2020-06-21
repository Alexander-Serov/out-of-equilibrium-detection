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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from filelock import FileLock, Timeout
from scipy.optimize import root, root_scalar
from scipy.special import gamma
from pickle import UnpicklingError
import warnings
from pathlib import Path
from numpy import arctan, pi

# from calculate import max_abs_lg_B_per_M

hostname = socket.gethostname()
data_folder = r'D:\calculated_data\out-of-equilibrium_detection'
if hostname == 'onsager-dbc' or Path(data_folder).exists():
    pass
else:
    data_folder = 'data'

MLE_guess_file = 'MLE_guesses.pyc'
stat_filename = 'statistics.dat'

LOCK_TIMEOUT = 3  # s


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
        rotated[:, j] = RM @ dR[:, j + lag]
    return rotated


def get_cluster_args_string(D1, D2, n1, n2, n12, dt, L0, M, model, trial=0,
                            recalculate_trajectory=False, recalculate_BF=False, verbose=False,
                            rotation=True, angle=None, gamma=None, **kwargs):
    args_string = '--D1={D1:g} --D2={D2:g} --n1={n1:g} --n2={n2:g} --n12={n12:g}  --dt={dt:g} ' \
                  '--L={L0:g} --trial={trial:d} --M={M} --model={model}'.format(
        D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, dt=dt, L0=L0, trial=trial,
        M=M, model=model)
    if recalculate_trajectory:
        args_string += ' --recalculate_trajectory'
    if recalculate_BF:
        args_string += ' --recalculate_BF'
    if verbose:
        args_string += ' --verbose'
    if rotation:
        args_string += ' --rotation'
    if angle is not None:
        args_string += f' --angle={angle:g}'
    args_string += '\n'
    return args_string


def _hash_me(str):
    """
    For external use, call hash_from_dictionary()
    """
    # hash_str = ''
    # for i, arg in enumerate(args):
    #     # print(i, arg)
    #     if i > 0:
    #         hash_str += '_'
    #     if isinstance(arg, str):
    #         hash_str += arg
    #     else:
    #         hash_str += f'{arg:g}'

    hash = hashlib.md5(str.encode('utf-8'))
    return hash.hexdigest()


def hash_from_dictionary(parameters, dim=2, use_model=False):
    """
    Keeping `use_model` for compatibility with old calculated data. Remove when not necessary.
    """
    args_dict = parameters.copy()
    args_dict['dim'] = dim
    if not use_model:
        args_dict.pop('model', None)

    args_dict_no_trial = args_dict.copy()
    args_dict_no_trial['trial'] = 0

    return _hash_me(str(args_dict)), _hash_me(str(args_dict_no_trial))


def load_data(hash):
    """Load a data dictionary from a pickle file"""
    filename = 'data_' + hash + '.pyc'
    filepath = os.path.join(data_folder, filename)
    dict_data, loaded = {}, False

    try:
        with open(filepath, 'rb') as file:
            dict_data = pickle.load(file)
        if isinstance(dict_data, dict):
            loaded = True
    except EOFError as e:
        print('Encountered incomplete file\n', e)
    except FileNotFoundError:
        pass
    except Exception as e:
        print('Unhandled exception while reading a data file', e)

    return dict_data, loaded


def delete_data(hash):
    """Delete a pickle file"""
    filename = os.path.join(data_folder, 'data_' + hash + '.pyc')
    try:
        os.unlink(filename)
        print(f'Deleted hash {hash}')
    except:
        pass
    return


def save_data(dict_data, hash):
    """Write dict_data to a pickle file"""

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    filename = os.path.join(data_folder, 'data_' + hash + '.pyc')
    try:
        with open(filename, 'wb') as file:
            pickle.dump(dict_data, file, pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as e:
        logging.warning("Enoucntered unhandled exception while saving a data file: ", e)
        return False


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
        mins = int(np.floor(delta / 60))
        secs = delta - 60 * mins
        if self.verbose:
            print(f'\n{self.name} completed in {mins} mins {round(secs, 0)} s.\n')


def stopwatch_dec(func):
    """An alternative decorator for measuring the elapsed time."""

    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        delta = time.time() - start
        print(f'\n{func.__name__} completed in {round(delta, 1)} s.\n')
        return results

    return wrapper


def save_MLE_guess(hash_no_trial, MLE_guess, ln_posterior_value, link, force_update=False):
    """
    For a faster MLE search on the next occasion, save the previous result in a separate file. Naturally, this guess is the same across all trials.

    Unless force_update, update the guess only if the found maximum is higher

    Parameters:
    hash_no_trial - set to None if do not want to save
    ln_posterior_value - natural logarithm of the postive-sign posterior
    link {bool} - whether the system has a link
    """
    success = True
    filename = MLE_guess_file
    lock_file = filename + '.lock'
    lock = FileLock(lock_file, timeout=LOCK_TIMEOUT)

    if hash_no_trial is None:
        return False
    hash_no_trial += f'{link:b}'

    try:
        with lock:
            # Load guesses
            try:
                with open(filename, 'rb') as file:
                    MLE_guesses = pickle.load(file)
            except FileNotFoundError:
                pass

            if 'MLE_guesses' not in locals() or not isinstance(MLE_guesses, dict):
                MLE_guesses = {}

            # Update value if lower
            if hash_no_trial in MLE_guesses.keys():
                old_ln_value = MLE_guesses[hash_no_trial][1]
            else:
                old_ln_value = -np.inf

            if old_ln_value < ln_posterior_value or force_update:
                MLE_guesses[hash_no_trial] = (MLE_guess, ln_posterior_value)

            # Save to file
            temp_filename = filename + '_tmp'
            with open(temp_filename, 'wb') as file:
                pickle.dump(MLE_guesses, file, pickle.HIGHEST_PROTOCOL)
            try:
                os.unlink(filename)
            except FileNotFoundError:
                pass
            os.rename(temp_filename, filename)


        print('Saved MLE guess updated successfully.')
        return True

    except Timeout:
        logging.warning("MLE guess file is locked by another instance. Skipping MLE guess save")

    except Exception as e:
        logging.warning('MLE guess save failed for unknown reason: ', e)

    return False

    # if os.path.exists(filename):
    #     try:
    #         with lock:
    #             with open(filename, 'rb') as file:
    #                 MLE_guesses = pickle.load(file)
    #                 if isinstance(MLE_guesses, dict):
    #                     success = True
    #     except:
    #         logging.warning('The MLE guess file exists, but unable to load the MLE guesses')
    #         pass
    # else:
    #     success = 'file_not_found'

    # if success_load is 'file_not_found':
    #     MLE_guesses = {}
    # elif success_load == False:
    #     logging.warning('Loading from the MLE guess file failed. Aborting save operation')
    #     return False
    #
    # if hash_no_trial in MLE_guesses.keys():
    #     old_ln_value = MLE_guesses[hash_no_trial][1]
    # else:
    #     old_ln_value = -np.inf
    #
    # if old_ln_value < ln_posterior_value or force_update:
    #     MLE_guesses[hash_no_trial] = (MLE_guess, ln_posterior_value)
    #
    #     # Save to file
    #     try:
    #         with lock:
    #             with open(filename, 'wb') as file:
    #                 pickle.dump(MLE_guesses, file, pickle.HIGHEST_PROTOCOL)
    #     except:
    #         logging.warning('Unable to save MLE guess')
    #         success = False
    #         pass
    # else:
    #     success = False
    #
    # if success:
    #     print('Saved MLE guess updated: ', MLE_guess)
    #     print(
    #         f'Log max value of the MLE guess increased from {old_ln_value:.3g} to {ln_posterior_value:.3g}')
    #
    # return success


def load_all_MLE_guesses():
    """
    Load all MLE guesses
    """
    success = False
    MLE_guesses = {}
    filename = MLE_guess_file
    lock_file = filename + '.lock'
    lock = FileLock(lock_file, timeout=LOCK_TIMEOUT)

    try:
        with lock:
            try:
                with open(filename, 'rb') as file:
                    MLE_guesses = pickle.load(file)
            except FileNotFoundError:
                pass
            except IOError as e:
                logging.warning("MLE guess file does not exist: ", e)
            except EOFError:
                logging.warning("Found a corrupt MLE guess file. The file will be rest.")
                os.unlink(filename)
            except Exception as e:
                logging.warning('Unhandled exception while loading MLE guesses: ', e)
            if isinstance(MLE_guesses, dict):
                success = True

    except Timeout:
        logging.warning("MLE guess file is in use by another application. Skipping MLE guess load.")

    return MLE_guesses, success


def load_MLE_guess(hash_no_trial, link):
    """
    Load an MLE guess for the given hash.

    The hash must be generated with trial = 0.
    """
    MLE_guess = np.nan
    old_ln_value = np.nan

    if hash_no_trial is None:
        success = False
        logging.warning('Empty hash provided. MLE guesses will not be loaded or saved')
        return MLE_guess, old_ln_value, success

    hash_no_trial += f'{link:b}'

    MLE_guesses, success = load_all_MLE_guesses()

    if success:
        if hash_no_trial in MLE_guesses.keys():
            MLE_guess, old_ln_value = MLE_guesses[hash_no_trial]
            check_positive = np.all([MLE_guess[key] >= 0 for key in MLE_guess.keys()])
            success = check_positive

        else:
            success = False

    # if success:
    #     print('MLE guess loaded successfully: ', MLE_guess)

    return MLE_guess, old_ln_value, success


def save_number_of_close_values(link, val, tries, frac):
    """
    """

    lock_file = stat_filename + '.lock'
    lock = FileLock(lock_file, timeout=LOCK_TIMEOUT)

    if not os.path.exists(stat_filename):
        with lock:
            with open(stat_filename, 'w') as file:
                strng = f'Link presence\tNo of similar function values in MLE search results\tTries\tFraction\n'
                file.write(strng)

    # Save to file
    try:
        with lock:
            with open(stat_filename, 'a') as file:
                strng = f'{link:d}\t{val:d}\t{tries:d}\t{frac:.3f}\n'
                file.write(strng)
    except:
        logging.warn('Unable to the number of close values')

    return


def set_figure_size(num, rows, page_width_frac, height_factor=1.0, clear=True):
    pagewidth_in = 6.85
    font_size = 8
    dpi = 100

    figsize = np.asarray([1.0, rows *
                          height_factor]) * page_width_frac * pagewidth_in  # in inches

    # Set default font size
    matplotlib.rcParams.update({'font.size': font_size})

    # Enable LaTeX and set font to Helvetica
    plt.rc('text', usetex=True)
    plt.rcParams['text.latex.preamble'] = [
        r'\usepackage{tgheros}',  # helvetica font
        r'\usepackage{sansmath}',  # math-font matching  helvetica
        r'\sansmath'  # actually tell tex to use it!
        r'\usepackage{siunitx}',  # micro symbols
        r'\sisetup{detect-all}',  # force siunitx to use the fonts
    ]
    # Enforce TrueType fonts for easier editing later on
    # matplotlib.rcParams['pdf.fonttype'] = 42
    # matplotlib.rcParams['ps.fonttype'] = 42

    # Create and return figure handle
    fig = plt.figure(num, clear=clear)

    # Set figure size and dpi
    fig.set_dpi(dpi)
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])

    # fig.tight_layout()

    # Return figure handle
    return (fig, figsize)


def find_inverse_gamma_parameters(interval, tau):
    """
    The function finds the parameters alpha and beta of the inverse gamma function such that at the borders of the given interval, the pdf is corresponds to tau \in [0,1] of the maximum
    """
    if tau < 0 or tau > 1:
        raise RuntimeError('Wrong tau value supplied')

    def eqn(alpha, beta, x):
        return -(alpha + 1) * np.log(x * (alpha + 1) / beta) - beta / x + (alpha + 1) - np.log(tau)

    def solve_me(args):
        # print(args)
        alpha, beta = args
        return [eqn(alpha, beta, interval[0]), eqn(alpha, beta, interval[1])]

    # Root search
    guess = [0.5, np.mean(interval)]
    # print(guess)
    sol = root(solve_me, guess)
    print(sol)


def find_inverse_gamma_function_scale(left, tau, alpha):
    """
    The function finds the value of beta that makes the given inverse gamma function with the the given alpha satisfy the condition that at a point x = left to the left of the mode, the function value is tau * max_value.
    """
    if tau < 0 or tau > 1:
        raise RuntimeError('Wrong tau value supplied')

    def eqn(alpha, beta, x):
        return -(alpha + 1) * np.log(x * (alpha + 1) / beta) - beta / x + (alpha + 1) - np.log(tau)

    def solve_me(beta):
        # print(args)
        return eqn(alpha, beta, left)

    # Root search
    guess = 2 * left
    # print(guess)
    sol = root_scalar(solve_me, x0=guess, bracket=[left, 1e10])
    if sol.converged:
        return sol.root
    else:
        return np.nan


def calculate_min_number_of_tries_with_a_binomial_model(stat_filename=stat_filename):
    """
    The function reads and analyzed the statistics.dat file.
    We assume a binomial model of finding or not finding the maximum in cases where several outcomes are possible (p<1).
    Cases with all the same values (p=1) were filtered out.
    """
    conf_level = 0.99

    if not os.path.exists(stat_filename):
        print('{} not found!'.format(stat_filename))
        return 1

    stats = pd.read_csv(stat_filename, sep='\t')
    # print(stats)
    for link in [0, 1]:
        filtered = stats[(stats['Link presence'] == link) & (stats['Fraction'] < 1)].sum(axis=0)
        if len(filtered) > 0:
            p = filtered['No of similar function values in MLE search results'] / filtered['Tries']
            min_N = np.log(1 - conf_level) / np.log(1 - p)
            print(
                f'Minimum number of tries to find a minimum with {conf_level * 100:.0f}% with link={link} is {min_N:.1f}')
        else:
            print(f'No data satisfying the criteria for link={link}')
    return 0


# %% Test
if __name__ == "__main__":
    find_inverse_gamma_function_scale(0.01, 0.01, 2)


    def test(beta=0.11683992564378747, alpha=2, x=0.01):
        return beta ** alpha / gamma(alpha) * x ** (-alpha - 1) * np.exp(-beta / x)


    test(x=0.01)
    test(x=0.11683992564378747 / 3)

    test(x=0.01) / test(x=0.11683992564378747 / (2 + 1))

    # test(0.0889, 0.5, 0.1)
    # test(0.0889, 0.5, 10)
    # test(0.0889, 0.5, 0.5 / 1.0889)
    #
    # test(0.0889, 0.5, 10) / test(0.0889, 0.5, 0.5 / 1.0889)
