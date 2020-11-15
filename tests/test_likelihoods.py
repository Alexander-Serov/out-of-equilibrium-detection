import copy
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from likelihood import (
    get_ln_likelihood_func_2_particles_x_link,
    new_likelihood_2_particles_x_link_one_point,
)

# Load sample data
energy_transfer_data_file = Path("tests/data/data-energy-transfer-example.json")
with open(energy_transfer_data_file) as f:
    energy_transfer_data = json.load(f)

# Convert complex numbers and convert to numpy arrays
def to_complex(x):
    if isinstance(x, Iterable) and not isinstance(x, str):
        return [to_complex(xi) for xi in x]
    else:
        return complex(x)


energy_transfer_data["dRk"] = np.array(to_complex(energy_transfer_data["dRk"]))
energy_transfer_data["k"] = np.array(energy_transfer_data["k"])

ATOL_LOG = 1e-5


def test_new_likelihood_2_particles_x_link_one_point():
    """Get one-component likelihood."""

    # -
    # Test 1
    ind = 1
    d = energy_transfer_data  # shortcut
    true_ln_likelihood = 17.44501495545533

    ln_estimate = new_likelihood_2_particles_x_link_one_point(
        dRk=tuple(d["dRk"][:, ind]),
        k=d["k"][ind],
        D1=d["D1"],
        D2=d["D2"],
        n1=d["n1"],
        n2=d["n2"],
        n12=d["n12"],
        M=d["M"],
        dt=d["dt"],
        alpha=d["alpha"],
    )
    assert np.isclose(ln_estimate, true_ln_likelihood, atol=ATOL_LOG), (
        f"One likelihood point test failed: {ln_estimate:.6g}"
        f" != {true_ln_likelihood:.6g}"
    )


def test_get_ln_likelihood_func_2_particles_x_link():
    """Get full likelihood of a model on a trajectory."""
    d = copy.copy(energy_transfer_data)  # shortcut

    # -
    # Energy transfer case
    expected_ln_likelihood = 5348.715008981629

    ln_likelihood_func = get_ln_likelihood_func_2_particles_x_link(
        ks=d["k"], M=d["M"], dt=d["dt"], dRks=d["dRk"]
    )
    ln_likelihood = ln_likelihood_func(
        D1=d["D1"],
        D2=d["D2"],
        n1=d["n1"],
        n2=d["n2"],
        n12=d["n12"],
        alpha=d["alpha"],
    )
    assert np.isclose(ln_likelihood, expected_ln_likelihood, atol=ATOL_LOG)

    # Same D case
    expected_ln_likelihood = 5348.834242269174

    ln_likelihood_func = get_ln_likelihood_func_2_particles_x_link(
        ks=d["k"],
        M=d["M"],
        dt=d["dt"],
        dRks=d["dRk"],
        same_D=True,
    )
    ln_likelihood = ln_likelihood_func(
        D1=d["D1"],
        n1=d["n1"],
        n2=d["n2"],
        n12=d["n12"],
        alpha=d["alpha"],
    )
    assert np.isclose(ln_likelihood, expected_ln_likelihood, atol=ATOL_LOG)

    # -
    # Speed test
    start = time.perf_counter()
    N = 29
    for n1 in np.logspace(
        np.log10(0.21594704084413527), np.log10(21.594704084413527), N
    ):
        d.update({"n1": n1})
        # -
        # Energy transfer case
        ln_likelihood_func = get_ln_likelihood_func_2_particles_x_link(
            ks=d["k"], M=d["M"], dt=d["dt"], dRks=d["dRk"]
        )
        ln_likelihood_func(
            D1=d["D1"],
            D2=d["D2"],
            n1=d["n1"],
            n2=d["n2"],
            n12=d["n12"],
            alpha=d["alpha"],
        )

        # Same D case
        ln_likelihood_func = get_ln_likelihood_func_2_particles_x_link(
            ks=d["k"],
            M=d["M"],
            dt=d["dt"],
            dRks=d["dRk"],
            same_D=True,
        )
        ln_likelihood_func(
            D1=d["D1"],
            n1=d["n1"],
            n2=d["n2"],
            n12=d["n12"],
            alpha=d["alpha"],
        )
    print(f"{N} likelihood calculations completed in {time.perf_counter() -start} s.")


test_get_ln_likelihood_func_2_particles_x_link()

# import unittest
#
# import numpy as np
# from matplotlib import pyplot as plt
# from numpy import cos, exp, pi
#
# from constants import color_sequence
# from likelihood import (get_ln_likelihood_func_2_particles_x_link,
#                         get_ln_prior_func, get_MLE, get_sigma2_matrix_func,
#                         likelihood_2_particles_x_link_one_point)

# class tests(unittest.TestCase):

# def test_likelihood_one_point(self):
#     """
#     Check if likelihoods are correctly calculated for different parameter values
#     """
#     k = 1
#     D1 = 1
#     D2 = 3
#     n1 = 1
#     n2 = 1
#     n12 = 1
#     M = 999
#     dt = 0.3
#     alpha = 0
#     atol = 1e-8
#     atol_log = 1e-3
#
#     """Test 1"""
#     z1 = 1e-4
#     true_ln_likelihood = 8.317699879
#     ln_estimate = likelihood_2_particles_x_link_one_point(
#         z=z1, k=k, D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M, dt=dt, alpha=alpha)
#
#     self.assertTrue(np.isclose(ln_estimate, true_ln_likelihood, atol=atol_log),
#                     f"One likelihood point test failed: {ln_estimate:.6g} != {true_ln_likelihood:.6g}")
#
#     """Test 2"""
#     z2 = 1e-1
#     true_ln_likelihood = -2786.649642
#     ln_estimate = likelihood_2_particles_x_link_one_point(
#         z=z2, k=k, D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M, dt=dt, alpha=alpha)
#
#     self.assertTrue(np.isclose(ln_estimate, true_ln_likelihood, atol=atol_log),
#                     f"One likelihood point test failed: {ln_estimate:.6g} != {true_ln_likelihood:.6g}")
#
#     """Test alpha = 1. This gives only one chi-squared function"""
#     k = 480
#     z2 = 0.5 * 2 * D1 * dt
#     alpha = 1
#
#     lamb1 = lamb2 = -1
#     J = -(((M * dt ** 2) * ((1 - exp(lamb1 * dt) * exp(lamb2 * dt)) * (1 + exp(lamb1 * dt) * exp(lamb2 * dt) - (exp(lamb1 * dt) + exp(lamb2 * dt)) * cos((2 * pi * k) / M)) * (1 - cos((2 * pi * k) / M)))) /
#           ((lamb1 + lamb2) * ((1 + exp(lamb1 * dt) ** 2 - 2 * exp(lamb1 * dt) * cos((2 * pi * k) / M)) * (1 + exp(lamb2 * dt) ** 2 - 2 * exp(lamb2 * dt) * cos((2 * pi * k) / M)))))
#     sigma2 = 2 * D1 * J / (M * dt)
#
#     true_ln_likelihood = -z2 / sigma2 / 2 - np.log(2) - np.log(sigma2)
#     ln_estimate = likelihood_2_particles_x_link_one_point(
#         z=z2, k=k, D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M, dt=dt, alpha=alpha)
#
#     # print('Likelihoods for the y-component', ln_estimate, true_ln_likelihood, J / M / dt)
#
#     self.assertTrue(np.isclose(ln_estimate, true_ln_likelihood, atol=atol_log),
#                     f"One likelihood point test failed: {ln_estimate:.6g} != {true_ln_likelihood:.6g}")

# def test_likelihood_several_points(self):
#      """Likelihood of the whole spectrum"""

# """Test 1bis"""
# ks_fit = np.arange(1, 50)
# dRks_fit = np.array([
#     [-0.00325137 + 0.00634241j, 0.01291825 - 0.06345991j, -0.05255444 + 0.02498131j, -0.00427763 - 0.00476471j, -0.02037643 + 0.03667096j, 0.01551485 + 0.00427473j, -0.01576897 - 0.01917842j, -0.03774972 - 0.01121879j, -0.04735799 + 0.00180407j, -0.04785612 - 0.04488017j, -0.00115852 + 0.02079864j, 0.04116296 + 0.00912709j, -0.0424548 - 0.01450416j, 0.07892817 - 0.00739656j, -0.00728199 + 0.03489695j, 0.02721687 - 0.00817511j, -0.00545668 - 0.00647667j, 0.03393649 + 0.00308079j, -0.01879607 - 0.01366452j, 0.04266098 + 0.03819161j, -0.01651085 - 0.00277542j, 0.02775057 + 0.00724213j, -0.04529796 - 0.00378615j, -0.03792255 - 0.00875209j, -0.00679845 +
#         0.00170425j, 0.00635937 + 0.00768819j, -0.02923592 + 0.03193154j, -0.00894103 - 0.01809968j, 0.02211621 + 0.02984551j, 0.01491307 + 0.01919253j, 0.0052379 + 0.00660592j, -0.0008194 + 0.04882443j, 0.00294467 + 0.01820085j, 0.0074528 - 0.00197897j, -0.02571783 + 0.00482061j, -0.03785542 - 0.05065658j, -0.00784263 + 0.05176622j, 0.01196266 + 0.02395296j, 0.00544107 + 0.0343064j, -0.00436193 - 0.01326126j, -0.00587801 + 0.00319844j, -0.02982299 + 0.00355881j, 0.01449614 - 0.01481056j, 0.01074832 - 0.01504251j, -0.01427094 - 0.00759009j, -0.0101859 - 0.00755336j, -0.00938588 - 0.0459604j, -0.00660651 + 0.00362191j, -0.01194545 - 0.00665009j],
#     [0.00101988 + 0.00040583j, -0.0153072 - 0.00742598j, -0.02279349 - 0.00223776j, -0.02863213 - 0.00605581j, 0.01919226 + 0.02212472j, -0.01132702 - 0.01676333j, 0.00489911 + 0.01482526j, -0.01260573 - 0.00429939j, 0.00364306 + 0.00143833j, -0.00483868 - 0.01306407j, 0.0093781 - 0.02045633j, 0.00220572 - 0.00485326j, 0.03125984 - 0.00275065j, 0.00837316 - 0.00216993j, 0.01027111 + 0.00505959j, -0.01444157 + 0.00493869j, 0.01201893 - 0.00442634j, 0.00751432 - 0.00466743j, -0.00054497 + 0.01713626j, -0.01433413 - 0.01827707j, 0.0055454 + 0.0107042j, 0.00251942 + 0.00993386j, -0.01801659 - 0.03134878j, -0.00395741 + 0.01434205j, -0.02137869 + 0.01532669j, -0.00739603 + 0.01720868j, -0.00169154 - 0.02156012j, -0.00571027 - 0.01148296j, 0.00087276 + 0.00983763j, -0.01449878 - 0.00216111j, -0.0001313 + 0.00477325j, 0.00334654 + 0.0037378j, 0.00391217 - 0.00601837j, 0.00045049 - 0.00797244j, -0.01540929 + 0.00244681j, 0.01772098 + 0.0184876j, 0.02102886 + 0.00534708j, -0.00243948 - 0.01840921j, -0.00767384 + 0.00072665j, 0.00052799 - 0.00397305j, 0.00837055 + 0.00645161j, 0.00551823 - 0.00763377j, -0.00577698 + 0.01361667j, -0.0094426 - 0.00345006j, -0.0024404 + 0.01639148j, -0.01679862 - 0.00372006j, -0.01832527 - 0.00133619j, -0.00096314 + 0.00138064j, -0.01580905 - 0.01465307j]])
# true_args = '--D1=0.011514 --D2=0.4 --n1=1 --n2=1 --n12=30 --gamma=1e-08 --T=5 --dt=0.05 --angle=0.000000 --L=0.500000 --trial=40 --M=100'
#
# M = 100
# dt = 0.05
#
# # The fitted minimum below has a problem
# fit_min_with_link = {'D1': 0.000345186899194981, 'D2': 0.004168877719576778,
#                      'n1': 0.05104314010002475, 'n2': 0.0019307836145609136, 'n12': 0.1533258905925943}
# # Get likelihood with link
#
# ln_lklh_func = get_ln_likelihood_func_2_particles_x_link(
#     ks=ks_fit, M=M, dt=dt, dRks=dRks_fit)
# ln_prior, sample_from_the_prior = get_ln_prior_func()
# names = ('D1', 'D2', 'n1', 'n2', 'n12')
#
# def minimize_me(args):
#     """-ln posterior to minimize"""
#     args_dict = {a: args[i] for i, a in enumerate(names)}
#     return -ln_lklh_func(**args_dict) - ln_prior(**args_dict)
#
# def to_list(dict):
#     return [dict[key] for key in names]
#
# ln_lklh_func(**fit_min_with_link)
# minimize_me(to_list(fit_min_with_link))
# link = True
#
# # %%
# plt.figure(36 + int(link), clear=True)
#
# def plot_func(xs, dim):
#     out = []
#     for x in xs:
#         args = to_list(fit_min_with_link).copy()
#         args[dim] = x
#         args_dict = {a: args[i] for i, a in enumerate(names)}
#         out.append(minimize_me(args))
#         # out.append(ln_lklh_func(**args_dict))
#     return out
# # step = 0.1
# for dim in range(len(names)):
#     # dim = 4
#     x_d = to_list(fit_min_with_link)[dim]
#     xs = np.linspace(np.min([x_d * 0.1, 0.1]),
#                      np.max([x_d * 20, 1]), num=100, endpoint=True)
#     plt.plot(xs, plot_func(xs, dim), label=names[dim], color=color_sequence[dim])
#     plt.scatter(x_d, plot_func([x_d], dim)[0], color=color_sequence[dim])
#
# plt.legend()
# plt.show(block=False)
# # import time
# # time.sleep(3)
# # t1 = np.complex128(3 + 5j)
# # t1.real

# %%
# 1
# M = 999
# dt = 0.3
# atol = 1e-8
# atol_log = 1e-3
# test_point = (1, 2, 1, 1, 1)    # D1, D2, n1, n2, n12
#
# """Test 1"""
# zs = [1e-4, 2.3e-3]
# ks = [1, 2]
#
# ln_lklh = get_ln_likelihood_func_2_particles_x_link(
#     ks=ks, zs_x=zs, M=M, dt=dt)
# ln_estimate = ln_lklh(*test_point)
# true_ln_estimate = 1.483560874
#
# self.assertTrue(np.isclose(ln_estimate, true_ln_estimate, atol=atol_log),
#                 f"Likelihood test failed for 2 points: {ln_estimate:.6g} != {true_ln_estimate:.6g}")
#
# """Test 2"""
# zs = [1.13e-1, 4.5e-2]
# ks = [101, 114]
#
# ln_lklh = get_ln_likelihood_func_2_particles_x_link(
#     ks=ks, zs_x=zs, M=M, dt=dt)
# ln_estimate = ln_lklh(*test_point)
# true_ln_estimate = 3.264467649
#
# self.assertTrue(np.isclose(ln_estimate, true_ln_estimate, atol=atol_log),
#                 f"Likelihood test failed for 2 points: {ln_estimate:.6g} != {true_ln_estimate:.6g}")

# def test_MLE_search(self):
#
#
#
#     1
#     """
#     Test if the MLE is found correctly for several values of the periodogram
#     """
#     M = 999
#     dt = 0.3
#     alpha = 0
#     atol = 1e-8
#     atol_log = 1e-3
#     test_point = (1, 2, 1, 1, 1)    # D1, D2, n1, n2, n12
#
#     """Test 1"""
#     zs_x = [dt**2, dt**2 * 0.95]
#     ks = [400, 402]
#
#     true_MLE = 1
#     true_fun = 1
#
#     MLE = get_MLE(ks=ks, zs_x=zs_x, M=M, dt=dt, link=True, hash_no_trial=None)
#     # # ln_estimate = ln_lklh(*test_point)
#     # true_MLE = 0.7514517168
#     #
#     # # self.assertTrue(np.isclose(MLE, true_MLE, atol=atol),
#     # #                 f"Likelihood test failed for 2 points: {MLE:.6g} != {true_MLE:.6g}")
#     print(MLE)
#     # print(true_MLE)
# # #
# # #     # """Test 2"""
# # #     # zs = [1.13e-1, 4.5e-2]
# # #     # ks = [101, 114]
# # #     #
# # #     # ln_lklh = get_ln_likelihood_func_2_particles_x_link(
# # #     #     ks=ks, zs=zs, M=M, dt=dt, alpha=alpha)
# # #     # ln_estimate = ln_lklh(*test_point)
# # #     # true_ln_estimate = 3.194565432
# # #     #
# # #     # self.assertTrue(np.isclose(ln_estimate, true_ln_estimate, atol=atol_log),
# # #     #                 f"Likelihood test failed for 2 points: {ln_estimate:.6g} != {true_ln_estimate:.6g}")
# #
#
# def test_mean(self):
#     """I know what is the mean expected value of the stochastic variable if there is no link"""
#     M = 999
#     dt = 0.3
#     alpha = 0
#     atol = 1e-8
#     atol_log = 1e-3
#
#     """Test 1 free particle"""
#     k = 100
#     D1 = 1
#     D2 = 3
#     n1 = 0
#     n2 = 0
#     n12 = 0
#
#     # zs = [1e-4, 2.3e-3]
#     # ks = [1, 2]
#
#     sigma2s_func = get_sigma2_matrix_func(D1, D2, n1, n2, n12, M, dt, alpha)
#     sigma2s = sigma2s_func(alpha, k)
#
#     mean_periodogram = sigma2s.sum() * 2
#     true_value = 2 * D1 * dt**2
#
#     self.assertTrue(np.isclose(mean_periodogram, true_value, atol=atol),
#                     f"Mean test failed for 1 free particle: {mean_periodogram:.6g} != {true_value:.6g}")
#
#     """Test 2 linked non-confined particles with D2/D1 = 1"""
#     k = 10
#     D1 = 1
#     D2 = 1
#     n1 = 0
#     n2 = 0
#     n12 = 1
#     true_value = 0.09100735399
#
#     # zs = [1e-4, 2.3e-3]
#     # ks = [1, 2]
#
#     sigma2s_func = get_sigma2_matrix_func(D1, D2, n1, n2, n12, M, dt, alpha)
#     sigma2s = sigma2s_func(alpha, k)
#
#     mean_periodogram = sigma2s.sum() * 2
#
#     # print(mean_periodogram, true_value)
#
#     self.assertTrue(np.isclose(mean_periodogram, true_value, atol=atol),
#                     f"Mean test failed for 1 free particle: {mean_periodogram:.6g} != {true_value:.6g}")
#
#     """Test 2 linked non-confined particles with D2/D1 = 3"""
#     k = 15
#     D1 = 1
#     D2 = 3
#     n1 = 0
#     n2 = 0
#     n12 = 1
#     true_value = 0.18
#
#     # zs = [1e-4, 2.3e-3]
#     # ks = [1, 2]
#
#     sigma2s_func = get_sigma2_matrix_func(D1, D2, n1, n2, n12, M, dt, alpha)
#     sigma2s = sigma2s_func(alpha, k)
#
#     mean_periodogram = sigma2s.sum() * 2
#
#     # print(mean_periodogram, true_value)
#
#     self.assertTrue(np.isclose(mean_periodogram, true_value, atol=atol),
#                     f"Mean test failed for 1 free particle: {mean_periodogram:.6g} != {true_value:.6g}")
#
#     """Test 2 linked confined particles with arbitrary values"""
#     k = 173
#     D1 = 1.2
#     D2 = 3.7
#     n1 = 4.3
#     n2 = 2.15
#     n12 = 2.4
#     true_value = 0.0930341
#
#     # zs = [1e-4, 2.3e-3]
#     # ks = [1, 2]
#
#     sigma2s_func = get_sigma2_matrix_func(D1, D2, n1, n2, n12, M, dt, alpha)
#     sigma2s = sigma2s_func(alpha, k)
#
#     mean_periodogram = sigma2s.sum() * 2
#
#     # print(mean_periodogram, true_value)
#
#     self.assertTrue(np.isclose(mean_periodogram, true_value, atol=atol),
#                     f"Mean test failed for 1 free particle: {mean_periodogram:.6g} != {true_value:.6g}")
#
# def test_max_search(self):
#     """
#     Test if the numerical search can find the correct posterior maximum
#     """
#
#     from likelihood import get_MLE
#
#     true_D1 = 1  # um^2/s
#     true_n1 = 1  # 1/s
#
#     M = 100
#     dt = 0.3
#     ks_fit = [48, 49]
#     PX_fit = np.array([3.7159e-02, 1.22e-1])
#     PY_fit = np.array([0.277, 0.1660])
#
#     start_point = {'D1': 1, 'n1': 1}
#
#     # MLE_free, ln_evidence_free, max_free, ln_model_evidence_direct = get_MLE(
#     #     ks=ks_fit, zs_x=PX_fit, zs_y=PY_fit, M=M, dt=dt, link=False, hash_no_trial=None)
#     #
#     # print(MLE_free)

# def test_simulate_and_calculate_BF():

# def plot_func(xs, dim):
#     out = []
#     for x in xs:
#         args = to_list(fit_min_with_link).copy()
#         args[dim] = x
#         args_dict = {a: args[i] for i, a in enumerate(names)}
#         out.append(minimize_me(args))
#         # out.append(ln_lklh_func(**args_dict))
#     return out
# # step = 0.1
# for dim in range(len(names)):
#     # dim = 4
#     x_d = to_list(fit_min_with_link)[dim]
#     xs = np.linspace(np.min([x_d * 0.1, 0.1]),
#                      np.max([x_d * 20, 1]), num=100, endpoint=True)
#     plt.plot(xs, plot_func(xs, dim), label=names[dim], color=color_sequence[dim])
#     plt.scatter(x_d, plot_func([x_d], dim)[0], color=color_sequence[dim])
#
# plt.legend()
# plt.show(block=False)
# # import time
# # time.sleep(3)
# # t1 = np.complex128(3 + 5j)
# # t1.real

# np.linspace(10, 10, 10)


# # if __name__ == "__main__"":
# """Check that the minimum is found correclty"""
# # %%
#
# import numpy as np
# from matplotlib import pyplot as plt
# from scipy.fftpack import fft, ifft
# from tqdm import tqdm
#
# from calculate import simulate_and_calculate_Bayes_factor_terminal
# from likelihood import (get_ln_likelihood_func_no_link, get_ln_prior_func,
#                         get_MLE, get_sigma2_matrix_func,
#                         likelihood_2_particles_x_link_one_point)
# from simulate import simulate_2_confined_particles_with_fixed_angle_bond
# from support import get_cluster_args_string, hash_from_dictionary
#
# D1 = 1
# D2 = 0.4
# n1 = 30
# n2 = 1
# n12 = 30 * 0
# gamma = 1e-8
# dt = 0.05
# angle = 0
# L = 20
# trial = 0
# M = 100
# verbose = True
# recalculate_trajectory = 0
# recalculate_BF = 1
#
# true_parameters = {name: val for name, val in zip(
#     ('D1 D2 n1 n2 n12 gamma dt angle L trial M'.split()),
#     (D1, D2, n1, n2, n12, gamma, dt, angle, L, trial, M))}
#
# t, R, dR, hash = simulate_2_confined_particles_with_fixed_angle_bond(
#     parameters=true_parameters, plot=1, save_figure=False, recalculate=recalculate_trajectory)
#
# dRk_short = dt * fft(dR[:2, :])
#
# M = np.shape(dR)[1]
#
# if not M % 2:    # even M
#     fit_indices = np.arange(1, M / 2, dtype=np.int)
# else:  # odd M
#     fit_indices = np.arange(1, (M - 1) / 2 + 1, dtype=np.int)
# ks_fit = fit_indices
# dRks_fit = dRk_short[:, fit_indices]
#
# # args_string = get_cluster_args_string(
# #     D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, dt=dt, angle=angle, L=L, trial=trial, M=M, verbose=verbose, recalculate_trajectory=recalculate_trajectory, recalculate_BF=recalculate_BF)
# # lg_BF_val, ln_evidence_with_link, ln_evidence_free, loaded = simulate_and_calculate_Bayes_factor_terminal(
# #     args_string)
# # print(lg_BF_val, ln_evidence_with_link, ln_evidence_free)
#
# # %% Make a 2D plot of the likelihood without link
#
#
# ln_lklh_func = get_ln_likelihood_func_no_link(
#     ks=ks_fit, M=M, dt=dt, dRks=dRks_fit)
# ln_prior, sample_from_the_prior = get_ln_prior_func()
# names = ('D1', 'n1')
#
#
# def minimize_me(args):
#     """-ln posterior to minimize"""
#     args_dict = {a: args[i] for i, a in enumerate(names)}
#     return -ln_lklh_func(**args_dict) - ln_prior(**args_dict)
#     # return - ln_prior(**args_dict)
#
#
# def to_list(dict):
#     return [dict[key] for key in names]
#
#
# # ln_lklh_func(**fit_min_with_link)
# # minimize_me(to_list(fit_min_with_link))
# link = False
#
# # %%
# hash, hash_no_trial = hash_from_dictionary(parameters=true_parameters)
# MLE_free, ln_evidence_free, max_free, success_free = get_MLE(
#       ks=ks_fit, zs_x=None, zs_y=None, dRks=dRks_fit, hash_no_trial=hash_no_trial, M=M, dt=dt, link=False, verbose=verbose, rotation=False)
# MLE_free
#
# # %%
# # Min 1
# min_true = {'D1': D1, 'n1': n1}
# min2 = MLE_free
# minimize_me(to_list(min_true))
# minimize_me(to_list(min2))
#
# # %%
#
# # lims = [[0.01, 10], [0.01, 100]]
# n = 100
# D1s = np.linspace(0.4, 3, 20)
# n1s = np.logspace(-1, max([np.log(2*min2['n1']), 1]), n)
# xx, yy = np.meshgrid(D1s, n1s)
# ln_funcs = np.full((len(n1s), len(D1s)), np.nan)
# for i, D1 in enumerate(tqdm(D1s)):
#     for j, n1 in enumerate(n1s):
#         ln_funcs[j, i] = minimize_me((D1, n1))
#
# # ln_funcs = [minimize_me((D1, n1)) for D1, n1 in zip(D1s, n1s)]
# # list(zip(xx, yy))
#
# # %%
#
# plt.figure(36 + int(link), clear=True)
# plt.contourf(D1s, n1s, ln_funcs, levels=50, cmap='jet')
# plt.colorbar()
# plt.scatter(*list(min_true.values()), marker = 'x', color = 'k')
# plt.scatter(*list(min2.values()), marker = 'o', color = 'k')
# plt.yscale('log')
# plt.xscale('linear')
# plt.xlabel('D1')
# plt.ylabel('n1')
#
# plt.show()
#
# list(min_true.values())
#
# # # %%
# # l = -n1
# # k=1
# # c1=np.exp(l * dt)
# # ck = np.exp(- 2 * np.pi * k * 1j /M)
# # Q11=M/2/l * (c1**2-1) / np.abs(c1-ck)**2
# # Q11
