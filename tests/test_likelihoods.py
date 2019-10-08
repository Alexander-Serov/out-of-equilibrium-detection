import unittest

import numpy as np
from numpy import cos, exp, pi

from likelihood import (get_ln_likelihood_func_2_particles_x_link, get_MLE,
                        get_sigma2_matrix_func,
                        likelihood_2_particles_x_link_one_point)


class tests(unittest.TestCase):

    def test_likelihood_one_point(self):
        """Check if likelihoods are correctly calculated for different parameter values"""
        k = 1
        D1 = 1
        D2 = 3
        n1 = 1
        n2 = 1
        n12 = 1
        M = 999
        dt = 0.3
        alpha = 0
        atol = 1e-8
        atol_log = 1e-3

        """Test 1"""
        z1 = 1e-4
        true_ln_likelihood = 8.317699879
        ln_estimate = likelihood_2_particles_x_link_one_point(
            z=z1, k=k, D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M, dt=dt, alpha=alpha)

        self.assertTrue(np.isclose(ln_estimate, true_ln_likelihood, atol=atol_log),
                        f"One likelihood point test failed: {ln_estimate:.6g} != {true_ln_likelihood:.6g}")

        """Test 2"""
        z2 = 1e-1
        true_ln_likelihood = -2786.649642
        ln_estimate = likelihood_2_particles_x_link_one_point(
            z=z2, k=k, D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M, dt=dt, alpha=alpha)

        self.assertTrue(np.isclose(ln_estimate, true_ln_likelihood, atol=atol_log),
                        f"One likelihood point test failed: {ln_estimate:.6g} != {true_ln_likelihood:.6g}")

        """Test alpha = 1. This gives only one chi-squared function"""
        k = 480
        z2 = 0.5 * 2 * D1 * dt
        alpha = 1

        lamb1 = lamb2 = -1
        J = -(((M * dt ** 2) * ((1 - exp(lamb1 * dt) * exp(lamb2 * dt)) * (1 + exp(lamb1 * dt) * exp(lamb2 * dt) - (exp(lamb1 * dt) + exp(lamb2 * dt)) * cos((2 * pi * k) / M)) * (1 - cos((2 * pi * k) / M)))) /
              ((lamb1 + lamb2) * ((1 + exp(lamb1 * dt) ** 2 - 2 * exp(lamb1 * dt) * cos((2 * pi * k) / M)) * (1 + exp(lamb2 * dt) ** 2 - 2 * exp(lamb2 * dt) * cos((2 * pi * k) / M)))))
        sigma2 = 2 * D1 * J / (M * dt)

        true_ln_likelihood = -z2 / sigma2 / 2 - np.log(2) - np.log(sigma2)
        ln_estimate = likelihood_2_particles_x_link_one_point(
            z=z2, k=k, D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, M=M, dt=dt, alpha=alpha)

        # print('Likelihoods for the y-component', ln_estimate, true_ln_likelihood, J / M / dt)

        self.assertTrue(np.isclose(ln_estimate, true_ln_likelihood, atol=atol_log),
                        f"One likelihood point test failed: {ln_estimate:.6g} != {true_ln_likelihood:.6g}")

    def test_likelihood_several_points(self):
        """Several values of the periodogram"""
        M = 999
        dt = 0.3
        atol = 1e-8
        atol_log = 1e-3
        test_point = (1, 2, 1, 1, 1)    # D1, D2, n1, n2, n12

        """Test 1"""
        zs = [1e-4, 2.3e-3]
        ks = [1, 2]

        ln_lklh = get_ln_likelihood_func_2_particles_x_link(
            ks=ks, zs_x=zs, M=M, dt=dt)
        ln_estimate = ln_lklh(*test_point)
        true_ln_estimate = 1.483560874

        self.assertTrue(np.isclose(ln_estimate, true_ln_estimate, atol=atol_log),
                        f"Likelihood test failed for 2 points: {ln_estimate:.6g} != {true_ln_estimate:.6g}")

        """Test 2"""
        zs = [1.13e-1, 4.5e-2]
        ks = [101, 114]

        ln_lklh = get_ln_likelihood_func_2_particles_x_link(
            ks=ks, zs_x=zs, M=M, dt=dt)
        ln_estimate = ln_lklh(*test_point)
        true_ln_estimate = 3.264467649

        self.assertTrue(np.isclose(ln_estimate, true_ln_estimate, atol=atol_log),
                        f"Likelihood test failed for 2 points: {ln_estimate:.6g} != {true_ln_estimate:.6g}")

    def test_MLE_search(self):
        """Several values of the periodogram"""
        M = 999
        dt = 0.3
        alpha = 0
        atol = 1e-8
        atol_log = 1e-3
        test_point = (1, 2, 1, 1, 1)    # D1, D2, n1, n2, n12

        """Test 1"""
        zs = [dt**2, dt**2 * 0.95]
        ks = [400, 402]

        # MLE = get_MLE(ks=ks, zs=zs, M=M, dt=dt, alpha=alpha)
        # # ln_estimate = ln_lklh(*test_point)
        # true_MLE = 0.7514517168
        #
        # # self.assertTrue(np.isclose(MLE, true_MLE, atol=atol),
        # #                 f"Likelihood test failed for 2 points: {MLE:.6g} != {true_MLE:.6g}")
        # print(MLE)
        # print(true_MLE)
    # #
    # #     # """Test 2"""
    # #     # zs = [1.13e-1, 4.5e-2]
    # #     # ks = [101, 114]
    # #     #
    # #     # ln_lklh = get_ln_likelihood_func_2_particles_x_link(
    # #     #     ks=ks, zs=zs, M=M, dt=dt, alpha=alpha)
    # #     # ln_estimate = ln_lklh(*test_point)
    # #     # true_ln_estimate = 3.194565432
    # #     #
    # #     # self.assertTrue(np.isclose(ln_estimate, true_ln_estimate, atol=atol_log),
    # #     #                 f"Likelihood test failed for 2 points: {ln_estimate:.6g} != {true_ln_estimate:.6g}")
    #

    def test_mean(self):
        """I know what is the mean expected value of the stochastic variable if there is no link"""
        M = 999
        dt = 0.3
        alpha = 0
        atol = 1e-8
        atol_log = 1e-3

        """Test 1 free particle"""
        k = 100
        D1 = 1
        D2 = 3
        n1 = 0
        n2 = 0
        n12 = 0

        # zs = [1e-4, 2.3e-3]
        # ks = [1, 2]

        sigma2s_func = get_sigma2_matrix_func(D1, D2, n1, n2, n12, M, dt, alpha)
        sigma2s = sigma2s_func(alpha, k)

        mean_periodogram = sigma2s.sum() * 2
        true_value = 2 * D1 * dt**2

        self.assertTrue(np.isclose(mean_periodogram, true_value, atol=atol),
                        f"Mean test failed for 1 free particle: {mean_periodogram:.6g} != {true_value:.6g}")

        """Test 2 linked non-confined particles with D2/D1 = 1"""
        k = 10
        D1 = 1
        D2 = 1
        n1 = 0
        n2 = 0
        n12 = 1
        true_value = 0.09100735399

        # zs = [1e-4, 2.3e-3]
        # ks = [1, 2]

        sigma2s_func = get_sigma2_matrix_func(D1, D2, n1, n2, n12, M, dt, alpha)
        sigma2s = sigma2s_func(alpha, k)

        mean_periodogram = sigma2s.sum() * 2

        # print(mean_periodogram, true_value)

        self.assertTrue(np.isclose(mean_periodogram, true_value, atol=atol),
                        f"Mean test failed for 1 free particle: {mean_periodogram:.6g} != {true_value:.6g}")

        """Test 2 linked non-confined particles with D2/D1 = 3"""
        k = 15
        D1 = 1
        D2 = 3
        n1 = 0
        n2 = 0
        n12 = 1
        true_value = 0.18

        # zs = [1e-4, 2.3e-3]
        # ks = [1, 2]

        sigma2s_func = get_sigma2_matrix_func(D1, D2, n1, n2, n12, M, dt, alpha)
        sigma2s = sigma2s_func(alpha, k)

        mean_periodogram = sigma2s.sum() * 2

        # print(mean_periodogram, true_value)

        self.assertTrue(np.isclose(mean_periodogram, true_value, atol=atol),
                        f"Mean test failed for 1 free particle: {mean_periodogram:.6g} != {true_value:.6g}")

        """Test 2 linked confined particles with arbitrary values"""
        k = 173
        D1 = 1.2
        D2 = 3.7
        n1 = 4.3
        n2 = 2.15
        n12 = 2.4
        true_value = 0.0930341

        # zs = [1e-4, 2.3e-3]
        # ks = [1, 2]

        sigma2s_func = get_sigma2_matrix_func(D1, D2, n1, n2, n12, M, dt, alpha)
        sigma2s = sigma2s_func(alpha, k)

        mean_periodogram = sigma2s.sum() * 2

        # print(mean_periodogram, true_value)

        self.assertTrue(np.isclose(mean_periodogram, true_value, atol=atol),
                        f"Mean test failed for 1 free particle: {mean_periodogram:.6g} != {true_value:.6g}")

    def test_max_search(self):
        """Test if the numerical search can find the correct posterior maximum"""

        from likelihood import get_MLE

        true_D1 = 1  # um^2/s
        true_n1 = 1  # 1/s

        M = 100
        dt = 0.3
        ks_fit = [48, 49]
        PX_fit = np.array([3.7159e-02, 1.22e-1])
        PY_fit = np.array([0.277, 0.1660])

        start_point = {'D1': 1, 'n1': 1}

        MLE_free, ln_evidence_free, max_free, ln_model_evidence_direct = get_MLE(
            ks=ks_fit, zs_x=PX_fit, zs_y=PY_fit, M=M, dt=dt, link=False, start_point=start_point)

        print(MLE_free)
