import unittest

import numpy as np
from matplotlib import pyplot as plt
from numpy import cos, exp, pi

from constants import color_sequence
from likelihood import (get_ln_likelihood_func_2_particles_x_link,
                        get_ln_prior_func, get_MLE, get_sigma2_matrix_func,
                        likelihood_2_particles_x_link_one_point)


class tests(unittest.TestCase):

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

    def test_likelihood_several_points(self):
        """Likelihood of the whole spectrum"""
        1

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
        1
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

    def test_simulate_and_calculate_BF():
        """Check that the minimum is found correclty"""
        # %%
        from support import get_cluster_args_string
        from calculate import simulate_and_calculate_Bayes_factor_terminal
        import numpy as np
        D1 = 2
        D2 = 0.4
        n1 = 0.482026
        n2 = 1
        n12 = 30
        gamma = 1e-8
        dt = 0.05
        angle = 0
        L = 20
        trial = 2
        M = 100
        verbose = True
        recalculate_trajectory = False
        recalculate_BF = True

        args_string = get_cluster_args_string(
            D1=D1, D2=D2, n1=n1, n2=n2, n12=n12, gamma=gamma, dt=dt, angle=angle, L=L, trial=trial, M=M, verbose=verbose, recalculate_trajectory=recalculate_trajectory, recalculate_BF=recalculate_BF)
        lg_BF_val, ln_evidence_with_link, ln_evidence_free, loaded = simulate_and_calculate_Bayes_factor_terminal(
            args_string)
        print(lg_BF_val, ln_evidence_with_link, ln_evidence_free)

        # %% Make a 2D plot of the likelihood without link
        import numpy as np
        ks_fit = range(1, 50)
        dRks_fit = np.array([
            [-9.42673977e-02 - 0.07830783, 4.99892066e-02 + 0.00716147j, -5.62130494e-02 - 0.16585038, -1.30686721e-01 + 0.07249521j, 5.36358168e-02 - 0.06084522, -9.73989596e-03 + 0.07519079j, 7.69816459e-02 + 0.04040272, 3.81341080e-02 - 0.00282007j, -9.46475518e-02 + 0.14101085, -1.72493847e-01 + 0.05680884j, -4.32296966e-02 + 0.05868715, -1.12375244e-01 - 0.18915243j, -1.45649895e-01 - 0.0025442, -8.40149692e-02 - 0.02567425j, 9.09710102e-02 + 0.00918383, 1.24404655e-01 + 0.08513545j, 3.48238147e-02 - 0.15115672, 7.43881763e-03 - 0.05127688j, -1.24561722e-01 + 0.0043112,  8.39442663e-02 - 0.00974443j, -1.99041569e-04 + 0.00082801, 8.30601415e-03 - 0.21257079j, -2.31752482e-01 - 0.01849718, 5.67117466e-02 + 0.19038092j, 5.01376992e-02 +
                0.07310527, -1.13334339e-01 - 0.02948302j, -1.43306845e-02 + 0.08839033, -7.16896894e-02 + 0.13650841j, 1.63010386e-01 + 0.07458665, 2.23571496e-02 + 0.08039395j, 1.67351275e-01 - 0.06136624, 5.32708818e-02 + 0.03055005j, -2.32503015e-01 - 0.01011695, -4.69016018e-02 + 0.032485j, -2.42760403e-02 - 0.00113698, 3.12318394e-02 - 0.14305091j, 8.33199033e-02 - 0.315718,   7.04069095e-02 - 0.28625322j, -9.08914438e-02 + 0.07510545, 1.79643095e-02 - 0.04653096j, 2.93792278e-01 + 0.02252216, 3.67871358e-02 - 0.07371123j, 2.55251094e-01 + 0.19216891, 1.99409415e-02 - 0.1452005j, 2.51916019e-02 + 0.13196488, -1.95085805e-01 - 0.01863178j, -3.73409053e-01 + 0.08616907, -3.67742491e-02 + 0.18409647j, -1.63878001e-01 + 0.15536625j],
            [-1.02939892e-01 - 0.00927484, 2.00817872e-01 - 0.20336064j, 4.35621007e-03 + 0.07832352, -3.94165661e-01 + 0.35657627j, -1.20323592e-01 - 0.01282175, 5.81545508e-02 - 0.3548926j, 4.71000129e-02 + 0.05647322, 4.11658369e-02 - 0.18302626j, -2.63901719e-01 - 0.22334997, 1.65399790e-01 - 0.07213713j, -1.36781750e-01 - 0.04478006, -5.44033231e-02 - 0.12666956j, 1.06566720e-01 - 0.10786352, 1.49342787e-01 - 0.14773613j, 8.22048272e-02 + 0.07531817, 2.94518973e-01 - 0.19972915j, -3.64585027e-03 + 0.19885127, -4.57097166e-02 + 0.04476418j, -1.06792959e-01 - 0.05989786, -1.39437068e-01 + 0.05732779j, -1.36245616e-01 - 0.17422502, 7.18620924e-02 + 0.13376619j, 1.53695347e-01 + 0.06229819, -6.89514986e-02 + 0.18154644j, -1.67357834e-03 + 0.11246296, -8.74728534e-02 + 0.01172282j, 2.09660767e-01 + 0.21316067, 9.76355216e-03 - 0.04522468j, -5.67369122e-01 - 0.22348069, 2.46506224e-01 + 0.03833962j, -6.53881343e-02 - 0.19654188, 8.23786216e-02 + 0.20165343j, -5.17280561e-02 - 0.12705471, 9.95339866e-02 - 0.09043727j, 2.08019534e-02 + 0.08387033, 1.28652526e-01 + 0.17038685j, -1.51907796e-01 - 0.05585611, 2.38721699e-01 - 0.29908385j, -4.06451655e-01 - 0.07762633, 9.77717696e-02 - 0.07865763j, 2.25253268e-02 + 0.12920006, -1.31056828e-01 + 0.15587177j, -8.31464689e-02 + 0.25330123, -3.11892175e-02 - 0.0140451j, -2.71227471e-03 - 0.00844309, -1.22727790e-01 + 0.32795989j, 6.28191641e-02 + 0.04053132, 2.23825205e-01 + 0.23997069j, -7.42875646e-02 + 0.11993988j]])

        # Get likelihood without link
        from matplotlib import pyplot as plt
        from likelihood import (get_ln_likelihood_func_no_link,
                                get_ln_prior_func, get_MLE, get_sigma2_matrix_func,
                                likelihood_2_particles_x_link_one_point)
        from tqdm import tqdm

        ln_lklh_func = get_ln_likelihood_func_no_link(
            ks=ks_fit, M=M, dt=dt, dRks=dRks_fit)
        ln_prior, sample_from_the_prior = get_ln_prior_func()
        names = ('D1', 'n1')

        def minimize_me(args):
            """-ln posterior to minimize"""
            args_dict = {a: args[i] for i, a in enumerate(names)}
            return -ln_lklh_func(**args_dict) - ln_prior(**args_dict)

        def to_list(dict):
            return [dict[key] for key in names]

        # ln_lklh_func(**fit_min_with_link)
        # minimize_me(to_list(fit_min_with_link))
        link = True

        # %%
        # Min 1
        min1 = {'D1': 3.558597699124776, 'n1': 59.50217118881246}
        min2 = {'D1': 1.276442952375804, 'n1': 0.06981484381621211}
        minimize_me(to_list(min1))
        minimize_me(to_list(min2))

        # %%

        # lims = [[0.01, 10], [0.01, 100]]
        n = 100
        D1s = np.logspace(0, 1, 20)
        n1s = np.logspace(-2, 2, n)
        xx, yy = np.meshgrid(D1s, n1s)
        ln_funcs = np.full((len(n1s), len(D1s)), np.nan)
        for i, D1 in enumerate(tqdm(D1s)):
            for j, n1 in enumerate(n1s):
                ln_funcs[j, i] = minimize_me((D1, n1))

        # ln_funcs = [minimize_me((D1, n1)) for D1, n1 in zip(D1s, n1s)]
        # list(zip(xx, yy))

        plt.figure(36 + int(link), clear=True)
        plt.contourf(D1s, n1s, ln_funcs, levels=30)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel('D1')
        plt.ylabel('n1')
        plt.colorbar()
        plt.show()

        1

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

        1
