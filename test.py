"""
Test functions for different parts of the code
"""

import dill
import numdifftools as nd
from scipy.optimize import minimize
import numpy as np
import logging
import pytest
from likelihood import get_MAP_and_hessian
from trajectory import Trajectory

ABS_TOL = 1E-3


def test_hessian_calculation():
    tol = ABS_TOL

    # Simple test
    print('=== Test 1: Simple function ===')

    def minimize_me(args):
        x, y, z = args
        return 100 * (x - 12) ** 2 + 5 * (y - 3) ** 2 + (z + 7) ** 2 / 100

    start_point_vals = (3, 2, 5)
    map, fun, det_hess_inv, ln_evidence, success = get_MAP_and_hessian(minimize_me,
                                                                       start_point_vals,
                                                                       verbose=1)
    assert map == pytest.approx((12, 3, -7), abs=tol)
    assert det_hess_inv == pytest.approx(1 / 40, abs=tol)
    assert ln_evidence == pytest.approx(0.91238, abs=tol)
    assert success

    # Real function test - 1
    print('=== Test 2: Complex function ===')
    hessian_test_function_file = 'hessian_test_func.pyc'
    try:
        with open(hessian_test_function_file, 'rb') as file:
            minimize_me = dill.load(file)
    except EOFError as e:
        print('Encountered incomplete file\n', e)
        raise
    except FileNotFoundError:
        raise
    except Exception as e:
        print('Unhandled exception while reading a data file', e)
        raise

    start_point_vals = (3, 2, 5, 1, 2, 7)
    map, fun, det_hess_inv, ln_evidence, success = get_MAP_and_hessian(minimize_me,
                                                                       start_point_vals,
                                                                       verbose=1)
    assert map == pytest.approx(
        [2.99227253e-02, 6.78750128e-01, 1.18660341e+01, 9.00752324e+00, 1.21261888e+02,
         -2.31894553e-02], abs=tol)
    assert det_hess_inv == pytest.approx(129.15415590768418, abs=tol)
    assert ln_evidence == pytest.approx(45.76465, abs=tol)
    assert success

    # Real function test - 2
    print('=== Test 3: Complex function ===')
    hessian_test_function_file = 'hessian_test_func-2.pyc'
    try:
        with open(hessian_test_function_file, 'rb') as file:
            minimize_me = dill.load(file)
    except EOFError as e:
        print('Encountered incomplete file\n', e)
        raise
    except FileNotFoundError:
        raise
    except Exception as e:
        print('Unhandled exception while reading a data file', e)
        raise

    start_point_vals = (2, 1.5, 29, 45, 137, 0)
    map, fun, det_hess_inv, ln_evidence, success = get_MAP_and_hessian(minimize_me,
                                                                       start_point_vals,
                                                                       verbose=2)

    assert map == pytest.approx([1.99021090e-03, 4.81194514e-01, 2.20998632e+01, 1.55956517e+01,
                                 1.24197374e+02, 1.96638748e-03], abs=tol)
    assert det_hess_inv == pytest.approx(0.6511567823117265, abs=tol)
    assert ln_evidence == pytest.approx(39.787001820166914, abs=tol)
    assert success

    ## Old
    # # Find MAP
    # # : Switch here to the search function from likelihood.py
    # # {'D1': 0.1864735353883898, 'D2': 1.3431483056003075, 'n1': 23.705008976373524,
    # # 'n2': 16.964125784499846, 'n12': 50.99559700893256, 'alpha': -0.08126032112623911}
    #
    # # start_point_vals = (0.5664510814113775, 0.47074839368289345, 12.443958217649788,
    # #                     15.879653787376705, 55.94163390661827, -1.6507656317291708)
    # start_point_vals = (2.99255698e-02, 6.78296685e-01, 1.18678635e+01, 9.00088132e+00,
    #                     1.21904680e+02, - 2.32033708e-02)

    # eps_default = 1.4901161193847656e-08
    # min_scale = 1e-5
    # hess_step = None

    # for tol in 10. ** np.arange(-5, -10, -1):
    #     print(f'tol={tol}\n', get_MAP_and_hessian(minimize_me, start_point_vals, verbose=0, tol=tol))

    # for tol in 10. ** np.arange(-5, -10, -1):
    #     _min = minimize(minimize_me, start_point_vals, method='BFGS',
    #                     options={'disp': True, 'gtol': tol, 'eps': min([eps_default, tol])})
    #     # _min = minimize(minimize_me, _min.x, method='BFGS',
    #     #                 options={'disp': True, 'gtol': tol, 'eps': min([eps_default, tol])})
    #     print(f'\ntol={tol}')
    #     print(f'\nmin.x={_min.x}')
    #     print(f'Det. Hess inv.:\t', np.linalg.det(_min.hess_inv))
    #     print('Manual det. inv. Hessian: ', 1 / np.linalg.det(nd.Hessian(minimize_me,
    #                                                                      step=hess_step)(_min.x)))
    #
    #     # Rescale
    #     scales = np.array([np.max([np.abs(xi), min_scale]) for xi in _min.x])
    #     inv_hessian_rescale_factor = (scales ** 2).prod()
    #     print('Scales:\t', scales)
    #     print('\nAfter rescaling:')
    #
    #     def rescaled_minimize_me(args):
    #         args = args * scales
    #         return minimize_me(args)
    #
    #     _min = minimize(rescaled_minimize_me, (_min.x > 0) * 1.0, method='BFGS',
    #                     options={'disp': True, 'gtol': tol, 'eps': min([eps_default, tol])})
    #     print(f'\nmin.x={_min.x}')
    #     print(f'Det. Hess inv.:\t', np.linalg.det(_min.hess_inv) * inv_hessian_rescale_factor)
    #     print('Manual det. inv. Hessian: ',
    #           1 / np.linalg.det(nd.Hessian(rescaled_minimize_me, step=hess_step)(_min.x))
    #           * inv_hessian_rescale_factor)

    # print('Test-1\t',rescaled_minimize_me(np.ones(6)))

    # print('\n\nManual det. inv. Hessian: ', 1/np.linalg.det(nd.Hessian(minimize_me)((0,0,0))))
    # print('True det. inv. Hessian: ', 1 / 40)

    # # # Simple gradient bug test
    # # res = nd.Gradient(lambda x, y: x + y, full_output=False)(1, 3)
    # # res = nd.Gradient(lambda x, y: x + y, full_output=True)(1, 3)
    #
    # man_jac = nd.Gradient(minimize_me, order=2, full_output=False)(min.x)
    # # print(out)
    # # man_jac, info = nd.Gradient(minimize_me, full_output=True)(min.x)
    #
    # print(f'\nManually checking gradient in the minimum (BFGS vs. numdifftools):\n {min.jac}\n'
    #       f'{man_jac}\n')
    # pw_range = range(0, -11, -1)
    # step_range = [10 ** pw for pw in pw_range]
    # for step in step_range:
    #     print(f'\nstep={step}:\t'
    #           f'{nd.Gradient(minimize_me, step=step, order=12, full_output=False)(min.x)}')
    # # print('Extra grad info:\n', info, '\n\n')
    # #
    # #
    # # BFGS_hess_inv = min.hess_inv
    # # print('Det. Hessian inverse BFGS: ', np.linalg.det(BFGS_hess_inv))
    # #
    # print('Exploring the det. of the inv. Hessian.\n')
    # for step in step_range:
    #     print(f'\nstep={step}:\t'
    #           f'{1 / np.linalg.det(nd.Hessian(minimize_me, step=step, full_output=False)(min.x))}')
    # # # Manual Hessian
    # # man_hess, info = nd.Hessian(minimize_me, full_output=True)(min.x)
    # # man_det_hess_inv = 1 / np.linalg.det(man_hess)
    # # print('Manual det. hess_inv: ', man_det_hess_inv)
    # #
    # # print('Man hess info: ', info)


def test_get_MLE():
    np.random.seed(0)

    test_traj = Trajectory.from_parameter_dictionary(
        {'D1': 0.004, 'D2': 0.4, 'n1': 1.0, 'n2': 1.0, 'n12': 0.02, 'M': 100, 'dt': 0.05,
         'L0': 20.0, 'model': 'localized_different_D_detect_angle', 'angle': 0.0, 'trial': 18,
         'recalculate': True}
    )
    print('\nLg Bayes factor for link: ', test_traj.lgB)
    assert test_traj.lgB == pytest.approx(-7.57388, ABS_TOL)

    np.random.seed()

#
#
# if __name__ == '__main__':
#     test_hessian_calculation()
