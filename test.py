"""
Test functions for different parts of the code
"""

import dill
import numdifftools as nd
from scipy.optimize import minimize
import numpy as np
import logging
import pytest


def test_hessian_calculation():
    tol = 1e-5

    # Simple test
    # def minimize_me(args):
    #     x, y, z = args
    #     return 100 * x ** 2 + 5 * y ** 2 + z ** 2 / 100
    #
    # start_point_vals = (3, 2, 5)
    # map, det_hess_inv, success = get_MAP_and_hessian(minimize_me, start_point_vals, verbose=1)
    # assert map == pytest.approx((0, 0, 0), abs=tol)
    # assert det_hess_inv == pytest.approx(1 / 40, abs=tol)
    # assert success

    # Real function test
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

    start_point_vals = (3, 2, 5, 1,2,7)
    map, det_hess_inv, success = get_MAP_and_hessian(minimize_me, start_point_vals, verbose=1)



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


def get_MAP_and_hessian(minimize_me, x0, verbose=False, tol=1e-5):
    """
    Get MLE and reliably calculate the Hessian determinant (through rescaling).

    Returns:
        (MAP, det_inv_hess, success)
    """
    min_scale = tol
    d = len(x0)
    options_NM = {'disp': verbose, 'xatol': tol, 'fatol': tol, 'maxiter': 1000 * d}

    _min = minimize(minimize_me, x0, method='Nelder-Mead', options=options_NM)
    if verbose:
        print('NM search 1:', _min)
        hess_diag = nd.Hessdiag(minimize_me)(_min.x)
        print('Diagonal Hessian: ', hess_diag)

    # Rescale
    scales = np.array([np.max([np.abs(xi), min_scale]) for xi in _min.x])
    inv_hessian_rescale_factor = (scales ** 2).prod()
    if verbose:
        print('\n\nScales:\t', scales)
        print('\nAfter rescaling:')

    def rescaled_minimize_me(args):
        args = args * scales
        return minimize_me(args)

    # Rerun search
    _min = minimize(rescaled_minimize_me, (_min.x > 0) * 1.0, method='Nelder-Mead',
                    options=options_NM)
    if verbose:
        print('NM search 2', _min)

    # Evaluate inverse Hessian determinant
    det_inv_hess = (1 / np.linalg.det(nd.Hessian(rescaled_minimize_me)(_min.x))
                    * inv_hessian_rescale_factor)
    if verbose:
        print('Det. inv. Hessian:\t', det_inv_hess)

    if not _min.success:
        logging.warning('MAP search failed to converge')

    return _min.x * scales, det_inv_hess, _min.success


if __name__ == '__main__':
    test_hessian_calculation()
