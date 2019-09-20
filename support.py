"""
Support functions for the calculations that do not rely on a certain structure of the data
"""

import numpy as np


def average_over_modes(input, dk):
    """Makes averages of the 1D array in over the interval of mode indices dk"""
    M = len(input)
    if dk == 1:
        k_new = np.arange(M)
        return input, k_new

    M_new = np.floor(M / dk).astype(int)
    out = np.zeros(M_new) * np.nan
    for i in range(M_new):
        out[i] = np.mean(input[dk * i: dk * (i + 1)])
    k_new = dk / 2 + dk * np.arange(M_new)

    return out, k_new


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
