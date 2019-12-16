

import argparse

import numpy as np

from calculate import simulate_and_calculate_Bayes_factor
from likelihood import calculate_bayes_factor
from simulate import simulate_2_confined_particles_with_fixed_angle_bond

# %% Constants
D2 = 0.4  # um^2/s
D1 = 5 * D2   # um^2/s; 0.4
n1 = 2e3
n2 = 2e3
# n12 = 10 * n2  # s^{-1}. Somehting interesting happens between [1e-9; 1e-6]
N = 101  # required number of points in a trajectory
dt = 0.05  # s 0.3
gamma = 1e-8    # viscous drag, in kg/s
L = 0.5
angle = 0
recalculate = False


def main(arg_str):
    # Define arguments
    arg_parser = argparse.ArgumentParser(
        description='Job manager. You must choose whether to resume simulations or restart and regenerate the arguments file')
    arg_parser.add_argument('--trial', action='store', type=int, required=True)
    arg_parser.add_argument('--n12', action='store', type=float, required=True)

    # Read arguments
    input_args = arg_parser.parse_args(arg_str.split())
    trial = input_args.trial
    n12 = input_args.n12

    T = dt * (N - 1)  # s

    true_parameters = {name: val for name, val in zip(
        ('D1 D2 n1 n2 n12 gamma T dt angle L trial'.split()),
        (D1, D2, n1, n2, n12, gamma, T, dt, angle, L, trial))}

    # Simulate the trajectories
    t, R, dR, hash = simulate_2_confined_particles_with_fixed_angle_bond(
        parameters=true_parameters, plot=False, save_figure=False, recalculate=recalculate)

    # Calculate and store the Bayes factor
    lg_bayes_factor, ln_evidence_with_link, ln_evidence_free = calculate_bayes_factor(
        t=t, dR=dR, true_parameters=true_parameters, hash=hash, recalculate=False,  plot=False)

    return
