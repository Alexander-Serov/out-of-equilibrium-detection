"""
Simulate random walk of two particles connected by a spring subject to two different diffusivities D1 and D2
"""

try:
    bl_has_run

except Exception:
    %matplotlib
    %load_ext autoreload
    %autoreload 2
    bl_has_run = True

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange

# matplotlib.use('TkAgg')

# % Constants
D1 = 1  # m^2/s
D2 = 2  # m^2/s
k12 = k1 = k2 = 1e2  # spring constant, N/m
gamma = 4  # viscous drag, in N * s / m
L12 = L1 = L2 = 1  # m
dt = 1e-2  # seconds
substeps = 100
N = int(1e3)  # time step
x10 = -L12 / 2
x20 = L12 / 2
file = './trajectory.dat'

# % Initialize
x1 = np.ones(N, dtype=np.float32) * np.nan
x2 = np.ones(N, dtype=np.float32) * np.nan
dx1 = np.zeros(N, dtype=np.float32)
dx2 = np.zeros(N, dtype=np.float32)
x1[0] = x10
x2[0] = x20
dt_sim = dt / substeps
t = np.arange(N) * dt
np.random.seed(0)
white_noise = np.random.randn(N * substeps, 2)


# % Simulate
for step in trange(N - 1):
    x1[step + 1] = x1[step]
    x2[step + 1] = x2[step]
    for int_step in range(substeps):
        dW = np.sqrt([2 * D1, 2 * D2]) * np.sqrt(dt_sim) * \
            white_noise[step * substeps + int_step, :]

        dx1[step] += (- k1 / gamma * (x1[step + 1] - x10) * dt_sim
                      + k12 / gamma * (x2[step + 1] - x1[step + 1] - L12) * dt_sim
                      + dW[0])

        dx2[step] += (- k2 / gamma * (x2[step + 1] - x20) * dt_sim
                      - k12 / gamma * (x2[step + 1] - x1[step + 1] - L12) * dt_sim
                      + dW[1])
        x1[step + 1] = x1[step] + dx1[step]
        x2[step + 1] = x2[step] + dx2[step]


fig = plt.figure(1, clear=True)
plt.plot(t, x1)
plt.plot(t, x2)
plt.xlabel('t, s')
plt.ylabel('x, m')

plt.show()

# % Save
output = np.stack([t, x1, dx1, x2, dx2], axis=1)
output = pd.DataFrame(data=output, columns=['t', 'x1', 'dx1', 'x2', 'dx2'])
output.to_csv(file, sep=';')
