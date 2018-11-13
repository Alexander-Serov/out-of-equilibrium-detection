"""
Get particle trajectory, create spatial bins, calculate the temperature in bins and in sliding time windows
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

from Bin import Bin, make_adaptive_mesh

# % Constants
min_n = int(5e2)
file = './trajectory.dat'
# file = './trajectory_with_1.dat'
# file = './trajectory_with_2.dat'

# % Load
data = pd.read_csv(file, sep=';')
data.drop(labels=['x2', 'dx2'], axis=1, inplace=True)

min_bin_width = np.std(data.dx, ddof=1) * 2
dt = data.t.loc[1] - data.t.loc[0]
Bin.dt = dt

# % Bin data
x_min = data.x.min()
x_max = data.x.max()
dx_abs_mean = data.dx.abs().mean()
# bin_borders = np.arange(x_min, x_max + bin_width, bin_width)
bin_borders, _, _ = make_adaptive_mesh(data, min_n, min_bin_width)
# print(bin_borders)
bins_len = len(bin_borders) - 1

bins = []
vars = []
D_apparents = []
dx_means = []
centers = []
for i in range(bins_len):
    bins.append(Bin(bin_borders[i:i + 2]))
    bins[i].bin_data(data)
    centers.append(bins[i].center)
    vars.append(bins[i].dx_var)
    D_apparents.append(bins[i].D_apparent)
    dx_means.append(bins[i].dx_mean)

# bins[5].data

print("Overall std(dX) / bin_width = {:.2g}".format(np.std(data.dx, ddof=1) / min_bin_width))
# print("std(dX) / bin_width for each bin:")
# print(np.sqrt(vars) / min_bin_width)
print("Min bin width = {:.2g}".format(min_bin_width))
print("Mean apparent D = {:.2g}".format(np.mean(D_apparents)))

# Plot
plt.figure(2, clear=True)
plt.plot(centers, D_apparents)
plt.xlabel('x, um')
plt.ylabel('apparent D, um^2/s')
plt.ylim([0, plt.ylim()[1]])
plt.show()

# Plot
plt.figure(3, clear=True)
plt.plot(centers, dx_means)
plt.xlabel('x, um')
plt.ylabel('<dX>, um')
plt.show()

# make_adaptive_mesh(data, 100, 0.05)
