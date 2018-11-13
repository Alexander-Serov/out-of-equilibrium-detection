import numpy as np
from tqdm import tqdm


class Bin(object):
    dt = np.nan

    def __init__(self, borders):
        self.a, self.b = borders
        self.center = np.mean(borders)
        self.data = None
        self.x_mean = np.nan
        self.dx_mean = np.nan
        self.dx_var = np.nan
        self.D_apparent = np.nan

    def bin_data(self, data):
        self.data = data[(data.x >= self.a) & (data.x < self.b)]
        self.x_mean = self.data.x.mean()
        self.dx_mean = self.data.dx.mean()
        self.dx_var = np.var(self.data.dx, ddof=1)
        self.D_apparent = self.dx_var / 2 / self.dt


def make_adaptive_mesh(data, min_n=1, min_width=0):
    """
    Bin the data into bins with a minimal number of points and minimal width)
    """

    data = data.sort_values('x')
    N = len(data)

    def x_left(i_left):
        if i_left == 0:
            return data.x.iloc[0] - (data.x.iloc[1] - data.x.iloc[0]) / 2
        else:
            return data.x.iloc[i_left - 1:i_left + 1].mean()

    def x_right(i_right):
        if i_right == N - 1:
            return data.x.iloc[N - 1] + (data.x.iloc[N - 1] - data.x.iloc[N - 2]) / 2
        else:
            return data.x.iloc[i_right:i_right + 2].mean()

    def width(i_left, i_right):
        return x_right(i_right) - x_left(i_left)

    i_left = 0
    i_bins = [i_left - 0.5]
    x_bins = [x_left(i_left)]

    with tqdm(total=N, desc='Binning data: ') as pbar:
        while i_left < N - min_n:
            i_right = i_left + min_n
            for i_right in range(i_right, N):
                if width(i_left, i_right) >= min_width:
                    break

            i_bins.append(i_right + 0.5)
            x_bins.append(x_right(i_right))

            pbar.update(i_right - i_left + 1)
            i_left = i_right + 1

        if i_left != N:
            i_bins.append(N - 1 + 0.5)
            x_bins.append(x_right(N - 1))
            pbar.update(N - i_left)

    # print(i_bins)
    # print(x_bins)
    return x_bins, i_bins, data
