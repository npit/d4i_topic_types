import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from scipy.interpolate import interp1d


def interpolate(xinp, yinp):
    k = 5
    newx, newy = [], []
    for j in range(len(yinp)):
        newy.append([])
        f = interp1d(xinp, yinp[j], kind='cubic')

        for i in range(len(xinp) - 1):
            startx, endx = xinp[i], xinp[i+1]
            # interpolate k points between
            i_x = np.linspace(startx, endx, k + 2)[1:-1]
            if j == 0:
                newx.append(xinp[i])
                newx.extend(i_x)
            i_y = [val if val > 0 else 0 for val in f(i_x)]
            newy[-1].append(yinp[j][i])
            newy[-1].extend(i_y)
    return newx, newy


def pretty_stackplot(x, y, colors, legend=None, rec=True):
    x, y = interpolate(x, y)
    print([len(yy) for yy in y])
    plt.figure()
    # p = plt.stackplot(x, y, colors=colors, alpha=0.6, baseline='weighted_wiggle')
    p = plt.stackplot(x, y, colors=colors, alpha=0.6, baseline='zero')
    if legend is not None:
        plt.legend(legend)
    # cumulative y
    cumul_y = []
    for i in range(len(y)):
        cy = np.cumsum(y[:i+1])
        cumul_y.append(
            np.sum(np.asarray(y[:(i+1)],np.float32), axis=0)
            )

    for i, y in enumerate(cumul_y):
        plt.plot(x, y, colors[i], alpha=0.4)

    plt.savefig("areastack.png")


# dummy data
x = list(range(15))
y = [[random.random() for _ in x] for _ in range(10)]

colors = list(mcolors.BASE_COLORS.keys()) + list(mcolors.TABLEAU_COLORS.keys())
colors = sorted([x for x in colors if x not in ['w']])
pretty_stackplot(x, y, colors, ["topic" + str(i) for i in x])
