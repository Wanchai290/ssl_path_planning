import matplotlib.pyplot as plt
import numpy as np
from algs.rrt import rrt, rrt_star


def grid(pmin, pmax, step) -> ([float], [float]):
    x = []
    y = []

    for i in np.arange(pmin, pmax + 1, step):
        for j in np.arange(pmin, pmax + 1, step):
            x.append(i)
            y.append(j)
    return x, y


if __name__ == '__main__':
    pmin, pmax, step = (-10, 10, 0.1)
    x, y = grid(pmin, pmax, step)

    start = np.array((-7.5, -6.3))
    target = np.array((8.1, 5.2))
    num_iters = 500

    g = rrt_star(start, target, pmin, pmax, num_iters)
    g.plot()

    # plt.xlim([pmin, pmax])
    # plt.ylim([pmin, pmax])
    plt.scatter(*start, c="green")
    plt.scatter(*target, c="red")
    plt.show()
