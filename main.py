import matplotlib.pyplot as plt
import numpy as np
from algs.rrt import rrt, rrt_star


if __name__ == '__main__':
    pmin, pmax, step = (-10, 10, 0.1)

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
