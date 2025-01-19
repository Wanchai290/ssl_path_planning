import matplotlib.pyplot as plt
import numpy as np
from algs.rrt import rrt, rrt_star


if __name__ == '__main__':
    pmin, pmax = (-5, 5)

    start = np.array((-2.1, 3.2))
    target = np.array((-0.4, 1.1))
    num_iters = 1000

    g, target_attained = rrt_star(start, target, pmin, pmax, num_iters)
    g.plot()

    plt.title(f"Target attained : {target_attained}")
    # plt.xlim([pmin, pmax])
    # plt.ylim([pmin, pmax])
    plt.scatter(*start, c="green")
    plt.scatter(*target, c="red")
    plt.show()
