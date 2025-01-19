import matplotlib.pyplot as plt
import numpy as np
from algs.rrt import rrt, rrt_star
from obstacle import add_obstacles, plot_obstacles, LineObstacle

if __name__ == '__main__':
    pmin, pmax = (-5, 5)

    # search parameters
    start = np.array((-2.1, 3.2))
    target = np.array((-0.4, 1.1))
    num_iters = 1000

    # load obstacles
    add_obstacles([
        LineObstacle((1.0, 3.4), (4.2, 1.0)),
    ])

    g, target_attained = rrt_star(start, target, pmin, pmax, num_iters)
    g.plot()

    # plot obstacles
    plot_obstacles()

    # plot graph
    plt.title(f"Target attained : {target_attained}")
    # plt.xlim([pmin, pmax])
    # plt.ylim([pmin, pmax])
    plt.scatter(*start, c="green")
    plt.scatter(*target, c="red")
    plt.show()
