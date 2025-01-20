import matplotlib.pyplot as plt
import numpy as np
from algs import rrt
from obstacle import add_obstacles, plot_obstacles, LineObstacle

if __name__ == '__main__':
    pmin, pmax = (-5., 5.)

    # search parameters
    start = np.array(((pmax + pmin) / 2., (pmax + pmin) / 2.))
    target = np.array((-2.7, 3.5))
    num_iters = 500

    # load obstacles
    add_obstacles([
        LineObstacle((3.0, 3.4), (4.2, 1.0)),
        LineObstacle((-0.4, 2.7), (3.7, 2.1)),
    ])

    # optional
    # rrt.set_parameters(0.1, 1, 0.5)

    g, target_attained = rrt.rrt_star(start, target, pmin, pmax, num_iters)

    if not target_attained:
        mp = min(g.nodes(), key=lambda n: np.linalg.norm(n.xy() - target))
        print(f"Closest node : {mp}, d=%.3f" % np.linalg.norm(mp.xy() - target))

    # plot graph
    g.plot()
    plot_obstacles()

    plt.title(f"Target attained : {target_attained}")
    # plt.xlim([pmin, pmax])
    # plt.ylim([pmin, pmax])
    plt.scatter(*start, c="b", zorder=1)
    plt.scatter(*target, c="red", zorder=1)
    plt.show()
