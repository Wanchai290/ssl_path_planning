import matplotlib.pyplot as plt
import numpy as np
from algs import rrt, informed_rrt
from obstacle import add_obstacles, plot_obstacles, LineObstacle, CircleObstacle

if __name__ == '__main__':
    pmin, pmax = (-5., 5.)

    # search parameters
    start = np.array(((pmax + pmin) / 2., (pmax + pmin) / 2.))
    target = np.array((-2.7, 3.5))
    num_iters = 750

    # load obstacles
    add_obstacles([
        LineObstacle((3.0, 3.4), (4.2, 1.0)),
        CircleObstacle((-1., 2), 1)
    ])

    # optional
    rrt.set_parameters(step_norm=0.15, d_target_reached=0.25)

    # g, target_attained = rrt.rrt_star(start, target, pmin, pmax, num_iters)
    g, target_attained = informed_rrt.informed_rrt_star(start, target, pmin, pmax, num_iters)

    if not target_attained:
        mp = min(g.nodes(), key=lambda n: np.linalg.norm(n.xy() - target))
        print(f"Closest node : {mp}, d=%.3f" % np.linalg.norm(mp.xy() - target))

    # plot graph
    g.plot()
    plot_obstacles()

    title = f"Target attained : {target_attained}"
    if target_attained:
        node_target = g.get_closest_node(target)
        title += ", path cost : %.3f" % g.get_cost(node_target)

    plt.title(title)
    plt.xlim([pmin, pmax])
    plt.ylim([pmin, pmax])
    plt.gca().set_aspect('equal')
    plt.scatter(*start, c="b", zorder=1)
    plt.scatter(*target, c="red", zorder=1)
    plt.show()
