import typing

import numpy as np

from algs import rrt

from algs.rrt import NU, DELTA_TARGET_REACHED, nearest_node, steer
from graph import TreeGraph, Node, CostTreeGraph
from obstacle import obstacle_free


def rand(a, b) -> float:
    return (b - a) * np.random.random_sample() + a


def midpoint(a, b) -> np.ndarray:
    return np.array((
        (b[0] - a[0]) / 2,
        (b[1] - a[1]) / 2
    ))


class RdSquareSample:
    def __init__(self, pmin, pmax, x_start, target):
        self._bounds = (pmin, pmax)
        self._best_cost = np.inf
        self._center = midpoint(x_start, target)

        xlength = target[0] - x_start[0]
        ylength = target[1] - x_start[1]
        self._s = xlength + ylength
        self._num_pimp = 0
        """number of times path has been improved"""

    @staticmethod
    def expand_coeff(x: int):
        """x: number of times path was improved"""
        return np.exp(-0.05 * x) + 1

    def random_sample(self, c_best: float):
        if c_best < np.inf:
            if c_best < self._best_cost:
                self._best_cost = c_best
                self._num_pimp += 1

            coeff = RdSquareSample.expand_coeff(self._num_pimp)
            x_min = self._center[0] - (self._s * coeff)
            x_max = self._center[0] + (self._s * coeff)
            y_min = self._center[1] - (self._s * coeff)
            y_max = self._center[1] + (self._s * coeff)

            return (
                rand(x_min, x_max),
                rand(y_min, y_max)
            )
        else:
            return rrt.random_sample(*self._bounds)


def informed_rrt_star(
        start: np.ndarray,
        target: np.ndarray,
        pmin: float, pmax: float,
        steps: int):
    """
    Informed RRT* implementation based on [2].
    Search is limited to a square space as defined in [3]
    """
    def r(num_nodes: int):
        """neighbour search radius"""
        return min(np.sqrt(np.log2(num_nodes) / num_nodes), NU)

    def near_nodes(g: TreeGraph, x_new: np.ndarray, r: float) -> typing.Set[Node]:
        """adds all neighbours in a circle of origin x_new and radius r"""
        ngh = set()
        for n in g.nodes():
            if np.linalg.norm(n.xy() - x_new) < r:
                ngh.add(n)
        return ngh

    def near_target(node: Node):
        return np.linalg.norm(target - node.xy()) < DELTA_TARGET_REACHED

    def line_cost(start: np.ndarray, end: np.ndarray):
        return np.linalg.norm(end - start)

    g = CostTreeGraph()
    n_start = Node(*start)
    g.add_node(n_start, None)
    g.set_cost(n_start, 0)

    XN_sol = []
    rd_square = RdSquareSample(pmin, pmax, start, target)

    for _ in range(steps):
        c_best = g.get_cost(min(XN_sol, key=lambda n: g.get_cost(n))) if len(XN_sol) > 0 else np.inf
        x_rand = rd_square.random_sample(c_best)  # (x, y)
        n_nearest = nearest_node(g, x_rand)  # Node
        x_new = steer(n_nearest, x_rand)  # (x, y)

        if obstacle_free(n_nearest.xy(), x_new):
            N_near = near_nodes(g, x_new, r(len(g.nodes())))

            # connect along a minimum-cost path
            n_min = n_nearest
            c_min = g.get_cost(n_nearest) + line_cost(n_nearest.xy(), x_new)
            for n_near in N_near:
                current_cost = g.get_cost(n_near) + line_cost(n_near.xy(), x_new)
                if obstacle_free(n_near.xy(), x_new) \
                        and current_cost < c_min:
                    n_min = n_near
                    c_min = current_cost

            # add this new lowest-cost edge to the graph
            node_xnew = Node(*x_new)
            g.add_node(node_xnew, n_min)
            g.set_cost(node_xnew, c_min)

            # rewiring graph with the new node
            for n_near in N_near:
                new_cost = g.get_cost(node_xnew) + line_cost(n_near.xy(), x_new)
                if obstacle_free(n_near.xy(), x_new) \
                        and new_cost < g.get_cost(n_near):
                    g.set_parent(n_near, node_xnew)

            # stop if target attained
            if near_target(node_xnew):
                XN_sol.append(node_xnew)

    return g, len(XN_sol) > 0


if __name__ == '__main__':
    r = RdSquareSample(0, 5, np.array((3, 6)), np.array((4, 2)))
    x, y = r.random_sample(10)
