import typing
import numpy as np
from graph import Node, TreeGraph, CostTreeGraph
from obstacle import obstacle_free

# global parameters
# don't change default values, assign them using
# the function `set_parameters()`
NU = 0.05
"""As defined in reference [1] (in README)"""

DELTA_TARGET_REACHED = 0.2
"""Distance condition to meet to consider target has been attained from coordinates (x, y)"""

STEP_NORM = 0.1
"""Defines the step from nearest node to random sample.
Modifying this must be of the same magnitude as DELTA_TARGET_REACHED (otherwise, target
will never be found)"""


def set_parameters(nu: float = NU,
                   d_target_reached: float = DELTA_TARGET_REACHED,
                   step_norm: float = STEP_NORM):
    """Change global parameters. See documentation in the `rrt` module"""
    global NU, DELTA_TARGET_REACHED, STEP_NORM
    NU = nu
    DELTA_TARGET_REACHED = d_target_reached
    STEP_NORM = step_norm


def random_sample(a, b) -> (float, float):
    x = (b - a) * np.random.random_sample() + a
    y = (b - a) * np.random.random_sample() + a
    return np.array((x, y))


def nearest_node(g: TreeGraph, x_rand: np.ndarray) -> Node:
    min_n = g.nodes()[0]
    min_d = np.linalg.norm(x_rand - min_n.xy())
    for n in g.nodes():
        d = np.linalg.norm(x_rand - n.xy())
        if d < min_d:
            min_d = d
            min_n = n

    return min_n


def set_vec_norm(vec: np.ndarray, n: float):
    """Returns a new vector of norm n"""
    return n * vec / np.linalg.norm(vec)


def steer(n_nearest: Node, x_rand: np.ndarray):
    """
    Not based on the definition of the `Steer() function` from [1]
    (see references in the readme.md). Instead,
    we use a step towards the next node, similar to [2],
    robots of the SSL are holonomic so this stepping should be valid
    in this use-case.

    :param n_nearest: Node
    :param x_rand: (x, y) ndarray
    :return: new point p satisfying min(dist(p, n_nearest)) while
    maintaining dist(x_rand, p) > n  where n > 0
    """
    n_nearest_xy = n_nearest.xy()
    v_to_xrand = x_rand - n_nearest_xy
    v = set_vec_norm(v_to_xrand, STEP_NORM)

    return n_nearest_xy + v


def rrt(start: np.ndarray, pmin, pmax, steps: int) -> (TreeGraph, bool):
    """
    RRT implementation as close to the paper as possible.
    Does not check wheter target is attained.

    Reference: uses the definition of [1]
    """
    g = TreeGraph()
    g.add_node(Node(*start), None)

    for _ in range(steps):
        x_rand = random_sample(pmin, pmax)
        n_nearest = nearest_node(g, x_rand)
        x_new = steer(n_nearest, x_rand)
        if obstacle_free(n_nearest.xy(), x_new):
            g.add_node(Node(*x_new), n_nearest)

    return g, False


def rrt_star(start: np.ndarray,
             target: np.ndarray,
             pmin: float, pmax: float,
             steps: int) -> (CostTreeGraph, bool):
    """
    RRT* (or Optimal RRT) implementation similar to the algorithm
    presented in the paper.

    There are subtle differences in this implementation.

    - Functional: L8 (line 8) was moved below L12, because of the way
    the graph is implemented, so we ensure every node but the root node
    has a valid parent when updating the graph.

    - Computation-wise: for the minimum-cost path computing,
    observe that c_min in L12 takes a values that was computed at L11.
    We simply store the calculated cost to avoid computing it twice.

    - Attaining the target: if an added node is considered close to the target
    by a certain delta (specified in function `near_target(Node)`), the algorithm
    stops and returns True, with the graph. If the target could not be reached,
    this value will be False instead.

    Reference: [1]
    """

    def r(num_nodes: int):
        """neighbour search radius"""
        # TODO: seems awfully low for num_nodes > 3000
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
        return np.linalg.norm(end - start)  # + target_cost(start)

    g = CostTreeGraph()
    n_start = Node(*start)
    g.add_node(n_start, None)
    g.set_cost(n_start, 0)

    for _ in range(steps):
        x_rand = random_sample(pmin, pmax)  # (x, y)
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
                return g, True

    return g, False
