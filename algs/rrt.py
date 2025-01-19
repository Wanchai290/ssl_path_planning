import typing
import numpy as np
from graph import Node, TreeGraph, CostTreeGraph
from obstacle import obstacle_free

# global parameters
NU_MIN_RADIUS = 0.05  # as defined by RRT* paper
DELTA_TARGET_REACHED = 0.1  # a min dist to consider whether target is attained or not
STEER_VECTOR_STEP_MUL = 0.1  # between 0 and 1


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


def steer(n_nearest: Node, x_rand: np.ndarray):
    """
    :param n_nearest: Node
    :param x_rand: (x, y) ndarray
    :return: new point p satisfying min(dist(p, n_nearest)) while
    maintaining dist(x_rand, p) <= n  where n > 0
    """
    n_nearest_xy = n_nearest.xy()
    v_to_xrand = x_rand - n_nearest_xy

    # TODO: use NU value as defined by paper
    return n_nearest_xy + ((v_to_xrand / np.linalg.norm(v_to_xrand)) * STEER_VECTOR_STEP_MUL)


def rrt(start: np.ndarray, pmin, pmax, steps: int) -> (TreeGraph, bool):
    """
    RRT implementation as close to the paper as possible.
    Does not check wheter target is attained
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
             pmin, pmax,
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

    Reference: Sampling-based Algorithms for Optimal Motion Planning
    https://arxiv.org/pdf/1105.1186https://arxiv.org/pdf/1105.1186
    """

    def r(num_nodes: int):
        """neighbour search radius"""
        # TODO: seems awfully low for num_nodes > 3000
        return min(np.sqrt(np.log2(num_nodes) / num_nodes), NU_MIN_RADIUS)

    def near_nodes(g: TreeGraph, x_new: np.ndarray, r: float) -> typing.Set[Node]:
        """adds all neighbours in a circle of origin x_new and radius r"""
        ngh = set()
        for n in g.nodes():
            if np.linalg.norm(n.xy() - x_new) < r:
                ngh.add(n)
        return ngh

    def target_cost(p: np.ndarray):
        return np.linalg.norm(target - p)

    def near_target(node: Node):
        return np.linalg.norm(target - node.xy()) < 0.1

    def line_cost(start: np.ndarray, end: np.ndarray):
        return np.linalg.norm(end - start)  # + target_cost(start)

    g = CostTreeGraph()
    n_start = Node(*start)
    g.add_node(n_start, None)
    g.set_cost(n_start, target_cost(start))

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
                if obstacle_free(n_near.xy(), x_new) \
                        and g.get_cost(node_xnew) + line_cost(n_near.xy(), x_new) < g.get_cost(n_near):
                    g.set_parent(n_near, node_xnew)

            # stop if target attained
            if near_target(node_xnew):
                return g, True

    return g, False
