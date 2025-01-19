import typing

import matplotlib.pyplot as plt
import numpy as np

# global parameters
NU_MIN_RADIUS = 0.05


def grid(pmin, pmax, step) -> ([float], [float]):
    x = []
    y = []

    for i in np.arange(pmin, pmax + 1, step):
        for j in np.arange(pmin, pmax + 1, step):
            x.append(i)
            y.append(j)
    return x, y


def random_sample(a, b) -> (float, float):
    x = (b - a) * np.random.random_sample() + a
    y = (b - a) * np.random.random_sample() + a
    return np.array((x, y))


class Node:
    # uuid for Node
    count = 0

    def __init__(self, x, y):
        self.x: float = x
        self.y: float = y
        self.parent: Node | None = None

        self.id = Node.count
        Node.count += 1

    def xy(self) -> (float, float):
        return np.array((self.x, self.y))

    def __repr__(self):
        return "Node(" + str(self.x) + ", " + str(self.y) + ")"

    def __eq__(self, other):
        return self.x == other.x \
            and self.y == other.y \
            and hash(self) == hash(other)

    def __hash__(self):
        return self.id


class TreeGraph:

    def _check_is_in_graph(self, node: Node):
        if node not in self._g.keys():
            raise KeyError(f"{node} not in graph")

    def __init__(self):
        # convention: root node has parent None
        self._g = {}  # node -> parent_node

    def add_node(self, node: Node, parent: Node | None):
        try:
            self._check_is_in_graph(node)
        except KeyError:
            self._g[node] = parent
        else:
            raise KeyError(f"{node} already in graph")

    def set_parent(self, node: Node, parent: Node):
        self._check_is_in_graph(node)
        self._check_is_in_graph(parent)
        self._g[node] = parent

    def get_parent(self, node: Node) -> Node | None:
        return self._g[node]

    def nodes(self) -> [Node]:
        return list(self._g.keys())

    def plot(self):
        """Plots the tree graph using matplotlib.pyplot"""
        for n in self.nodes():
            start_x, start_y = n.xy()
            end_x, end_y = self._g[n].xy() if self._g[n] is not None else n.xy()
            plt.plot([start_x, end_x], [start_y, end_y], marker='+', color="gray")


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
    step_mul = 0.8  # between 0 and 1
    return n_nearest_xy + ((v_to_xrand / np.linalg.norm(v_to_xrand)) * step_mul)


def obstacle_free(start: np.ndarray, end: np.ndarray):
    return True


def rrt(start: (float, float), pmin, pmax, steps: int) -> TreeGraph:
    g = TreeGraph()
    g.add_node(Node(*start), None)

    for _ in range(steps):
        x_rand = random_sample(pmin, pmax)
        n_nearest = nearest_node(g, x_rand)
        x_new = steer(n_nearest, x_rand)
        if obstacle_free(n_nearest.xy(), x_new):
            g.add_node(Node(*x_new), n_nearest)

    return g


class CostTreeGraph(TreeGraph):

    def __init__(self):
        super().__init__()
        self._cost: dict[Node, float] = {}

    def get_cost(self, node: Node) -> float:
        super()._check_is_in_graph(node)
        return self._cost[node]

    def set_cost(self, node: Node, cost: float):
        super()._check_is_in_graph(node)
        self._cost[node] = cost


def rrt_star(start: np.ndarray,
             target: np.ndarray,
             pmin, pmax,
             steps: int):
    """
    RRT* (or Optimal RRT) implementation similar to the algorithm
    presented in the paper.

    There are subtle differences in this implementation.

    Functional: L8 (line 8) was moved below L12, because of the way
    the graph is implemented, so we ensure every node but the root node
    has a valid parent when updating the graph.

    Computation-wise: for the minimum-cost path computing,
    observe that c_min in L12 takes a values that was computed at L11.
    We simply store the calculated cost to avoid computing it twice.

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

    return g


if __name__ == '__main__':
    NU_MIN_RADIUS = 0.5
    pmin, pmax, step = (-10, 10, 0.1)
    x, y = grid(pmin, pmax, step)

    start = np.array((-7.5, -6.3))
    target = np.array((8.1, 5.2))
    num_iters = 500

    g = rrt_star(start, target, pmin, pmax, num_iters)
    g.plot()

    # plt.xlim([pmin, pmax])
    # plt.ylim([pmin, pmax])
    plt.scatter(*start, c="yellow")
    plt.scatter(*target, c="red")
    plt.show()
