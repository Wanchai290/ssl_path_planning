import numpy as np
import matplotlib.pyplot as plt


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
            plt.plot([start_x, end_x], [start_y, end_y], marker='+', color="gray", zorder=0)


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
