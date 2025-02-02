import abc
import typing
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt

Point = namedtuple('Point', ['x', 'y'])


class Obstacle(abc.ABC):
    @abc.abstractmethod
    def obstacle_free(self, l_start: np.ndarray, l_end: np.ndarray) -> bool:
        pass

    @abc.abstractmethod
    def plot(self):
        pass


def line_intersect(l_a: (np.ndarray, np.ndarray), l_b: (np.ndarray, np.ndarray)) -> bool:
    """
    taken from https://gamedev.stackexchange.com/questions/26004/how-to-detect-2d-line-on-line-collision
    :param l_a: (Line start xy, Line end xy)
    :param l_b: (Line start xy, Line end xy)
    :return: whether both lines collide
    """
    p1, p2 = Point(*l_a[0]), Point(*l_a[1])
    p3, p4 = Point(*l_b[0]), Point(*l_b[1])

    denom = ((p2.x - p1.x) * (p4.y - p3.y)) - ((p2.y - p1.y) * (p4.x - p3.x))
    numer1 = ((p1.y - p3.y) * (p4.x - p3.x)) - ((p1.x - p3.x) * (p4.y - p3.y))
    numer2 = ((p1.y - p3.y) * (p2.x - p1.x)) - ((p1.x - p3.x) * (p2.y - p1.y))

    if denom == 0:
        return numer1 == 0 and numer2 == 0

    r = numer1 / denom
    s = numer2 / denom
    return (0 <= r <= 1) and (0 <= s <= 1)


class LineObstacle(Obstacle):

    def __init__(self, start: typing.Union[np.ndarray, typing.Tuple[float, float]],
                 end: typing.Union[np.ndarray, typing.Tuple[float, float]]):
        self._start = start if start is np.ndarray else np.array(start)
        self._end = end if end is np.ndarray else np.array(end)

    def obstacle_free(self, l_start: np.ndarray, l_end: np.ndarray) -> bool:
        return line_intersect((self._start, self._end), (l_start, l_end))

    def plot(self):
        plt.plot(
            [self._start[0], self._end[0]],
            [self._start[1], self._end[1]],
            color="black"
        )


class CircleObstacle(Obstacle):
    def __init__(self, center: (float, float), radius: float):
        self._c = center if type(center) is np.ndarray else np.array(center)
        self._r = radius

    def obstacle_free(self, l_start: np.ndarray, l_end: np.ndarray) -> bool:
        # check l_end lies inside circle
        # with the current use case, this is sufficient,
        # because we search using a small step
        return np.linalg.norm(l_end - self._c) < self._r

    def plot(self):
        ax = plt.gca()
        c = plt.Circle(self._c, self._r)
        ax.add_patch(c)


# global
__g_obstacles: [Obstacle] = []


def add_obstacles(li: [Obstacle]):
    __g_obstacles.extend(li)


def plot_obstacles():
    for obs in __g_obstacles:
        obs.plot()


def obstacle_free(start: np.ndarray, end: np.ndarray):
    colliding_obstacles = set(filter(lambda o: o.obstacle_free(start, end), __g_obstacles))
    return len(colliding_obstacles) == 0
