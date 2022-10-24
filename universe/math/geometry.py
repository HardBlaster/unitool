from typing import NamedTuple, Tuple, Union, Optional

import numpy as np

Point3d = NamedTuple('Point3d', x=float, y=float, z=float)


class Point2d:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __sub__(self, other: 'Point2d') -> 'Point2d':
        return Point2d(self.x - other.x, self.y - other.y)

    def __add__(self, other: 'Point2d') -> 'Point2d':
        return Point2d(self.x + other.x, self.y + other.y)

    def __abs__(self) -> 'Point2d':
        return Point2d(abs(self.x), abs(self.y))

    def __matmul__(self, other: 'Point2d') -> float:
        return self.x * other.y - self.y * other.x


class Line2d:
    def __init__(self, a: Point2d, b: Point2d):
        self.a = a
        self.b = b

    def direction_to_point(self, point: Point2d) -> float:
        return (point - self.a) @ (self.b - self.a)

    def is_on_line(self, point: Point2d) -> bool:
        pass

    def is_intersect(self, line: 'Line2d', endpoints: bool = True) -> bool:
        d1 = line.direction_to_point(self.a)
        d2 = line.direction_to_point(self.b)
        d3 = self.direction_to_point(line.a)
        d4 = self.direction_to_point(line.b)

        if ((d1 > 0 > d2) or (d1 < 0 < d2)) and ((d3 > 0 > d4) or (d3 < 0 < d4)):
            return True

        if not endpoints:
            return False

        elif d1 == 0 and line.is_on_line(self.a):
            return True
        elif d2 == 0 and line.is_on_line(self.b):
            return True
        elif d3 == 0 and self.is_on_line(line.a):
            return True

        return d4 == 0 and self.is_on_line(line.b)


class LineFunction:
    def __init__(self, c: float = .0):
        self.c = c

    def intersection(self, line: 'LineFunction') -> Point2d:
        pass

    def move_point(self, point: Point2d, dist: float) -> Point2d:
        pass

    def is_parallel(self, line: 'LineFunction') -> bool:
        pass

    def perpendicular(self, point: Point2d) -> 'LineFunction':
        pass

    def calculate_y(self, x: float) -> float:
        pass

    def __call__(self, x, *args, **kwargs) -> float:
        return self.calculate_y(x)


class XLine2dFunction(LineFunction):
    def __str__(self):
        return f'x={self.c}'

    def calculate_y(self, x: float) -> float:
        raise ValueError(f'XLineFunction represents a line parallel to the Y axis, therefore y is an interval [-inf, '
                         f'+inf] at {x=}. Undefined at other x values.')

    def perpendicular(self, point: Point2d) -> LineFunction:
        return YLine2dFunction(m=.0, c=point.y)

    def is_parallel(self, line: 'XLine2dFunction') -> bool:
        return type(line) == XLine2dFunction

    def move_point(self, point: Point2d, dist: float) -> Point2d:
        if point.x != self.c:
            raise ValueError(f'Point {point=} is not on line {self=}')

        return Point2d(point.x, point.y + dist)

    def intersection(self, line: Union['XLine2dFunction', 'YLine2dFunction']) -> Optional[Point2d]:
        if type(line) == XLine2dFunction:
            return None

        return Point2d(self.c, line(self.c))


class YLine2dFunction(LineFunction):
    def __init__(self, c: float = .0, m: float = 1.):
        super().__init__(c)
        self.m = m

    def __str__(self):
        return f'y = {self.m}x + {self.c}'

    @property
    def norm_vec(self):
        p1 = Point2d(0, self.c)
        p2 = Point2d(1, self(1))
        dist = dist_2d(p1, p2)

        return Point2d(1/dist, (p2.y - p1.y)/dist)

    def calculate_y(self, x: float) -> float:
        return self.m * x + self.c

    def perpendicular(self, point):
        return XLine2dFunction(c=point.x) if not self.m else YLine2dFunction(m=-1/self.m, c=point.y - self.m * point.x)

    def is_parallel(self, line: 'YLine2dFunction') -> bool:
        return type(line) == YLine2dFunction and self.m == line.m

    def move_point(self, point: Point2d, dist: float) -> Point2d:
        if point.y != self(point.x):
            raise ValueError(f'Point {point=} is not on line {self=}')

        norm_vec = self.norm_vec

        return Point2d(point.x + dist*norm_vec.x, dist*norm_vec.y)

    def intersection(self, line: Union[XLine2dFunction, 'YLine2dFunction']) -> Point2d:
        if type(line) == XLine2dFunction:
            return Point2d(line.c, self(line.c))

        x = (line.c - self.c) / (self.m - line.m)

        return Point2d(x, self(x))


class Rectangle:
    def __init__(self, top_left: Point2d, bottom_right: Point2d):
        self.top_left = top_left
        self.bottom_right = bottom_right

    @property
    def area(self) -> float:
        wh = abs(self.bottom_right - self.top_left)

        return wh.x * wh.y


def make_2d_line_function(p1: Point2d, p2: Point2d) -> Union[XLine2dFunction, YLine2dFunction]:
    if p1.x == p2.x:
        return XLine2dFunction(p1.x)

    m = (p1.y - p2.y) / (p1.x - p2.x)
    c = p1.y - m * p1.x

    return YLine2dFunction(c, m)


def dist_2d(p1: Point2d, p2: Point2d) -> float:
    """
    Calculates the distance between two points on a plane.

    :param p1: point 1.
    :param p2: point 2.
    :return: distance
    """
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


def dist_3d(p1: Point3d, p2: Point3d) -> float:
    """
    Calculates the distance between two points in space.

    :param p1: point 1.
    :param p2: point 2.
    :return: distance.
    """
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5


def rotate_point(point: Point2d, around: Point2d, angle: float) -> Tuple[int, int]:
    """
    Rotates a point around another point by an angle.

    :param point: point to rotate.
    :param around: center point, rotate around this point.
    :param angle: angle in radians.
    :return: x and y coordinates after rotation.
    """
    translated_point = Point2d(point.x - around.x, point.y - around.y)
    sin_alpha, cos_alpha = np.sin(angle), np.cos(angle)

    return int(cos_alpha * translated_point.x - sin_alpha * translated_point.y + around.x), int(
        sin_alpha * translated_point.x + cos_alpha * translated_point.y + around.y)


def angle_3points(p1: Point2d, p2: Point2d, p3: Point2d) -> float:
    """
    Calculates the angle ABC.

    :param p1: startpoint.
    :param p2: angle point.
    :param p3: endpoint.
    :return: angle in radians.
    """
    angle = np.arctan2(p3.y - p2.y, p3.x - p2.x) - np.arctan2(p1.y - p2.y, p1.x - p2.x)

    return angle + 2 * np.pi if angle < 0 else angle
