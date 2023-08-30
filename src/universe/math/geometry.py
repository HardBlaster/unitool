from typing import NamedTuple, Optional, Union

import numpy as np

Point3d = NamedTuple('Point3d', x=float, y=float, z=float)


class Point2d:
    """
    Represents a 2-dimensional point.
    """

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'({self.x}, {self.y})'

    def __eq__(self, other: 'Point2d') -> bool:
        return self.x == other.x and self.y == other.y

    def __sub__(self, other: 'Point2d') -> 'Point2d':
        return Point2d(self.x - other.x, self.y - other.y)

    def __add__(self, other: 'Point2d') -> 'Point2d':
        return Point2d(self.x + other.x, self.y + other.y)

    def __abs__(self) -> 'Point2d':
        return Point2d(abs(self.x), abs(self.y))

    def __matmul__(self, other: 'Point2d') -> float:
        """
        Cross-product: x1*y2 - y1*x2.

        :param other: the other point.
        :return: cross-product of the 2 points' coordinates.
        """
        return self.x * other.y - self.y * other.x


class Line2d:
    """
    Represents a 2-dimensional line between two points.
    """

    def __init__(self, a: Point2d, b: Point2d):
        self.a = a
        self.b = b

    def __eq__(self, other: 'Line2d') -> bool:
        return (self.a == other.a and self.b == other.b) or (self.a == other.b and self.b == other.a)

    def __repr__(self):
        return f'{self.a}, {self.b}'

    def direction_to_point(self, point: Point2d) -> float:
        """
        Calculates the cross-product of two vectors.
        First vector: offset of 'point' based on point 'a' of the line by axis (x_offset, y_offset).
        Second vector: offset of point 'b' of line based on point 'a' of the line by axis (x_offset, y_offset).
        The cross-product of these vectors equals to +-2 * (y_offset of 'point' and the projection of 'point' on the
        line (extended on the interval [-inf, +inf])). The sign depends on the relative position of the point. If the
        point is in the counter-clockwise direction (on the top of the line) of the line 'ab' the value is negative, it
        is positive otherwise.

        :param point: 2-dimensional point.
        :return: +-2 * the distance between the point end the point's projection on the line.
        """
        # TODO: rename it
        return (point - self.a) @ (self.b - self.a)

    def is_in_segment(self, point: Point2d) -> bool:
        """
        Checks if the given point is on the area of the rectangle having the line as diagonal.

        :param point: 2-dimensional point.
        :return: true if 'point' is on the area of the rectangle, false otherwise.
        """
        # TODO: maybe rename it
        return min(self.a.x, self.b.x) <= point.x <= max(self.a.x, self.b.x) and \
            min(self.a.x, self.b.x) <= point.x <= max(self.a.x, self.b.x)

    def is_intersect(self, line: 'Line2d', endpoints: bool = True) -> bool:
        """
        Check if 'line' intersects the line.

        :param line: 2-dimensional line.
        :param endpoints: if true, the function returns true if the lines intersect at the endpoints.
        :return: true if the lines intersect, false otherwise.
        """
        # TODO: check if the function works, when the two lines have the same line function, but the Line2d objects
        #  represent it on different intervals. (two endpoints are equal)
        if self == line:
            return True

        d1 = line.direction_to_point(self.a)
        d2 = line.direction_to_point(self.b)
        d3 = self.direction_to_point(line.a)
        d4 = self.direction_to_point(line.b)

        return (((d2 < 0 < d1) or (d1 < 0 < d2)) and ((d4 < 0 < d3) or (d3 < 0 < d4))) or (
                endpoints and ((d1 == 0 and line.is_in_segment(self.a)) or
                               (d2 == 0 and line.is_in_segment(self.b)) or
                               (d3 == 0 and self.is_in_segment(line.a)) or
                               (d4 == 0 and self.is_in_segment(line.b))))


class LineFunction:
    """
    Parent class for abstractions purposes.
    """
    def __init__(self, c: float = .0):
        self.c = c

    def __call__(self, x, *args, **kwargs) -> float:
        return self.calculate_y(x)

    def intersection(self, line: 'LineFunction') -> Point2d:
        """
        Calculates the intersection point of 'line'.

        :param line: 2-dimensional line function.
        :return: point of intersection if it exists.
        """
        pass

    def move_point(self, point: Point2d, dist: float) -> Point2d:
        """
        Moves 'point' on the line by the given distance.

        :param point: 2-dimensional point.
        :param dist: distance to the target position.
        :return: point 'dist' units away from 'point' on the line.
        """
        pass

    def is_parallel(self, line: 'LineFunction') -> bool:
        """
        Checks if 'line' is parallel.

        :param line: 2-dimensional line function.
        :return: true if the given line is parallel, false otherwise.
        """
        pass

    def perpendicular(self, point: Point2d) -> 'LineFunction':
        """
        Creates a perpendicular line and which through 'point'.

        :param point: 2-dimensional point.
        :return: perpendicular line function.
        """
        pass

    def calculate_y(self, x: float) -> float:
        """
        Calculates y at 'x'.

        :param x: coordinate.
        :return: y coordinate.
        """
        pass

    @staticmethod
    def make_2d_line_function(p1: Point2d, p2: Point2d) -> Union['XLine2dFunction', 'YLine2dFunction']:
        """
        Creates a line function which contains the line described by 'p1' and 'p2'.

        :param p1: start of line.
        :param p2: end of line.
        :return: line function.
        """
        if p1.x == p2.x:
            return XLine2dFunction(p1.x)

        m = (p1.y - p2.y) / (p1.x - p2.x)
        c = p1.y - m * p1.x

        return YLine2dFunction(c, m)


class XLine2dFunction(LineFunction):
    """
    Class for representing lines parallel to the Y-axis: x = c.
    """

    def __repr__(self):
        return f'x = {self.c}'

    def calculate_y(self, x: float) -> None:
        """
        Raises error, since x=c function has no single y value.

        :param x: coordinate.
        """
        # TODO: check if undefined is the correct mathematical phrasing.
        raise ValueError(f'XLineFunction represents a line parallel to the Y axis, therefore y is an interval [-inf, '
                         f'+inf] at {x=}. Undefined at other x values.')

    def perpendicular(self, point: Point2d) -> 'YLine2dFunction':
        """
        Creates a line function which represents a perpendicular line and goes through 'point'.

        :param point: 2-dimensional point.
        :return: perpendicular line function.
        """
        return YLine2dFunction(m=.0, c=point.y)

    def is_parallel(self, line: 'XLine2dFunction') -> bool:
        """
        Checks if 'line' is parallel.

        :param line: 2-dimensional line function.
        :return: true if the given line is parallel, false otherwise.
        """
        return type(line) == XLine2dFunction

    def move_point(self, point: Point2d, dist: float) -> Point2d:
        """
        Moves 'point' on the line by the given 'dist' units if the line goes through 'point', raises error otherwise.

        :param point: 2-dimensional point.
        :param dist: distance to the target position.
        :return: point 'dist' units away from 'point' on the line.
        """
        if point.x != self.c:
            raise ValueError(f'Point {point=} is not on line {self=}')

        return Point2d(point.x, point.y + dist)

    def intersection(self, line: Union['XLine2dFunction', 'YLine2dFunction']) -> Optional[Point2d]:
        """
        Calculates the intersection point of 'line'. If 'line' is x = c type of function it does not intersect with the
        line, since they are parallel, or they are the same line. In this case the returned value is None.

        :param line: 2-dimensional line function.
        :return: point of intersection if it exists.
        """
        # TODO: should it raise an error instead? like in calculate_y
        if type(line) == XLine2dFunction:
            return None

        return Point2d(self.c, line(self.c))


class YLine2dFunction(LineFunction):
    """
    Class for representing lines not parallel to the Y-axis: y = mx + c.
    """

    def __init__(self, c: float = .0, m: float = 1.):
        super().__init__(c)
        self.m = m

    def __str__(self):
        return f'y = {self.m}x + {self.c}'

    def __repr__(self):
        return self.__str__()

    @property
    def dir_vec(self) -> Point2d:
        """
        Calculates the direction vector of the line.

        :return: direction vector.
        """
        p1 = Point2d(0, self.c)
        p2 = Point2d(1, self(1))
        dist = dist_2d(p1, p2)

        return Point2d(1 / dist, (p2.y - p1.y) / dist)

    def calculate_y(self, x: float) -> float:
        """
        Calculates y at 'x'.

        :param x: coordinate.
        :return: y coordinate.
        """
        return self.m * x + self.c

    def perpendicular(self, point) -> Union[XLine2dFunction, 'YLine2dFunction']:
        """
        Creates a line function which represents a perpendicular line and goes through 'point'.

        :param point: 2-dimensional point.
        :return: perpendicular line function.
        """
        return XLine2dFunction(c=point.x) if not self.m else YLine2dFunction(m=-1 / self.m,
                                                                             c=point.y - (-1 / self.m) * point.x)

    def is_parallel(self, line: 'YLine2dFunction') -> bool:
        """
        Checks if 'line' is parallel.

        :param line: 2-dimensional line function.
        :return: true if the given line is parallel, false otherwise.
        """
        # TODO: do identical lines count as parallel? (same goes for X line).
        return type(line) == YLine2dFunction and self.m == line.m

    def move_point(self, point: Point2d, dist: float, eps = 1e-5) -> Point2d:
        """
        Moves 'point' on the line by the given 'dist' units if the line goes through 'point', raises error otherwise.

        :param point: 2-dimensional point.
        :param dist: distance to the target position.
        :param eps: epsilon for floating point comparison. (default: 0.00001)
        :return: point 'dist' units away from 'point' on the line.
        """
        if abs(point.y - self(point.x)) > eps:
            raise ValueError(f'Point {point} is not on line {self}')

        dir_vec = self.dir_vec

        return Point2d(point.x + dist * dir_vec.x, point.y + dist * dir_vec.y)

    def intersection(self, line: Union[XLine2dFunction, 'YLine2dFunction']) -> Optional[Point2d]:
        """
        Calculates the intersection point of 'line'. If the two lines are parallel None is returned.

        :param line: 2-dimensional line function.
        :return: point of intersection if it exists.
        """
        if type(line) == XLine2dFunction:
            return Point2d(line.c, self(line.c))

        if self.is_parallel(line):
            return None

        x = (line.c - self.c) / (self.m - line.m)

        return Point2d(x, self(x))


class Rectangle:
    """
    Class for representing a rectangle by its top-left and bottom-right corners.
    """

    def __init__(self, top_left: Point2d, bottom_right: Point2d):
        self.top_left = top_left
        self.bottom_right = bottom_right

    @property
    def area(self) -> float:
        """
        Calculates the area of the rectangle.

        :return: area of the rectangle.
        """
        wh = abs(self.bottom_right - self.top_left)

        return wh.x * wh.y


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


def rotate_point(point: Point2d, around: Point2d, angle: float) -> Point2d:
    """
    Rotates a point around another point by an angle.

    :param point: point to rotate.
    :param around: center point, rotate around this point.
    :param angle: angle in radians.
    :return: x and y coordinates after rotation.
    """
    # TODO: move to Point2d class.
    translated_point = Point2d(point.x - around.x, point.y - around.y)
    sin_alpha, cos_alpha = np.sin(angle), np.cos(angle)

    return Point2d(
        round(cos_alpha * translated_point.x - sin_alpha * translated_point.y + around.x, 10),
        round(sin_alpha * translated_point.x + cos_alpha * translated_point.y + around.y, 10))


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
