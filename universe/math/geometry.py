from typing import NamedTuple, Tuple

import numpy as np

Point2d = NamedTuple('Point2d', [('x', float), ('y', float)])
Point3d = NamedTuple('Point3d', [('x', float), ('y', float), ('z', float)])
Rectangle = NamedTuple('Rectangle', [('top_left', Point2d), ('bottom_right', Point2d)])


def dist_2d(p1: Point2d, p2: Point2d) -> float:
    """
    Calculates the distance between two points on a plane.

    :param p1: point 1.
    :param p2: point 2.
    :return: distance
    """
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2) ** 0.5


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


def angle_3points(A: Point2d, B: Point2d, C: Point2d) -> float:
    """
    Calculates the angle ABC.

    :param A: startpoint.
    :param B: angle point.
    :param C: endpoint.
    :return: angle in radians.
    """
    angle = np.arctan2(C.y - B.y, C.x - B.x) - np.arctan2(A.y - B.y, A.x - B.x)

    return angle + 2*np.pi if angle < 0 else angle
