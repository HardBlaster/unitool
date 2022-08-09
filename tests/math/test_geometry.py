from universe.math.geometry import (Point2d, angle_3points, dist_2d,
                                    rotate_point)


def test_dist_2d():
    assert dist_2d(Point2d(0, 0), Point2d(1, 0)) == 1
    assert dist_2d(Point2d(0, 0), Point2d(-1, 0)) == 1
    assert dist_2d(Point2d(0, 0), Point2d(0, 1)) == 1
    assert dist_2d(Point2d(0, 0), Point2d(0, -1)) == 1
    assert dist_2d(Point2d(0, 0), Point2d(1, 1)) == 2 ** .5
    assert dist_2d(Point2d(0, 0), Point2d(1, -1)) == 2 ** .5
    assert dist_2d(Point2d(0, 0), Point2d(-1, -1)) == 2 ** .5
    assert dist_2d(Point2d(0, 0), Point2d(-1, 1)) == 2 ** .5


def test_rotate_point():
    assert rotate_point(Point2d(1, 0), Point2d(0, 0), 90) == Point2d(0, 1)
    assert rotate_point(Point2d(0, 1), Point2d(0, 0), 90) == Point2d(-1, 0)
    assert rotate_point(Point2d(0, 1), Point2d(0, 0), -90) == Point2d(1, 0)
    assert rotate_point(Point2d(1, 0), Point2d(10, 11), 90) == Point2d(21, 2)


def test_angle_3points():
    assert angle_3points(Point2d(-1, 0), Point2d(0, 0), Point2d(1, 0)) == 180
    assert angle_3points(Point2d(0, 1), Point2d(0, 0), Point2d(0, -1)) == 180
    assert angle_3points(Point2d(0, 1), Point2d(0, 0), Point2d(1, 0)) == 270
    assert angle_3points(Point2d(1, 0), Point2d(0, 0), Point2d(0, -1)) == 270
    assert angle_3points(Point2d(1, 1), Point2d(0, 0), Point2d(0, -1)) == 225
