import unittest

import numpy as np

from universe.math.geometry import (Point2d, angle_3points, dist_2d,
                                    rotate_point)


class TestGeometry(unittest.TestCase):
    def test_dist_2d(self):
        self.assertEqual(dist_2d(Point2d(0, 0), Point2d(1, 0)), 1)
        self.assertEqual(dist_2d(Point2d(0, 0), Point2d(-1, 0)), 1)
        self.assertEqual(dist_2d(Point2d(0, 0), Point2d(0, 1)), 1)
        self.assertEqual(dist_2d(Point2d(0, 0), Point2d(0, -1)), 1)
        self.assertEqual(dist_2d(Point2d(0, 0), Point2d(1, 1)), 2 ** .5)
        self.assertEqual(dist_2d(Point2d(0, 0), Point2d(1, -1)), 2 ** .5)
        self.assertEqual(dist_2d(Point2d(0, 0), Point2d(-1, -1)), 2 ** .5)
        self.assertEqual(dist_2d(Point2d(0, 0), Point2d(-1, 1)), 2 ** .5)

    def test_rotate_point(self):
        self.assertEqual(rotate_point(Point2d(1, 0), Point2d(0, 0), np.deg2rad(90)), Point2d(0, 1))
        self.assertEqual(rotate_point(Point2d(0, 1), Point2d(0, 0), np.deg2rad(90)), Point2d(-1, 0))
        self.assertEqual(rotate_point(Point2d(0, 1), Point2d(0, 0), np.deg2rad(-90)), Point2d(1, 0))
        self.assertEqual(rotate_point(Point2d(1, 0), Point2d(10, 11), np.deg2rad(90)), Point2d(21, 2))

    def test_angle_3points(self):
        self.assertEqual(angle_3points(Point2d(-1, 0), Point2d(0, 0), Point2d(1, 0)), np.deg2rad(180))
        self.assertEqual(angle_3points(Point2d(0, 1), Point2d(0, 0), Point2d(0, -1)), np.deg2rad(180))
        self.assertEqual(angle_3points(Point2d(0, 1), Point2d(0, 0), Point2d(1, 0)), np.deg2rad(270))
        self.assertEqual(angle_3points(Point2d(1, 0), Point2d(0, 0), Point2d(0, -1)), np.deg2rad(270))
        self.assertEqual(angle_3points(Point2d(1, 1), Point2d(0, 0), Point2d(0, -1)), np.deg2rad(225))
