import pickle as pkl
import unittest

from universe.data import cvat

DATA_BASE_PATH = 'test_data/data/cvat/'
ANNOTATIONS = DATA_BASE_PATH + 'annotations.xml'
POLYLINES = DATA_BASE_PATH + 'polylines.pkl'
BOXES = DATA_BASE_PATH + 'boxes.pkl'


class TestCvat(unittest.TestCase):

    def test_load_polylines(self):
        with open(POLYLINES, 'rb') as pl_file:
            polylines = pkl.load(pl_file)

        self.assertEqual(cvat.load_polylines(ANNOTATIONS), polylines)

    def test_load_boxes(self):
        with open(BOXES, 'rb') as box_file:
            boxes = pkl.load(box_file)

        self.assertEqual(cvat.load_boxes(ANNOTATIONS), boxes)
