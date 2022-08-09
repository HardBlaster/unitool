import pickle as pkl
import unittest

from universe.data import cvat

DATA_BASE_PATH = 'test_data/data/cvat/'
ANNOTATIONS = DATA_BASE_PATH + 'annotations.xml'
POLYLINES = DATA_BASE_PATH + 'polylines.pkl'
BOXES = DATA_BASE_PATH + 'boxes.pkl'


class TestCvat(unittest.TestCase):

    def test_load_polylines(self):
        with open(POLYLINES, 'rb') as df_file:
            parsed_df = pkl.load(df_file)

        self.assertTrue(cvat.load_polylines(ANNOTATIONS).equals(parsed_df))

    def test_load_boxes(self):
        with open(BOXES, 'rb') as df_file:
            parsed_df = pkl.load(df_file)

        self.assertTrue(cvat.load_boxes(ANNOTATIONS).equals(parsed_df))
