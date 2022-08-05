import pickle as pkl

from universe.data import cvat

DATA_BASE_PATH = 'test_data/data/cvat/'
ANNOTATIONS = DATA_BASE_PATH + 'annotations.xml'
POLYLINES = DATA_BASE_PATH + 'polylines.pkl'
BOXES = DATA_BASE_PATH + 'boxes.pkl'


def test_load_polylines():
    with open(POLYLINES, 'rb') as df_file:
        parsed_df = pkl.load(df_file)

    assert cvat.load_polylines(ANNOTATIONS).equals(parsed_df)


def test_load_boxes():
    with open(BOXES, 'rb') as df_file:
        parsed_df = pkl.load(df_file)

    assert cvat.load_boxes(ANNOTATIONS).equals(parsed_df)
