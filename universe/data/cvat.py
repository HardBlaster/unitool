from xml.etree import ElementTree as et

import pandas as pd


def load_polylines(path: str) -> pd.DataFrame:
    """
    CVAT1.1 polyline annotation extractor. Returns a dataframe containing the following columns:
    - point_count: number of point on the given polyline,
    - coordinates: list of coordinates (x1,y1,x2,y2,...),
    - image: image name.

    :param path: path to the annotation file.
    :return: pandas dataframe.
    """
    polylines = pd.DataFrame(columns=[
        'point_count',
        'coordinates',
        'image'
    ])

    tree = et.parse(path)
    root = tree.getroot()
    for image in root.findall('image'):
        img_name = image.get('name')

        for pl in image.findall('polyline'):

            # coordinates are stored in textual format (x0,y0;x1,y1;...)
            # for some reason cvat stores coord values in subpixel accuracy -> cast the string to float then to int
            coords = [int(float(coord)) for coord in pl.get('points').replace(';', ',').split(',')]
            polylines.loc[len(polylines.index)] = [
                len(coords)//2,
                coords,
                img_name
            ]

    return polylines


def load_boxes(path: str) -> pd.DataFrame:
    """
    CVAT1.1 box annotation extractor. Returns a dataframe containing the following columns:
    - x1: top-left corner's X coordinate,
    - y1: top-left corner's Y coordinate,
    - x2: bottom-right corner's X coordinate,
    - y2: bottom-right corner's Y coordinate,
    - label: annotation label,
    - image: image name.

    :param path: path to the annotation file.
    :return: pandas dataframe
    """
    boxes = pd.DataFrame(columns=[
        'x1',
        'y1',
        'x2',
        'y2',
        'label',
        'image'
    ])

    tree = et.parse(path)
    root = tree.getroot()
    for image in root.findall('image'):
        img_name = image.get('name')

        for box in image.findall('box'):

            # coordinates are stored in textual format (x0,y0;x1,y1;...)
            # for some reason cvat stores coord values in subpixel accuracy -> cast the string to float then to int
            boxes.loc[len(boxes.index)] = [
                int(float(box.get('xtl'))),
                int(float(box.get('ytl'))),
                int(float(box.get('xbr'))),
                int(float(box.get('ybr'))),
                box.get('label'),
                img_name
            ]

    return boxes
