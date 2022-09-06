from typing import Dict, List, Tuple
from xml.etree import ElementTree as et

from universe.iteration import stacked
from universe.math.geometry import Point2d


def load_polylines(path: str, label: str = None) -> Dict[str, List[List[Point2d]]]:
    """
    CVAT1.1 polyline annotation extractor. Returns a dictionary with image names as keys and a list of polylines as
    values.

    :param label: load only this label.
    :param path: path to the annotation file.
    :return: dictionary of polylines. Key: image name, value: list of polylines. Polylines are represented as a list of
    points.
    """
    polylines = {}

    tree = et.parse(path)
    root = tree.getroot()
    for image in root.findall('image'):
        img_name = image.get('name')
        polylines.update({img_name: []})

        for pl in image.findall('polyline'):
            if label and pl.get('label') != label:
                continue

            # coordinates are stored in textual format (x0,y0;x1,y1;...)
            # for some reason cvat stores coord values in subpixel accuracy -> cast the string to float then to int
            coords = [int(float(coord)) for coord in pl.get('points').replace(';', ',').split(',')]
            points = [Point2d(x, y) for x, y in stacked(coords, 2, 2)]

            polylines[img_name].append(points)

    return polylines


def load_boxes(path: str, label: str = None) -> Dict[str, List[Tuple[Point2d, Point2d]]]:
    """
    CVAT1.1 box annotation extractor. Returns a dictionary with image names as keys and a list of tuples containing the
    top-left and bottom-right points of the boxes as values.

    :param label: load only this label.
    :param path: path to the annotation file.
    :return: dictionary of boxes. Key: image name, value: list of tuples.
    """
    boxes = {}

    tree = et.parse(path)
    root = tree.getroot()
    for image in root.findall('image'):
        img_name = image.get('name')
        boxes.update({img_name: []})

        for box in image.findall('box'):
            if label and box.get('label') != label:
                continue

            # coordinates are stored in textual format (x0,y0;x1,y1;...)
            # for some reason cvat stores coord values in subpixel accuracy -> cast the string to float then to int
            boxes[img_name].append((
                Point2d(
                    int(float(box.get('xtl'))),
                    int(float(box.get('ytl'))),
                ),
                Point2d(
                    int(float(box.get('xbr'))),
                    int(float(box.get('ybr'))),
                ),
            ))

    return boxes
