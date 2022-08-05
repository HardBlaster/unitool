from typing import List, Tuple

import cv2 as cv
import numpy as np

from universe.iteration import stacked
from universe.math.geometry import Point2d, Rectangle


def draw_rectangles(img: np.ndarray, rectangles: List[Rectangle], color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 4) -> np.ndarray:
    """
    Draws the list of rectangles on the image.

    :param img: opencv image.
    :param rectangles: rectangles (top-right and bottom-left corners).
    :param color: color of the perimeter in RGB.
    :param thickness: perimeter's thickness in pixels.
    :return: image with rectangles.
    """
    for rectangle in rectangles:
        img = cv.rectangle(img, (rectangle.top_left.x, rectangle.top_left.y),
                           (rectangle.bottom_right.x, rectangle.bottom_right.y), color, thickness)

    return img


def draw_polylines(img: np.ndarray, polylines: List[List[Point2d]], color: Tuple[int, int, int],
                   thickness: int = 4) -> np.ndarray:
    """
    Draws the list of polylines on the image.

    :param img: opencv image.
    :param polylines: list of polylines.
    :param color: color of the perimeter in RGB.
    :param thickness: perimeter's thickness in pixels.
    :return: image with rectangles.
    """
    for line in polylines:
        for p1, p2 in stacked(line, 2, 1):
            img = cv.line(img, (p1.x, p1.y), (p2.x, p2.y), color, thickness)

    return img
