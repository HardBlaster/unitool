import cv2 as cv
import numpy as np


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotates an image by the angle.

    :param img: opencv image.
    :param angle: angle in degrees.
    :return: rotated image.
    """
    rows, cols, _ = img.shape
    rot_mat = cv.getRotationMatrix2D((cols // 2, rows // 2), angle, 1)

    return cv.warpAffine(img, rot_mat, (cols, rows))
