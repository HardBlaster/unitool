from typing import Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def show_img(img: np.ndarray, figsize: Tuple[int, int], conversion: int = cv.COLOR_BGR2RGB) -> None:
    """
    Displays an image via pyplot. (Jupyter notebooks trick)

    :param conversion: color conversion.
    :param figsize: size of the figure.
    :param img: opencv image (BGR format, numpy array).
    :return:
    """
    plt.figure(figsize=figsize)
    plt.axis("off")

    image = cv.cvtColor(img, conversion) if conversion else img
    plt.imshow(image)
    plt.show()
