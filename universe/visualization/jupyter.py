import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def show_img(img: np.ndarray, figsize: tuple) -> None:
    """
    Displays an image via pyplot. (Jupyter notebooks trick)

    :param figsize: size of the figure.
    :param img: opencv image (BGR format, numpy array).
    :return:
    """
    plt.figure(figsize=figsize)
    plt.axis("off")

    image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()
