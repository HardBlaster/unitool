from abc import abstractmethod, ABC
from typing import List, Tuple, Optional
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt, patches as mpatches

from src.universe.math.geometry import Line2d


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


class _Layer(ABC):
    """
    A layer that can be drawn on an image.
    """

    def __init__(self, color: Tuple[int, int, int], opacity: float, name: Optional[str] = None):
        """
        Creates a new layer with the given color, opacity and name.

        :param color: color of the layer (RGB)
        :param opacity: opacity of the layer (0-100)
        :param name: name of the layer
        """
        self._opacity = None
        self.set_opacity(opacity)
        self.color = color
        self.name = name

    def set_opacity(self, opacity: float) -> None:
        """
        Sets the opacity of the layer.

        :param opacity: opacity of the layer (0-1)
        """
        self._opacity = opacity

    @abstractmethod
    def draw_on(self, image: np.ndarray) -> np.ndarray:
        ...


class LinesLayer(_Layer):
    """
    A layer with lines that can be drawn on an image.
    """

    def __init__(self, lines: List[Line2d], color: Tuple[int, int, int], endpoints: bool = False,
                 line_width: float = 0.004, opacity: float = 0.75, name: Optional[str] = None):
        """
        Creates a new layer with the given lines, color, opacity and name.

        :param lines: lines to draw
        :param color: color of the lines (RGB)
        :param endpoints: whether to draw circles at the endpoints of the lines
        :param line_width: this value multiplied by the width of the image is the line width in pixels
        :param opacity: opacity of the lines (0-1)
        :param name: name of the layer
        """
        super().__init__(color, opacity, name)
        self.lines = lines
        self.endpoints = endpoints
        self.line_width = line_width

    def draw_on(self, image: np.ndarray) -> np.ndarray:
        """
        Draws the lines on the given image.

        :param image: image to draw the lines on
        :return: image with the lines drawn on it
        """
        return draw_lines(image, self.lines, self.color, self.endpoints, self.line_width, self._opacity)


class MaskLayer(_Layer):
    """
    A layer with a mask that can be drawn on an image.
    """

    def __init__(self, mask: np.ndarray, color: Tuple[int, int, int], threshold: int = 30,
                 opacity: float = 0.75, name: Optional[str] = None):
        """
        Creates a new layer with the given mask, color, opacity and name.

        :param mask: mask to draw
        :param color: color of the mask (RGB)
        :param opacity: opacity of the mask (0-1)
        :param threshold: threshold of the mask (0-255)
        :param name: name of the layer
        """
        super().__init__(color, opacity, name)
        self.mask = mask
        self.threshold = threshold

    def draw_on(self, image: np.ndarray) -> np.ndarray:
        """
        Draws the mask on the given image.

        :param image: image to draw the mask on
        :return: image with the mask drawn on it
        """
        return draw_mask(image, self.mask, self.color, self.threshold, self._opacity)


class LayeredImage:
    """
    An image that can have layers drawn on it.
    """

    def __init__(self, image: np.ndarray):
        """
        Creates a new layered image with the given image. The original image is not modified.

        :param image: image to draw the layers on
        """
        if image is None:
            raise ValueError('Image does not exist')
        self._original = image
        self.grayscale = False
        self._darkness = 0
        self.layers = []
        self.width = image.shape[1]
        self.height = image.shape[0]

    def set_grayscale(self, grayscale: bool) -> None:
        """
        Sets whether the image should be converted to grayscale.

        :param grayscale: whether to convert the image to grayscale
        """
        self.grayscale = grayscale

    def set_darkness(self, darkness: float) -> None:
        """
        Sets the percentage of darkness of the image.

        :param darkness: percentage of darkness (0-1)
        """
        self._darkness = darkness

    def add_layer(self, layer: _Layer) -> None:
        """
        Adds the given layer to the image.

        :param layer: layer to add
        """
        self.layers.append(layer)

    @property
    def image(self) -> np.ndarray:
        """
        Returns the image with the layers drawn on it.

        :return: image with the layers drawn on it
        """
        result = cv.cvtColor(self._original.copy(), cv.COLOR_BGR2RGB)
        # convert the image to grayscale if necessary
        if self.grayscale:
            result = grayscale_image(result)
        # darken the image if necessary
        if self._darkness > 0:
            result = darken_image(result, self._darkness)
        # draw the layers on the image
        for layer in self.layers:
            result = layer.draw_on(result)
        # resize the image if necessary
        if self.width != self._original.shape[1] or self.height != self._original.shape[0]:
            result = cv.resize(result, (self.width, self.height))
        return result

    def plot(self, title: Optional[str] = None, axis: bool = False) -> None:
        """
        Shows a matplotlib plot of the image with the given title. Also shows the legend.

        :param title: title of the plot
        :param axis: whether to show the axis or not
        """

        # create legend
        legend = [mpatches.Patch(color=[x / 255 for x in layer.color], label=layer.name) for layer in self.layers
                  if layer.name]
        if legend:
            plt.legend(handles=legend, loc='lower right')

        if title:
            plt.title(title)
        if not axis:
            plt.axis('off')
        # show plot
        plt.imshow(self.image)

    def save(self, path: str) -> bool:
        """
        Saves the image with the layers drawn on it to the given path.

        :param path: path to save the image to
        :return: whether the image was saved successfully or not
        """
        return cv.imwrite(path, cv.cvtColor(self.image, cv.COLOR_RGB2BGR))

    def resize(self, width: Optional[int] = None, height: Optional[int] = None) -> None:
        """
        Resizes the image to the given width and height.

        :param width: new width of the image
        :param height: new height of the image
        """
        if width:
            if width <= 0:
                raise ValueError('Width must be greater than 0')
            else:
                self.width = width
        if height:
            if height <= 0:
                raise ValueError('Height must be greater than 0')
            else:
                self.height = height

    def scale(self, multiplier: float) -> None:
        """
        Scales the image by the given multiplier.

        :param multiplier: multiplier to scale the image by
        """
        if multiplier <= 0:
            raise ValueError('Multiplier must be greater than 0')

        self.width = round(self.width * multiplier)
        self.height = round(self.height * multiplier)


def grayscale_image(image: np.ndarray) -> np.ndarray:
    """
    Converts the given image to grayscale and then to RGB color space.

    :param image: image to convert (RGB or grayscale)
    :return: grayscale image converted to RGB color space
    """
    if image is None:
        raise ValueError('Image does not exist')

    # if the image is not grayscale, convert it to grayscale
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    # convert the image to RGB color space
    return cv.cvtColor(image, cv.COLOR_GRAY2RGB)


def darken_image(image: np.ndarray, darkness: float = 0.5) -> np.ndarray:
    """
    Darkens the given image.

    :param image: image to darken
    :param darkness: percentage of darkness (0-100)
    :return: darkened image
    """
    if image is None:
        raise ValueError('Image does not exist')
    if not (0. <= darkness <= 1.):
        new_darkness = min(max(0., darkness), 1.)
        print('Darkness percentage must be between 0 and 1.'
              f'The given value ({darkness}) will be interpreted as {new_darkness}.')
        darkness = new_darkness

    return cv.addWeighted(np.zeros_like(image), darkness, image, 1. - darkness, 0.)


def draw_lines(image: np.ndarray, lines: List[Line2d], color: Tuple[int, int, int], endpoints: bool = False,
               line_width: float = 0.004, opacity: float = 0.75) -> np.ndarray:
    """
    Draws the given lines on the given image with the given color.
    Make sure that the lines are scaled to the image size.

    :param image: image to draw the lines on
    :param lines: lines to draw on the image (scaled to the image size)
    :param color: color of the lines (in the same color space as the image)
    :param endpoints: whether to draw circles at the endpoints of the lines
    :param line_width: this value multiplied by the width of the image is the line width in pixels
    :param opacity: opacity of the lines (0-100)
    :return: image with the lines drawn on it
    """

    # return the image if there are no lines to draw
    if not lines:
        return image

    if image is None:
        raise ValueError('Image does not exist')
    if not (0. <= opacity <= 1.):
        new_opacity = min(max(0., opacity), 1.)
        print('Opacity percentage must be between 0 and 1.'
              f'The given value ({opacity}) will be interpreted as {new_opacity}.')
        opacity = new_opacity

    # calculate the line width in pixels based on the image width
    line_width = round(line_width * image.shape[1])

    # create an overlay for the lines
    overlay = np.zeros_like(image)
    for line in lines:
        # round the points to integers
        p1 = (round(line.a.x), round(line.a.y))
        p2 = (round(line.b.x), round(line.b.y))

        # draw line on the overlay
        cv.line(overlay, p1, p2, color, line_width)
        # draw endpoints on the overlay
        if endpoints:
            cv.circle(overlay, p1, line_width, color, -1)
            cv.circle(overlay, p2, line_width, color, -1)

    # draw the overlay on the image
    return cv.addWeighted(overlay, opacity, image, 1., 0.)


def draw_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int],
              threshold: int = 30, opacity: float = 0.75) -> np.ndarray:
    """
    Draws the given mask on the given image.

    :param image: image to draw the mask on
    :param mask: mask to draw
    :param color: color of the mask (in the same color space as the image)
    :param threshold: threshold of the mask (0-255)
    :param opacity: opacity of the mask (0-1)
    :return: image with the mask drawn on it
    """

    # return the image if there is no mask to draw
    if mask is None:
        return image

    if image is None:
        raise ValueError('Image does not exist')
    if not (0. <= opacity <= 1.):
        new_opacity = min(max(0., opacity), 1.)
        print(f'Opacity percentage must be between 0 and 1.'
              f'The given value ({opacity}) will be interpreted as {new_opacity}.')
        opacity = new_opacity

    # resize the mask
    mask = cv.resize(mask, (image.shape[1], image.shape[0]))
    # convert the mask to grayscale
    mask = cv.cvtColor(mask, cv.COLOR_RGB2GRAY)
    # threshold the mask
    mask = np.where(mask < threshold, 0, 1)
    # create overlay
    overlay = np.zeros_like(image)
    # draw mask on the overlay
    overlay[mask > 0] = color

    # draw the overlay on the image
    return cv.addWeighted(overlay, opacity, image, 1., 0.)
