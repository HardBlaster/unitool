import string
import time
from datetime import datetime
from functools import reduce
from typing import List, Tuple, Union

import numpy as np

from src.universe.utils import reduce_mult

LETTERS = list(string.ascii_letters)


def bools(size: Union[int, Tuple[int], List[int]]) -> np.ndarray:
    """
    Generates a numpy array filled with 0s and 1s.

    :param size: size of the array.
    :return: numpy array.
    """
    return np.random.randint(0, 1, size=size)


def strings(size: Union[int, Tuple[int], List[int]], min_len: int = 1, max_len: int = 10) -> np.ndarray:
    """
    Generates a numpy array filled with strings.

    :param size: size of the array.
    :param min_len: minimum length of a random string.
    :param max_len: maximum length of a random string.
    :return: numpy array.
    """
    return np.array([
        [''.join(np.random.choice(list(string.ascii_letters), size=np.random.randint(min_len, max_len+1)))]
        for _ in range(reduce(reduce_mult, size)
                       if not isinstance(size, int)
                       else size)
    ]).reshape(size)


def datetimes(size: Union[int, Tuple[int], List[int]], start_date: int = 0,
              end_date: int = int(time.time())) -> np.ndarray:
    """
    Generates numpy array filled with random date and time values.

    :param size: size of the array.
    :param start_date: earliest date and time in ms.
    :param end_date: latest date and time in ms.
    :return: numpy array.
    """
    return np.array([
        datetime.fromtimestamp(np.random.randint(start_date, end_date))
        for _ in range(reduce(reduce_mult, size))
    ]).reshape(size)
