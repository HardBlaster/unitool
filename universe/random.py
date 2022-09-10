import string
import time
from datetime import datetime
from functools import reduce
from typing import List, Union, Tuple
from numpy import random

import numpy as np

from universe.utils import reduce_mult

LETTERS = list(string.ascii_letters)


def bools(size: Union[int, Tuple[int], List[int]]) -> np.ndarray:
    return np.random.randint(0, 1, size=size)


def strings(size: Union[int, Tuple[int], List[int]], min_len: int = 1, max_len: int = 10) -> np.ndarray:
    return np.array([
        [''.join(np.random.choice(list(string.ascii_letters), size=np.random.randint(min_len, max_len+1)))]
        for _ in range(reduce(reduce_mult, size)
                       if not isinstance(size, int)
                       else size)
    ]).reshape(size)


def datetimes(size: Union[int, Tuple[int], List[int]], start_date: int = 0,
              end_date: int = int(time.time())) -> np.ndarray:
    return np.array([
        datetime.fromtimestamp(np.random.randint(start_date, end_date))
        for _ in range(reduce(reduce_mult, size))
    ]).reshape(size)

