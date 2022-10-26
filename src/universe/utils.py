from itertools import product
from typing import Any, List

CHARS: str = 'abcdefghijklmnopqrstuvwxyz'


def generate_str_index(length: int, dictionary: str = CHARS) -> List[str]:
    """
    Generates string indexes. The indexes contain from 1 to 'length' number of sequences. Every sequence is sampled from
    'dictionary'. Every possible variation is generated meaning the length of the returned list is: dict_len^1 + ... +
    dict_len^length.

    :param length: length of the longest index.
    :param dictionary: iterable of characters or strings.
    :return:
    """
    return [idx for x_len_indices in [[''.join(strs) for strs in product(dictionary, repeat=current_len)]
                                      for current_len in range(1, length+1)] for idx in x_len_indices]


def reduce_mult(x: Any, y: Any) -> Any:
    """
    Multiplication function for functools.reduce().

    :param x: object with implemented * operator.
    :param y: object with implemented * operator.
    :return: product of x and y.
    """
    return x * y
