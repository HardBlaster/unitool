from typing import Any, Generator, List


def stacked(lst: List[Any], stack_size: int, step: int) -> Generator:
    """
    Iterates through the list yielding a slice of it. In every iteration yields a list with stack_size values in it. The
    index is stepped by step units.

    :param lst: list.
    :param stack_size: length of the yielded lists.
    :param step: stepper for the index.
    :return: a generator yielding the stacked sub-lists.
    """
    for i in range(0, len(lst)-stack_size+1, step):
        yield lst[i:i+stack_size]
