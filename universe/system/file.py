import glob
import os
from typing import List, Optional


def file_count(path: str, extension: Optional[str] = None, recursive: Optional[bool] = False) -> int:
    """
    Counts the files with the given extension under the given path. All files included if no extension passed.

    :param recursive: recursive search of files.
    :param path: base directory.
    :param extension: file extension.
    :return: number of files.
    """
    return len(get_files(path, extension, recursive))


def get_files(path: str, extension: Optional[str] = None, recursive: Optional[bool] = False) -> List[str]:
    """
    Collects the files with the given extension under the given path. All files included if no extension passed.

    :param recursive: recursive search of files.
    :param path: base directory.
    :param extension: file extension.
    :return: list of files.
    """
    files = glob.glob(os.path.join(path, '**' if recursive else '', f'*.{extension}' if extension else '*'),
                      recursive=recursive)

    return list(filter(lambda file_path: os.path.isfile(file_path), files))
