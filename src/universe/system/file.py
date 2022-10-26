import glob
import os
from typing import List, Optional


def file_count(path: str, extension: Optional[str] = None, recursive: bool = False, case_sensitive: bool = True) -> int:
    """
    Counts the files with the given extension under the given path. All files included if no extension passed.

    :param case_sensitive: extension case sensitivity.
    :param recursive: recursive search of files.
    :param path: base directory.
    :param extension: file extension.
    :return: number of files.
    """
    return len(get_files(path, extension=extension, recursive=recursive, case_sensitive=case_sensitive))


def get_files(path: str, extension: Optional[str] = None, recursive: bool = False,
              case_sensitive: bool = True) -> List[str]:
    """
    Collects the files with the given extension under the given path. All files included if no extension passed.

    :param case_sensitive: extension case sensitivity.
    :param recursive: recursive search of files.
    :param path: base directory.
    :param extension: file extension.
    :return: list of files.
    """
    pattern = os.path.join(path,
                           '**' if recursive else '',
                           f'*.{extension if case_sensitive else "".join([c.lower() + c.upper() for c in extension])}'
                           if extension else '*')
    files = glob.glob(pattern, recursive=recursive)

    return list(filter(lambda file_path: os.path.isfile(file_path), files))
