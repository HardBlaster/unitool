import sys
from typing import Union, Optional

import numpy as np
import pandas as pd

from universe import random
from universe.data.common import generate_str_index


def random_dataframe(rows: int, columns: Union[int, list, tuple], dtypes: Optional[dict] = None):
    if isinstance(columns, int):
        char_count = 26
        columns = generate_str_index(columns // char_count + 1)[:columns]

    if not dtypes:
        df = pd.DataFrame(np.random.random((rows, len(columns))), columns=columns)

    else:
        df = pd.DataFrame([[None] * len(columns)] * rows, columns=columns)

        for col in df.columns:
            if col in dtypes.keys():
                if dtypes[col] in ('int', int):
                    df.loc[:, [col]] = np.random.randint(-sys.maxsize, sys.maxsize, size=[rows, 1])

                elif dtypes[col] in ('float', float):
                    df.loc[:, [col]] = np.random.random([rows, 1])

                elif dtypes[col] in ('bool', bool):
                    df.loc[:, [col]] = random.bools([rows, 1])

                elif dtypes[col] in ('str', str):
                    df.loc[:, [col]] = random.strings([rows, 1])

                elif dtypes[col] == 'datetime64':
                    df.loc[:, [col]] = random.datetimes([rows, 1])

                else:
                    raise TypeError(
                        f'Type {dtypes[col]} is not supported. Available types: int, float, str, datetime64.'
                    )

            else:
                df.loc[:, [col]] = np.random.random(rows)

        df = df.astype(dtypes)

    return df
