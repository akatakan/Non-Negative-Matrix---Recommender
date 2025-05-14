from collections.abc import Sequence
from typing import Union
import pandas as pd


def read_dat(
    file: str,
    sep: str,
    columns: Sequence[int] | Sequence[str] | None = None,
    *args,
    **kwargs,
) -> pd.DataFrame:
    data: pd.DataFrame = pd.read_csv(
        file, sep=sep, names=columns, header=None, encoding="latin-1", *args, **kwargs
    )
    return data
