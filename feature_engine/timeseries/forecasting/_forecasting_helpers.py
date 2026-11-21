"""Functions to check and sort the rows of a dataframe in time.

With pandas, the rows are ordered in time by the dataframe's index. Other
dataframes, like polars, have no index: their rows are taken in the order given.
The forecasting transformers call these functions instead of handling each
dataframe library themselves.

All functions take the dataframe returned by `check_X`: a pandas dataframe, or a
narwhals dataframe for the other libraries.
"""

from typing import Any, Union

import narwhals as nw
from narwhals.typing import IntoDataFrame

DataFrame = Union[IntoDataFrame, nw.DataFrame]


def check_index(X: DataFrame, freq: Any) -> None:
    """
    Checks that the rows of the dataframe can be ordered in time. With pandas, the
    index must be unique and not contain missing data. Other dataframes have no
    index, so they can't use `freq`.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    freq : str, list or None
        The `freq` parameter of the transformer.
    """
    # only pandas has an index.
    if not isinstance(X, nw.DataFrame):
        if X.index.isnull().any():
            raise NotImplementedError(
                "The dataframe's index contains NaN values or missing data. "
                "Only dataframes with complete indexes are compatible with "
                "this transformer."
            )

        if X.index.is_unique is False:
            raise NotImplementedError(
                "The dataframe's index does not contain unique values. "
                "Only dataframes with unique values in the index are "
                "compatible with this transformer."
            )

    elif freq is not None:
        raise NotImplementedError(
            "freq is only supported with pandas dataframes, because it uses the "
            "dataframe's DatetimeIndex. With other dataframes, leave freq=None "
            f"to shift the rows by periods. Got {freq} instead."
        )


def sort_by_index(X: DataFrame) -> DataFrame:
    """
    Returns the dataframe with its rows sorted by the index. Dataframes without an
    index are returned as is: their rows are used in the order given.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    Returns
    -------
    X : pandas or narwhals dataframe
        The dataframe with the rows sorted by the index.
    """
    # only pandas has an index.
    if not isinstance(X, nw.DataFrame):
        return X.sort_index()
    else:
        return X
