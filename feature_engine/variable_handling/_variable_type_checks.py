import warnings

import pandas as pd
from pandas.core.dtypes.common import is_datetime64_any_dtype as is_datetime
from pandas.core.dtypes.common import is_numeric_dtype as is_numeric
from pandas.core.dtypes.common import is_object_dtype as is_object


def _is_categorical_and_is_not_datetime(column: pd.Series) -> bool:
    # check for datetime only if object cannot be cast as numeric because
    # if it could pd.to_datetime would convert it to datetime regardless
    if is_object(column):
        is_cat = _is_convertible_to_num(column) or not _is_convertible_to_dt(column)

    # check for datetime only if the type of the categories is not numeric
    # because pd.to_datetime throws an error when it is an integer
    elif isinstance(column.dtype, pd.CategoricalDtype):
        is_cat = _is_categories_num(column) or not _is_convertible_to_dt(column)

    return is_cat


def _is_categories_num(column: pd.Series) -> bool:
    return is_numeric(column.dtype.categories)


def _is_convertible_to_dt(column: pd.Series) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return is_datetime(pd.to_datetime(column, errors="ignore", utc=True))


def _is_convertible_to_num(column: pd.Series) -> bool:
    try:
        ser = pd.to_numeric(column)
    except (ValueError, TypeError):
        ser = column
    return is_numeric(ser)


def _is_categorical_and_is_datetime(column: pd.Series) -> bool:
    # check for datetime only if object cannot be cast as numeric because
    # if it could pd.to_datetime would convert it to datetime regardless
    if is_object(column):
        is_dt = not _is_convertible_to_num(column) and _is_convertible_to_dt(column)

    # check for datetime only if the type of the categories is not numeric
    # because pd.to_datetime throws an error when it is an integer
    elif isinstance(column.dtype, pd.CategoricalDtype):
        is_dt = not _is_categories_num(column) and _is_convertible_to_dt(column)

    return is_dt
