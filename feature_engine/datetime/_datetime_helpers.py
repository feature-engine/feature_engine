"""Functions to parse datetime variables, or the index, into numpy arrays.

The datetime transformers compute their features in numpy, so they call these
functions instead of handling each dataframe library themselves.

All functions take the dataframe returned by `check_X`: a pandas dataframe, or a
narwhals dataframe for the other libraries.
"""

from datetime import timezone
from typing import List, Optional, Sequence, Union

import narwhals as nw
import narwhals.dependencies as nwd
import numpy as np
from dateutil.parser import parse as _dateutil_parse
from narwhals.typing import IntoDataFrame

from feature_engine.variable_handling._variable_type_checks import (
    _is_categorical_and_is_datetime,
)

DataFrame = Union[IntoDataFrame, nw.DataFrame]


def datetime_to_numpy(
    X: DataFrame,
    variables: Sequence[Union[str, int]],
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: Optional[bool] = None,
    format: Optional[str] = None,
    local_time: bool = False,
    dateutil_fallback: bool = False,
) -> List[np.ndarray]:
    """
    Parses the variables to datetime and returns them as numpy arrays.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    variables : list
        The names of the datetime variables, or of the variables that can be
        parsed to datetime.

    dayfirst, yearfirst, utc : bool
        Parsing options of `pandas.to_datetime`. Only used with pandas, except
        that `dayfirst`, `yearfirst` and `utc` also apply to the dateutil
        fallback.

    format : str, default=None
        The strftime format to parse strings, e.g. "%d/%m/%Y".

    local_time : bool, default=False
        numpy datetimes have no time zone. If True, time zone aware variables keep
        their local time; if False, they are converted to UTC.

    dateutil_fallback : bool, default=False
        If True and `format` is None, strings that narwhals can't parse are parsed
        value by value with dateutil. Not used with pandas.

    Returns
    -------
    arrays : list
        One 1D numpy datetime64 array per variable, in the order of `variables`.
        A pandas variable that does not parse to a single datetime dtype, for
        example one with mixed time zones, is returned with dtype object.
    """
    # pandas is faster than narwhals.
    if not isinstance(X, nw.DataFrame):
        to_datetime = nwd.get_pandas().to_datetime
        arrays = []
        for var in variables:
            parsed = X[var]
            # to_datetime() scans the values even if they are datetime already.
            if parsed.dtype.kind != "M" or utc is True:
                parsed = to_datetime(
                    parsed,
                    dayfirst=dayfirst,
                    yearfirst=yearfirst,
                    utc=utc,
                    format=format,
                )
            if getattr(parsed.dtype, "tz", None) is not None:
                if local_time is True:
                    parsed = parsed.dt.tz_localize(None)
                else:
                    parsed = parsed.dt.tz_convert(None)
            arrays.append(parsed.to_numpy())
        return arrays
    else:
        arrays = []
        for var in variables:
            parsed = _parse_narwhals_column(
                X.get_column(var),
                dayfirst=dayfirst,
                yearfirst=yearfirst,
                utc=utc,
                format=format,
                dateutil_fallback=dateutil_fallback,
            )
            # to_numpy() returns time zone aware datetimes in UTC.
            if local_time is True and parsed.dtype.time_zone is not None:
                parsed = parsed.dt.replace_time_zone(None)
            arrays.append(parsed.to_numpy())
        return arrays


def _parse_narwhals_column(
    col: nw.Series,
    dayfirst: bool,
    yearfirst: bool,
    utc: Optional[bool],
    format: Optional[str],
    dateutil_fallback: bool,
) -> nw.Series:
    if isinstance(col.dtype, nw.Datetime):
        return col
    if isinstance(col.dtype, nw.Date):
        # Date has no time, so it is cast to Datetime to extract hours and minutes.
        return col.cast(nw.Datetime())

    col = col.cast(nw.String())
    try:
        return col.str.to_datetime(format=format)
    except Exception:
        if dateutil_fallback is False or format is not None:
            raise
        # narwhals needs one format for the whole column; dateutil guesses the
        # format of each value.
        values = [
            None
            if value is None
            else _dateutil_parse(value, dayfirst=dayfirst, yearfirst=yearfirst)
            for value in col.to_list()
        ]
        if utc is True:
            values = [
                None
                if v is None
                else (
                    v.astimezone(timezone.utc)
                    if v.tzinfo is not None
                    else v.replace(tzinfo=timezone.utc)
                )
                for v in values
            ]
        return nw.new_series(col.name, values, backend=col.implementation)


def index_to_datetime_numpy(
    X: DataFrame,
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: Optional[bool] = None,
    format: Optional[str] = None,
) -> np.ndarray:
    """
    Parses the dataframe index to datetime and returns it as a numpy array.
    Time zone aware indexes keep their local time.

    Parameters
    ----------
    X : pandas or narwhals dataframe
        The dataframe returned by `check_X`.

    dayfirst, yearfirst, utc, format :
        Parsing options of `pandas.to_datetime`.

    Raises
    ------
    TypeError
        If X is not a pandas dataframe, or if the index is not datetime and can't
        be parsed to datetime.

    Returns
    -------
    index : numpy array
        The index as a 1D numpy datetime64 array.
    """
    if isinstance(X, nw.DataFrame):
        raise TypeError(
            "variables='index' requires a pandas dataframe, since only "
            f"pandas dataframes have an index. Got {type(X.to_native())} instead."
        )

    index = X.index
    if index.dtype.kind != "M" and (
        index.dtype.kind in "biufc"
        or _is_categorical_and_is_datetime(
            nw.from_native(index.to_series(), series_only=True)
        )
        is False
    ):
        raise TypeError("The dataframe index is not datetime.")

    parsed = nwd.get_pandas().to_datetime(
        index, dayfirst=dayfirst, yearfirst=yearfirst, utc=utc, format=format
    )
    if parsed.tz is not None:
        parsed = parsed.tz_localize(None)
    return parsed.to_numpy()
