from datetime import date, datetime

import narwhals as nw
from dateutil.parser import parser


def _is_date_or_datetime(dtype) -> bool:
    # nw.selectors.datetime() only matches Datetime, not Date, so this needs
    # its own explicit check.
    return isinstance(dtype, (nw.Date, nw.Datetime))


def _looks_like_date_string(value) -> bool:
    # taken from pandas
    # https://github.com/pandas-dev/pandas/blob/cbae8aea4a31a4052736ab0d23f284ff1e78aa06/pandas/_libs/tslibs/parsing.pyx#L666
    try:
        result, _ = parser()._parse(value)
    except TypeError:
        return False

    if result is None:
        return False

    fields = ("year", "month", "day", "hour", "minute", "second")
    found_fields = sum(1 for field in fields if getattr(result, field) is not None)
    return found_fields >= 2


def _is_convertible_to_num(s: "nw.Series") -> bool:
    values = s.drop_nulls().to_list()
    if len(values) == 0:
        return False
    try:
        for value in values:
            float(value)
    except (ValueError, TypeError):
        return False
    return True


def _is_convertible_to_dt(s: "nw.Series") -> bool:
    values = s.drop_nulls().to_list()
    if len(values) == 0:
        return False
    for value in values:
        if isinstance(value, (date, datetime)):
            continue
        if _looks_like_date_string(value) is False:
            return False
    return True


def _is_categories_num(s: "nw.Series") -> bool:
    return s.cat.get_categories().dtype.is_numeric()


def _is_categorical_and_is_not_datetime(s: "nw.Series") -> bool:
    if isinstance(s.dtype, nw.Enum):
        # an explicit, user-defined category set is an unambiguous categorical
        # signal, unlike a generic string column, so skip the datetime check
        return True

    if isinstance(s.dtype, nw.Categorical):
        # check for datetime only if the categories are not numeric, because
        # a numeric-backed categorical (pandas-only - polars categories are
        # always string-backed) can never hold dates
        categories_are_numeric = _is_categories_num(s)
        is_convertible_to_dt = _is_convertible_to_dt(s)
        return categories_are_numeric is True or is_convertible_to_dt is False

    if isinstance(s.dtype, (nw.String, nw.Object)):
        # check for datetime only if the column cannot be cast as numeric,
        # because if it could, it would be a numeric column, not a date
        is_convertible_to_num = _is_convertible_to_num(s)
        is_convertible_to_dt = _is_convertible_to_dt(s)
        return is_convertible_to_num is True or is_convertible_to_dt is False

    return False


def _is_categorical_and_is_datetime(s: "nw.Series") -> bool:
    if isinstance(s.dtype, nw.Enum):
        return False

    if isinstance(s.dtype, nw.Categorical):
        categories_are_numeric = _is_categories_num(s)
        is_convertible_to_dt = _is_convertible_to_dt(s)
        return categories_are_numeric is False and is_convertible_to_dt is True

    if isinstance(s.dtype, (nw.String, nw.Object)):
        is_convertible_to_num = _is_convertible_to_num(s)
        is_convertible_to_dt = _is_convertible_to_dt(s)
        return is_convertible_to_num is False and is_convertible_to_dt is True

    return False
