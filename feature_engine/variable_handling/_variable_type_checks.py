from datetime import date, datetime

import narwhals as nw
from dateutil.parser import parse as _dateutil_parse

# ---------------------------------------------------------------------------
# narwhals implementation, used for every backend (pandas, polars, etc.)
#
# Flexible date-string recognition (e.g. "01-Jan-2010", "10/11/12", not just
# ISO-8601) is implemented directly on top of `dateutil` - the same library
# pandas.to_datetime delegates to internally for this - so it works
# identically regardless of the underlying dataframe library.
# ---------------------------------------------------------------------------


def _nw_is_date_or_datetime(dtype) -> bool:
    # nw.selectors.datetime() only matches Datetime, not Date, so this needs
    # its own explicit check.
    return isinstance(dtype, (nw.Date, nw.Datetime))


_DATE_PARSE_DEFAULT_1 = datetime(1, 1, 1, 1, 1, 1)
_DATE_PARSE_DEFAULT_2 = datetime(2, 2, 2, 2, 2, 2)
_DATETIME_FIELDS = ("year", "month", "day", "hour", "minute", "second")


def _looks_like_date_string(value: str) -> bool:
    # dateutil.parser.parse() fills in any date/time component that isn't
    # present in the string from a `default` datetime, so a bare number like
    # "20" "parses" successfully as day=20 - it would wrongly be treated as a
    # date. Parsing twice, with two defaults that differ in every field,
    # reveals which fields were actually present in the string: those are the
    # fields that agree between the two parses. Requiring at least 2 fields to
    # be corroborated this way rejects bare numbers while still accepting real
    # dates (including non-ISO formats like "01-Jan-2010") and bare times
    # (like "21:45:23").
    try:
        first = _dateutil_parse(value, default=_DATE_PARSE_DEFAULT_1)
        second = _dateutil_parse(value, default=_DATE_PARSE_DEFAULT_2)
    except (ValueError, OverflowError, TypeError):
        return False

    corroborated = sum(
        1 for attr in _DATETIME_FIELDS if getattr(first, attr) == getattr(second, attr)
    )
    return corroborated >= 2


def _nw_is_convertible_to_num(s: "nw.Series") -> bool:
    values = s.drop_nulls().to_list()
    if not values:
        return False
    try:
        for value in values:
            float(value)
    except (ValueError, TypeError):
        return False
    return True


def _nw_is_convertible_to_dt(s: "nw.Series") -> bool:
    values = s.drop_nulls().to_list()
    if not values:
        return False
    for value in values:
        if isinstance(value, (date, datetime)):
            continue
        if not _looks_like_date_string(str(value)):
            return False
    return True


def _nw_categories_are_numeric(s: "nw.Series") -> bool:
    return s.cat.get_categories().dtype.is_numeric()


def _nw_is_categorical_and_is_not_datetime(s: "nw.Series") -> bool:
    if isinstance(s.dtype, nw.Enum):
        # an explicit, user-defined category set is an unambiguous categorical
        # signal, unlike a generic string column, so skip the datetime check
        return True

    if isinstance(s.dtype, nw.Categorical):
        # check for datetime only if the categories are not numeric, because
        # a numeric-backed categorical (pandas-only - polars categories are
        # always string-backed) can never hold dates
        return _nw_categories_are_numeric(s) or not _nw_is_convertible_to_dt(s)

    if isinstance(s.dtype, (nw.String, nw.Object)):
        # check for datetime only if the column cannot be cast as numeric,
        # because if it could, it would be a numeric column, not a date
        return _nw_is_convertible_to_num(s) or not _nw_is_convertible_to_dt(s)

    return False


def _nw_is_categorical_and_is_datetime(s: "nw.Series") -> bool:
    if isinstance(s.dtype, nw.Enum):
        return False

    if isinstance(s.dtype, nw.Categorical):
        return not _nw_categories_are_numeric(s) and _nw_is_convertible_to_dt(s)

    if isinstance(s.dtype, (nw.String, nw.Object)):
        return not _nw_is_convertible_to_num(s) and _nw_is_convertible_to_dt(s)

    return False
