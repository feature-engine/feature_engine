import narwhals as nw
import pandas as pd
from pandas.api.types import is_object_dtype, is_string_dtype
from pandas.core.dtypes.common import is_datetime64_any_dtype as is_datetime
from pandas.core.dtypes.common import is_numeric_dtype as is_numeric

# ---------------------------------------------------------------------------
# pandas-only implementation.
#
# These functions rely on pandas' flexible, dateutil-backed `pd.to_datetime`
# string guessing and on pandas' `object` dtype (which, unlike any narwhals
# dtype, can hold arbitrary non-string Python objects). Neither has a
# polars/narwhals equivalent, so they are kept exactly as they were before the
# narwhals migration and are only ever called on pandas input. See the `_nw_*`
# functions below for the polars/narwhals-backend equivalents.
# ---------------------------------------------------------------------------


def is_object(s) -> bool:
    return is_object_dtype(s) or is_string_dtype(s)


def _is_categorical_and_is_not_datetime(column: pd.Series) -> bool:
    # check for datetime only if the type of the categories is not numeric
    # because pd.to_datetime throws an error when it is an integer
    if isinstance(column.dtype, pd.CategoricalDtype):
        is_cat = _is_categories_num(column) or not _is_convertible_to_dt(column)

    # check for datetime only if object cannot be cast as numeric because
    # if it could pd.to_datetime would convert it to datetime regardless
    elif is_object(column):
        is_cat = _is_convertible_to_num(column) or not _is_convertible_to_dt(column)

    else:
        is_cat = False

    return is_cat


def _is_categories_num(column: pd.Series) -> bool:
    return is_numeric(column.dtype.categories)


def _is_convertible_to_dt(column: pd.Series) -> bool:
    try:
        var = pd.to_datetime(column, utc=True)
        return is_datetime(var)
    except Exception:
        return False


def _is_convertible_to_num(column: pd.Series) -> bool:
    try:
        ser = pd.to_numeric(column)
    except (ValueError, TypeError):
        ser = column
    return is_numeric(ser)


def _is_categorical_and_is_datetime(column: pd.Series) -> bool:
    # check for datetime only if the type of the categories is not numeric
    # because pd.to_datetime throws an error when it is an integer
    if isinstance(column.dtype, pd.CategoricalDtype):
        is_dt = not _is_categories_num(column) and _is_convertible_to_dt(column)

    # check for datetime only if object cannot be cast as numeric because
    # if it could pd.to_datetime would convert it to datetime regardless
    elif is_object(column):
        is_dt = not _is_convertible_to_num(column) and _is_convertible_to_dt(column)

    else:
        is_dt = False

    return is_dt


# ---------------------------------------------------------------------------
# narwhals implementation, used for every backend other than pandas (polars,
# in practice).
#
# narwhals has no lenient/"try" cast (no `strict=False`, unlike raw polars)
# and its `str.to_datetime()` requires ISO-8601 or an explicit `format=` - it
# cannot reproduce pandas' dateutil-based guessing. So a string column such as
# "01-Jan-2010" or "10/11/12" is not auto-detected as datetime for polars,
# even though it is for pandas. ISO-8601 strings and native Date/Datetime
# columns are detected correctly. Users can always pass `variables` explicitly
# to sidestep this.
# ---------------------------------------------------------------------------


def _nw_is_date_or_datetime(dtype) -> bool:
    # nw.selectors.datetime() only matches Datetime, not Date, so this needs
    # its own explicit check.
    return isinstance(dtype, (nw.Date, nw.Datetime))


def _nw_is_convertible_to_num(s: "nw.Series") -> bool:
    try:
        s.cast(nw.String()).cast(nw.Float64())
    except Exception:
        return False
    return True


def _nw_is_convertible_to_dt(s: "nw.Series") -> bool:
    try:
        s.cast(nw.String()).str.to_datetime()
    except Exception:
        return False
    return True


def _nw_is_categorical_and_is_not_datetime(s: "nw.Series") -> bool:
    if isinstance(s.dtype, nw.Enum):
        # an explicit, user-defined category set is an unambiguous categorical
        # signal, unlike a generic string column, so skip the datetime check
        return True

    if isinstance(s.dtype, nw.Categorical):
        # polars categorical categories are always string-backed, unlike
        # pandas' pd.Categorical, which can have numeric categories
        return not _nw_is_convertible_to_dt(s)

    if isinstance(s.dtype, nw.String):
        return _nw_is_convertible_to_num(s) or not _nw_is_convertible_to_dt(s)

    return False


def _nw_is_categorical_and_is_datetime(s: "nw.Series") -> bool:
    if isinstance(s.dtype, nw.Enum):
        return False

    if isinstance(s.dtype, nw.Categorical):
        return _nw_is_convertible_to_dt(s)

    if isinstance(s.dtype, nw.String):
        return not _nw_is_convertible_to_num(s) and _nw_is_convertible_to_dt(s)

    return False
