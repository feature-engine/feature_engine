import narwhals as nw
import numpy as np

FEATURES_SUPPORTED = [
    "month",
    "quarter",
    "semester",
    "year",
    "week",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "weekend",
    "month_start",
    "month_end",
    "quarter_start",
    "quarter_end",
    "year_start",
    "year_end",
    "leap_year",
    "days_in_month",
    "hour",
    "minute",
    "second",
]

FEATURES_DEFAULT = [
    "month",
    "year",
    "day_of_week",
    "day_of_month",
    "hour",
    "minute",
    "second",
]

FEATURES_SUFFIXES = {
    "month": "_month",
    "quarter": "_quarter",
    "semester": "_semester",
    "year": "_year",
    "week": "_week",
    "day_of_week": "_day_of_week",
    "day_of_month": "_day_of_month",
    "day_of_year": "_day_of_year",
    "weekend": "_weekend",
    "month_start": "_month_start",
    "month_end": "_month_end",
    "quarter_start": "_quarter_start",
    "quarter_end": "_quarter_end",
    "year_start": "_year_start",
    "year_end": "_year_end",
    "leap_year": "_leap_year",
    "days_in_month": "_days_in_month",
    "hour": "_hour",
    "minute": "_minute",
    "second": "_second",
}

FEATURES_FUNCTIONS = {
    "month": lambda x: x.dt.month,
    "quarter": lambda x: x.dt.quarter,
    "semester": lambda x: np.where(x.dt.month <= 6, 1, 2).astype(np.int64),
    "year": lambda x: x.dt.year,
    "week": lambda x: x.dt.isocalendar().week.astype(np.int64),
    "day_of_week": lambda x: x.dt.dayofweek,
    "day_of_month": lambda x: x.dt.day,
    "day_of_year": lambda x: x.dt.dayofyear,
    "weekend": lambda x: np.where(x.dt.dayofweek <= 4, 0, 1).astype(np.int64),
    "month_start": lambda x: x.dt.is_month_start.astype(np.int64),
    "month_end": lambda x: x.dt.is_month_end.astype(np.int64),
    "quarter_start": lambda x: x.dt.is_quarter_start.astype(np.int64),
    "quarter_end": lambda x: x.dt.is_quarter_end.astype(np.int64),
    "year_start": lambda x: x.dt.is_year_start.astype(np.int64),
    "year_end": lambda x: x.dt.is_year_end.astype(np.int64),
    "leap_year": lambda x: x.dt.is_leap_year.astype(np.int64),
    "days_in_month": lambda x: x.dt.days_in_month.astype(np.int64),
    "hour": lambda x: x.dt.hour,
    "minute": lambda x: x.dt.minute,
    "second": lambda x: x.dt.second,
}


def _nw_quarter(x: nw.Series) -> nw.Series:
    return ((x.dt.month() - 1) // 3) + 1


def _nw_semester(x: nw.Series) -> nw.Series:
    return (x.dt.month() > 6).cast(nw.Int64()) + 1


def _nw_week(x: nw.Series) -> nw.Series:
    # narwhals has no isocalendar(); the "%V" strftime code (ISO week) round-trips
    # correctly on every backend tested (pandas, polars) via to_string().
    return x.dt.to_string("%V").cast(nw.Int64())


def _nw_day_of_week(x: nw.Series) -> nw.Series:
    # narwhals weekday() is 1=Monday..7=Sunday; pandas dayofweek is 0=Monday..6=Sunday.
    return x.dt.weekday() - 1


def _nw_weekend(x: nw.Series) -> nw.Series:
    return (_nw_day_of_week(x) >= 5).cast(nw.Int64())


def _nw_is_month_start(x: nw.Series) -> nw.Series:
    return x.dt.day() == 1


def _nw_is_month_end(x: nw.Series) -> nw.Series:
    # no days_in_month()/is_month_end() in narwhals: a day belongs to the last
    # day of its month iff the next day rolls over into a different month.
    return x.dt.offset_by("1d").dt.month() != x.dt.month()


def _nw_month_start(x: nw.Series) -> nw.Series:
    return _nw_is_month_start(x).cast(nw.Int64())


def _nw_month_end(x: nw.Series) -> nw.Series:
    return _nw_is_month_end(x).cast(nw.Int64())


def _nw_quarter_start(x: nw.Series) -> nw.Series:
    # quarters start in Jan/Apr/Jul/Oct, the only months where month % 3 == 1.
    return (_nw_is_month_start(x) & (x.dt.month() % 3 == 1)).cast(nw.Int64())


def _nw_quarter_end(x: nw.Series) -> nw.Series:
    # quarters end in Mar/Jun/Sep/Dec, the only months where month % 3 == 0.
    return (_nw_is_month_end(x) & (x.dt.month() % 3 == 0)).cast(nw.Int64())


def _nw_year_start(x: nw.Series) -> nw.Series:
    return (_nw_is_month_start(x) & (x.dt.month() == 1)).cast(nw.Int64())


def _nw_year_end(x: nw.Series) -> nw.Series:
    return (_nw_is_month_end(x) & (x.dt.month() == 12)).cast(nw.Int64())


def _nw_leap_year(x: nw.Series) -> nw.Series:
    year = x.dt.year()
    return (((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)).cast(
        nw.Int64()
    )


def _nw_days_in_month(x: nw.Series) -> nw.Series:
    # start of month, plus a month, minus a day = last day of the original month;
    # its day number is the month's length. Handles leap years automatically.
    return x.dt.truncate("1mo").dt.offset_by("1mo").dt.offset_by("-1d").dt.day()


# Narwhals-native equivalents of FEATURES_FUNCTIONS above, used for dataframe
# backends other than pandas. Kept separate from FEATURES_FUNCTIONS 
# merged into one dispatch) because roughly a third of these features (week,
# month_end, quarter_end, quarter_start, year_start, year_end, leap_year,
# days_in_month) benchmarked 2x-53x slower than pandas-native when run through
# narwhals on a pandas backend, so pandas keeps its fast, unchanged native path.
FEATURES_FUNCTIONS_NARWHALS = {
    "month": lambda x: x.dt.month(),
    "quarter": _nw_quarter,
    "semester": _nw_semester,
    "year": lambda x: x.dt.year(),
    "week": _nw_week,
    "day_of_week": _nw_day_of_week,
    "day_of_month": lambda x: x.dt.day(),
    "day_of_year": lambda x: x.dt.ordinal_day(),
    "weekend": _nw_weekend,
    "month_start": _nw_month_start,
    "month_end": _nw_month_end,
    "quarter_start": _nw_quarter_start,
    "quarter_end": _nw_quarter_end,
    "year_start": _nw_year_start,
    "year_end": _nw_year_end,
    "leap_year": _nw_leap_year,
    "days_in_month": _nw_days_in_month,
    "hour": lambda x: x.dt.hour(),
    "minute": lambda x: x.dt.minute(),
    "second": lambda x: x.dt.second(),
}
