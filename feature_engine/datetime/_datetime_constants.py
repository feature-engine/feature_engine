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

# The functions below take a numpy datetime64 array and return the feature as an
# int64 array. Casting to a coarser unit ("M8[M]", "M8[D]") floors the dates.


def _month(x: np.ndarray) -> np.ndarray:
    return x.astype("M8[M]").astype(np.int64) % 12 + 1


def _quarter(x: np.ndarray) -> np.ndarray:
    return (_month(x) - 1) // 3 + 1


def _semester(x: np.ndarray) -> np.ndarray:
    return np.where(_month(x) <= 6, 1, 2)


def _year(x: np.ndarray) -> np.ndarray:
    return x.astype("M8[Y]").astype(np.int64) + 1970


def _day_of_week(x: np.ndarray) -> np.ndarray:
    # 0 is Monday; 1970-01-01, day 0 in numpy, was a Thursday.
    return (x.astype("M8[D]").astype(np.int64) + 3) % 7


def _week(x: np.ndarray) -> np.ndarray:
    # An ISO week belongs to the year of its Thursday.
    thursday = x.astype("M8[D]") + (3 - _day_of_week(x)).astype("m8[D]")
    first_day = thursday.astype("M8[Y]").astype("M8[D]")
    return (thursday - first_day).astype(np.int64) // 7 + 1


def _day_of_month(x: np.ndarray) -> np.ndarray:
    first_day: np.ndarray = x.astype("M8[M]").astype("M8[D]")
    return (x.astype("M8[D]") - first_day).astype(np.int64) + 1


def _day_of_year(x: np.ndarray) -> np.ndarray:
    first_day: np.ndarray = x.astype("M8[Y]").astype("M8[D]")
    return (x.astype("M8[D]") - first_day).astype(np.int64) + 1


def _days_in_month(x: np.ndarray) -> np.ndarray:
    month: np.ndarray = x.astype("M8[M]")
    next_month = month + np.timedelta64(1, "M")
    return (next_month.astype("M8[D]") - month.astype("M8[D]")).astype(np.int64)


def _is_month_start(x: np.ndarray) -> np.ndarray:
    return _day_of_month(x) == 1


def _is_month_end(x: np.ndarray) -> np.ndarray:
    return _day_of_month(x) == _days_in_month(x)


def _leap_year(x: np.ndarray) -> np.ndarray:
    year = _year(x)
    is_leap = ((year % 4 == 0) & (year % 100 != 0)) | (year % 400 == 0)
    return is_leap.astype(np.int64)


FEATURES_FUNCTIONS = {
    "month": _month,
    "quarter": _quarter,
    "semester": _semester,
    "year": _year,
    "week": _week,
    "day_of_week": _day_of_week,
    "day_of_month": _day_of_month,
    "day_of_year": _day_of_year,
    "weekend": lambda x: (_day_of_week(x) >= 5).astype(np.int64),
    "month_start": lambda x: _is_month_start(x).astype(np.int64),
    "month_end": lambda x: _is_month_end(x).astype(np.int64),
    "quarter_start": lambda x: (_is_month_start(x) & (_month(x) % 3 == 1)).astype(
        np.int64
    ),
    "quarter_end": lambda x: (_is_month_end(x) & (_month(x) % 3 == 0)).astype(
        np.int64
    ),
    "year_start": lambda x: (_is_month_start(x) & (_month(x) == 1)).astype(np.int64),
    "year_end": lambda x: (_is_month_end(x) & (_month(x) == 12)).astype(np.int64),
    "leap_year": _leap_year,
    "days_in_month": _days_in_month,
    "hour": lambda x: (x - x.astype("M8[D]")).astype("m8[h]").astype(np.int64),
    "minute": lambda x: (x - x.astype("M8[h]")).astype("m8[m]").astype(np.int64),
    "second": lambda x: (x - x.astype("M8[m]")).astype("m8[s]").astype(np.int64),
}
