import pandas as pd
import pytest


@pytest.fixture
def df_datetime():
    data = {
        "Name": ["tom", "nick", "krish", "jack"],
        "Age": [20, 21, 19, 18],
        "datetime_range": pd.date_range("2020-02-24", periods=4, freq="D"),
        "date_obj1": ["01-Jan-2010", "24-Feb-1945", "14-Jun-2100", "17-May-1999"],
        "date_obj2": ["10/11/12", "12/31/09", "06/30/95", "03/17/04"],
        "time_obj": ["21:45:23", "09:15:33", "12:34:59", "03:27:02"],
    }

    df = pd.DataFrame(data)

    return df


@pytest.fixture
def df_datetime_transformed(df_datetime):
    today = pd.Timestamp.today()
    data = {
        "datetime_range_month": [2, 2, 2, 2],
        "datetime_range_quarter": [1, 1, 1, 1],
        "datetime_range_semester": [1, 1, 1, 1],
        "datetime_range_year": [2020, 2020, 2020, 2020],
        "datetime_range_week": [9, 9, 9, 9],
        "datetime_range_day_of_week": [0, 1, 2, 3],
        "datetime_range_day_of_month": [24, 25, 26, 27],
        "datetime_range_day_of_year": [55, 56, 57, 58],
        "datetime_range_weekend": [0, 0, 0, 0],
        "datetime_range_month_start": [0, 0, 0, 0],
        "datetime_range_month_end": [0, 0, 0, 0],
        "datetime_range_quarter_start": [0, 0, 0, 0],
        "datetime_range_quarter_end": [0, 0, 0, 0],
        "datetime_range_year_start": [0, 0, 0, 0],
        "datetime_range_year_end": [0, 0, 0, 0],
        "datetime_range_leap_year": [1, 1, 1, 1],
        "datetime_range_days_in_month": [29, 29, 29, 29],
        "datetime_range_hour": [0] * 4,
        "datetime_range_minute": [0] * 4,
        "datetime_range_second": [0] * 4,
        "date_obj1_month": [1, 2, 6, 5],
        "date_obj1_quarter": [1, 1, 2, 2],
        "date_obj1_semester": [1, 1, 1, 1],
        "date_obj1_year": [2010, 1945, 2100, 1999],
        "date_obj1_week": [53, 8, 24, 20],
        "date_obj1_day_of_week": [4, 5, 0, 0],
        "date_obj1_day_of_month": [1, 24, 14, 17],
        "date_obj1_day_of_year": [1, 55, 165, 137],
        "date_obj1_weekend": [0, 1, 0, 0],
        "date_obj1_month_start": [1, 0, 0, 0],
        "date_obj1_month_end": [0, 0, 0, 0],
        "date_obj1_quarter_start": [1, 0, 0, 0],
        "date_obj1_quarter_end": [0, 0, 0, 0],
        "date_obj1_year_start": [1, 0, 0, 0],
        "date_obj1_year_end": [0, 0, 0, 0],
        "date_obj1_leap_year": [0, 0, 0, 0],
        "date_obj1_days_in_month": [31, 28, 30, 31],
        "date_obj1_hour": [0] * 4,
        "date_obj1_minute": [0] * 4,
        "date_obj1_second": [0] * 4,
        "date_obj2_month": [10, 12, 6, 3],
        "date_obj2_quarter": [4, 4, 2, 1],
        "date_obj2_semester": [2, 2, 1, 1],
        "date_obj2_year": [2012, 2009, 1995, 2004],
        "date_obj2_week": [41, 53, 26, 12],
        "date_obj2_day_of_week": [3, 3, 4, 2],
        "date_obj2_day_of_month": [11, 31, 30, 17],
        "date_obj2_day_of_year": [285, 365, 181, 77],
        "date_obj2_weekend": [0, 0, 0, 0],
        "date_obj2_month_start": [0, 0, 0, 0],
        "date_obj2_month_end": [0, 1, 1, 0],
        "date_obj2_quarter_start": [0, 0, 0, 0],
        "date_obj2_quarter_end": [0, 1, 1, 0],
        "date_obj2_year_start": [0, 0, 0, 0],
        "date_obj2_year_end": [0, 1, 0, 0],
        "date_obj2_leap_year": [1, 0, 0, 1],
        "date_obj2_days_in_month": [31, 31, 30, 31],
        "date_obj2_hour": [0] * 4,
        "date_obj2_minute": [0] * 4,
        "date_obj2_second": [0] * 4,
        "time_obj_month": [today.month] * 4,
        "time_obj_quarter": [today.quarter] * 4,
        "time_obj_semester": [1 if today.month <= 6 else 2] * 4,
        "time_obj_year": [today.year] * 4,
        "time_obj_week": [today.week] * 4,
        "time_obj_day_of_week": [today.dayofweek] * 4,
        "time_obj_day_of_month": [today.day] * 4,
        "time_obj_day_of_year": [today.dayofyear] * 4,
        "time_obj_weekend": [1 if today.dayofweek > 4 else 0] * 4,
        "time_obj_month_start": [int(today.is_month_start)] * 4,
        "time_obj_month_end": [int(today.is_month_end)] * 4,
        "time_obj_quarter_start": [int(today.is_quarter_start)] * 4,
        "time_obj_quarter_end": [int(today.is_quarter_end)] * 4,
        "time_obj_year_start": [int(today.is_year_start)] * 4,
        "time_obj_year_end": [int(today.is_year_end)] * 4,
        "time_obj_leap_year": [int(today.is_leap_year)] * 4,
        "time_obj_days_in_month": [today.days_in_month] * 4,
        "time_obj_hour": [21, 9, 12, 3],
        "time_obj_minute": [45, 15, 34, 27],
        "time_obj_second": [23, 33, 59, 2],
    }
    df = df_datetime.join(pd.DataFrame(data))
    return df
