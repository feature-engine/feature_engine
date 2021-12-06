import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.datetime import ExtractDatetimeFeatures
from feature_engine.datetime.datetime_constants import (
    FEATURES_DEFAULT,
    FEATURES_SUFFIXES,
)


def test_extract_date_features(df_datetime):
    vars_dt = ["datetime_range", "date_obj1", "date_obj2", "time_obj"]
    vars_non_dt = ["Name", "Age"]
    vars_mix = ["Name", "Age", "date_obj1"]
    feat_names_default = [FEATURES_SUFFIXES[feat] for feat in FEATURES_DEFAULT]
    today = pd.Timestamp.today()
    df_transformed_full = df_datetime.join(
        pd.DataFrame(
            {
                "datetime_range_month": [2, 2, 2, 2],
                "datetime_range_quarter": [1, 1, 1, 1],
                "datetime_range_semester": [1, 1, 1, 1],
                "datetime_range_year": [2020, 2020, 2020, 2020],
                "datetime_range_wotm": [4, 4, 4, 4],
                "datetime_range_woty": [9, 9, 9, 9],
                "datetime_range_dotw": [0, 1, 2, 3],
                "datetime_range_dotm": [24, 25, 26, 27],
                "datetime_range_doty": [55, 56, 57, 58],
                "datetime_range_weekend": [False, False, False, False],
                "datetime_range_month_start": [False, False, False, False],
                "datetime_range_month_end": [False, False, False, False],
                "datetime_range_quarter_start": [False, False, False, False],
                "datetime_range_quarter_end": [False, False, False, False],
                "datetime_range_year_start": [False, False, False, False],
                "datetime_range_year_end": [False, False, False, False],
                "datetime_range_leap_year": [True, True, True, True],
                "datetime_range_days_in_month": [29, 29, 29, 29],
                "datetime_range_hour": [0] * 4,
                "datetime_range_minute": [0] * 4,
                "datetime_range_second": [0] * 4,
                "date_obj1_month": [1, 2, 6, 5],
                "date_obj1_quarter": [1, 1, 2, 2],
                "date_obj1_semester": [1, 1, 1, 1],
                "date_obj1_year": [2010, 1945, 2100, 1999],
                "date_obj1_wotm": [1, 4, 2, 3],
                "date_obj1_woty": [53, 8, 24, 20],
                "date_obj1_dotw": [4, 5, 0, 0],
                "date_obj1_dotm": [1, 24, 14, 17],
                "date_obj1_doty": [1, 55, 165, 137],
                "date_obj1_weekend": [False, True, False, False],
                "date_obj1_month_start": [True, False, False, False],
                "date_obj1_month_end": [False, False, False, False],
                "date_obj1_quarter_start": [True, False, False, False],
                "date_obj1_quarter_end": [False, False, False, False],
                "date_obj1_year_start": [True, False, False, False],
                "date_obj1_year_end": [False, False, False, False],
                "date_obj1_leap_year": [False, False, False, False],
                "date_obj1_days_in_month": [31, 28, 30, 31],
                "date_obj1_hour": [0] * 4,
                "date_obj1_minute": [0] * 4,
                "date_obj1_second": [0] * 4,
                "date_obj2_month": [10, 12, 6, 3],
                "date_obj2_quarter": [4, 4, 2, 1],
                "date_obj2_semester": [2, 2, 1, 1],
                "date_obj2_year": [2012, 2009, 1995, 2004],
                "date_obj2_wotm": [2, 5, 5, 3],
                "date_obj2_woty": [41, 53, 26, 12],
                "date_obj2_dotw": [3, 3, 4, 2],
                "date_obj2_dotm": [11, 31, 30, 17],
                "date_obj2_doty": [285, 365, 181, 77],
                "date_obj2_weekend": [False, False, False, False],
                "date_obj2_month_start": [False, False, False, False],
                "date_obj2_month_end": [False, True, True, False],
                "date_obj2_quarter_start": [False, False, False, False],
                "date_obj2_quarter_end": [False, True, True, False],
                "date_obj2_year_start": [False, False, False, False],
                "date_obj2_year_end": [False, True, False, False],
                "date_obj2_leap_year": [True, False, False, True],
                "date_obj2_days_in_month": [31, 31, 30, 31],
                "date_obj2_hour": [0] * 4,
                "date_obj2_minute": [0] * 4,
                "date_obj2_second": [0] * 4,
                "time_obj_month": [today.month] * 4,
                "time_obj_quarter": [today.quarter] * 4,
                "time_obj_semester": [1 if today.month <= 6 else 2] * 4,
                "time_obj_year": [today.year] * 4,
                "time_obj_wotm": [(today.day - 1) // 7 + 1] * 4,
                "time_obj_woty": [today.week] * 4,
                "time_obj_dotw": [today.dayofweek] * 4,
                "time_obj_dotm": [today.day] * 4,
                "time_obj_doty": [today.dayofyear] * 4,
                "time_obj_weekend": [True if today.dayofweek > 4 else False] * 4,
                "time_obj_month_start": [today.is_month_start] * 4,
                "time_obj_month_end": [today.is_month_end] * 4,
                "time_obj_quarter_start": [today.is_quarter_start] * 4,
                "time_obj_quarter_end": [today.is_quarter_end] * 4,
                "time_obj_year_start": [today.is_year_start] * 4,
                "time_obj_year_end": [today.is_year_end] * 4,
                "time_obj_leap_year": [today.is_leap_year] * 4,
                "time_obj_days_in_month": [today.days_in_month] * 4,
                "time_obj_hour": [21, 9, 12, 3],
                "time_obj_minute": [45, 15, 34, 27],
                "time_obj_second": [23, 33, 59, 2],
            }
        )
    )
    non_constant_features = [
        "datetime_range_dotw",
        "datetime_range_dotm",
        "datetime_range_doty",
        "date_obj1_month",
        "date_obj1_quarter",
        "date_obj1_year",
        "date_obj1_wotm",
        "date_obj1_woty",
        "date_obj1_dotw",
        "date_obj1_dotm",
        "date_obj1_doty",
        "date_obj1_weekend",
        "date_obj1_month_start",
        "date_obj1_quarter_start",
        "date_obj1_days_in_month",
        "date_obj2_month",
        "date_obj2_quarter",
        "date_obj2_semester",
        "date_obj2_year",
        "date_obj2_wotm",
        "date_obj2_woty",
        "date_obj2_dotw",
        "date_obj2_dotm",
        "date_obj2_doty",
        "date_obj2_month_end",
        "date_obj2_quarter_end",
        "date_obj2_year_end",
        "date_obj2_leap_year",
        "date_obj2_days_in_month",
        "time_obj_hour",
        "time_obj_minute",
        "time_obj_second",
    ]
    dates_nan = pd.DataFrame({"dates_na": ["Feb-2010", np.nan, "Jun-1922", np.nan]})

    # check exceptions upon class instantiation
    with pytest.raises(ValueError):
        assert ExtractDatetimeFeatures(features_to_extract=["not_supported"])
    with pytest.raises(ValueError):
        assert ExtractDatetimeFeatures(features_to_extract=["year", 1874])
    with pytest.raises(ValueError):
        assert ExtractDatetimeFeatures(variables=3.519)
    with pytest.raises(ValueError):
        assert ExtractDatetimeFeatures(missing_values="wrong_option")
    with pytest.raises(TypeError):
        assert ExtractDatetimeFeatures(features_to_extract="year")
    with pytest.raises(TypeError):
        assert ExtractDatetimeFeatures(features_to_extract=14198)

    # check transformer attributes
    transformer = ExtractDatetimeFeatures()
    assert isinstance(transformer, ExtractDatetimeFeatures)
    assert transformer.variables is None
    assert ExtractDatetimeFeatures(variables="Age").variables == "Age"
    assert ExtractDatetimeFeatures(variables=["Age", "datetime_range"]).variables == [
        "Age",
        "datetime_range",
    ]
    transformer.fit(df_datetime)
    assert transformer.features_to_extract_ == FEATURES_DEFAULT
    transformer = ExtractDatetimeFeatures(features_to_extract=["year"])
    transformer.fit(df_datetime)
    assert transformer.features_to_extract_ == ["year"]

    # check exceptions upon calling fit method
    transformer = ExtractDatetimeFeatures()
    with pytest.raises(TypeError):
        transformer.fit("not_a_df")
    with pytest.raises(TypeError):
        ExtractDatetimeFeatures(variables=["Age"]).fit(df_datetime)
    with pytest.raises(TypeError):
        ExtractDatetimeFeatures(variables=vars_mix).fit(df_datetime)
    with pytest.raises(ValueError):
        transformer.fit(df_datetime[vars_non_dt])
    with pytest.raises(ValueError):
        transformer.fit(dates_nan)

    # check exceptions upon calling transform method
    transformer.fit(df_datetime)
    with pytest.raises(ValueError):
        transformer.transform(df_datetime[vars_dt])
    df_na = df_datetime.copy()
    df_na.loc[0, "date_obj1"] = np.nan
    with pytest.raises(ValueError):
        transformer.transform(df_na)
    with pytest.raises(NotFittedError):
        ExtractDatetimeFeatures().transform(df_datetime)

    # check default initialized transformer
    transformer = ExtractDatetimeFeatures()
    X = transformer.fit_transform(df_datetime)
    assert transformer.variables_ == [
        "datetime_range",
        "date_obj1",
        "date_obj2",
        "time_obj",
    ]
    assert transformer.n_features_in_ == 6
    pd.testing.assert_frame_equal(
        X,
        df_transformed_full[
            vars_non_dt
            + [
                var + feat
                for var in vars_dt
                for feat in feat_names_default
                if var + feat in non_constant_features
            ]
        ],
    )

    # check transformer with specified variables to process
    transformer = ExtractDatetimeFeatures(variables="date_obj1")
    assert isinstance(transformer, ExtractDatetimeFeatures)
    assert transformer.variables == "date_obj1"
    assert transformer.features_to_extract is None

    X = transformer.fit_transform(df_datetime)
    assert transformer.variables_ == ["date_obj1"]
    pd.testing.assert_frame_equal(
        X,
        df_transformed_full[
            vars_non_dt
            + ["datetime_range", "date_obj2", "time_obj"]
            + [
                "date_obj1" + feat
                for feat in feat_names_default
                if "date_obj1" + feat in non_constant_features
            ]
        ],
    )
    X = ExtractDatetimeFeatures(
        variables=["datetime_range", "date_obj2"]
    ).fit_transform(df_datetime)
    pd.testing.assert_frame_equal(
        X,
        df_transformed_full[
            vars_non_dt
            + ["date_obj1", "time_obj"]
            + [
                var + feat
                for var in ["datetime_range", "date_obj2"]
                for feat in feat_names_default
                if var + feat in non_constant_features
            ]
        ],
    )
    X = ExtractDatetimeFeatures(variables=["date_obj2", "date_obj1"]).fit_transform(
        df_datetime
    )
    pd.testing.assert_frame_equal(
        X,
        df_transformed_full[
            vars_non_dt
            + ["datetime_range", "time_obj"]
            + [
                var + feat
                for var in ["date_obj2", "date_obj1"]
                for feat in feat_names_default
                if var + feat in non_constant_features
            ]
        ],
    )

    # check transformer with specified date features to extract
    transformer = ExtractDatetimeFeatures(
        features_to_extract=["semester", "week_of_the_year"]
    )
    assert isinstance(transformer, ExtractDatetimeFeatures)
    assert transformer.variables is None
    assert transformer.features_to_extract == ["semester", "week_of_the_year"]

    X = transformer.fit_transform(df_datetime)
    assert transformer.variables_ == vars_dt
    pd.testing.assert_frame_equal(
        X,
        df_transformed_full[
            vars_non_dt
            + [
                var + "_" + feat
                for var in vars_dt
                for feat in ["semester", "woty"]
                if var + "_" + feat in non_constant_features
            ]
        ],
    )

    # check transformer with option to drop datetime features turned off
    X = ExtractDatetimeFeatures(
        variables=["datetime_range", "date_obj2"],
        features_to_extract=["week_of_the_year", "quarter"],
        drop_datetime=False,
    ).fit_transform(df_datetime)

    pd.testing.assert_frame_equal(
        X,
        pd.concat(
            [df_transformed_full[column] for column in vars_non_dt]
            + [
                pd.to_datetime(df_datetime["datetime_range"]),
                df_datetime["date_obj1"],
                pd.to_datetime(df_datetime["date_obj2"]),
                df_datetime["time_obj"],
            ]
            + [
                df_transformed_full[feat]
                for feat in [
                    var + "_" + feat
                    for var in ["datetime_range", "date_obj2"]
                    for feat in ["woty", "quarter"]
                    if var + "_" + feat in non_constant_features
                ]
            ],
            axis=1,
        ),
    )

    # check transformer with allowed nan option
    transformer = ExtractDatetimeFeatures(
        features_to_extract=["year"], missing_values="ignore"
    )
    pd.testing.assert_frame_equal(
        transformer.fit_transform(dates_nan),
        pd.DataFrame({"dates_na_year": [2010, np.nan, 1922, np.nan]}),
    )

    # check transformer with different pd.to_datetime options
    transformer = ExtractDatetimeFeatures(
        features_to_extract=["day_of_the_month"], dayfirst=True
    )
    pd.testing.assert_frame_equal(
        transformer.fit_transform(df_datetime[["date_obj2"]]),
        pd.DataFrame({"date_obj2_dotm": [10, 31, 30, 17]}),
    )

    transformer = ExtractDatetimeFeatures(features_to_extract=["year"], yearfirst=True)
    pd.testing.assert_frame_equal(
        transformer.fit_transform(df_datetime[["date_obj2"]]),
        pd.DataFrame({"date_obj2_year": [2010, 2009, 1995, 2004]}),
    )

    # check transformer with drop_constant turned off
    transformer = ExtractDatetimeFeatures(
        features_to_extract=["day_of_the_week", "quarter_start", "hour"],
        drop_constant=False,
    )
    pd.testing.assert_frame_equal(
        transformer.fit_transform(df_datetime[["time_obj"]]),
        df_transformed_full[
            ["time_obj_dotw", "time_obj_quarter_start", "time_obj_hour"]
        ],
    )
