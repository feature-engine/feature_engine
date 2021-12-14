import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.datetime import DatetimeFeatures
from feature_engine.datetime.datetime_constants import (
    FEATURES_DEFAULT,
    FEATURES_SUFFIXES,
    FEATURES_SUPPORTED,
)

vars_dt = ["datetime_range", "date_obj1", "date_obj2", "time_obj"]
vars_non_dt = ["Name", "Age"]
feat_names_default = [FEATURES_SUFFIXES[feat] for feat in FEATURES_DEFAULT]
dates_nan = pd.DataFrame({"dates_na": ["Feb-2010", np.nan, "Jun-1922", np.nan]})

# _false_input_params = [
#     (["not_supported"], ["year", 1874], "year", 14198),
#     (3.519, [1, -1.09, "var3"],  [True, False], {True}),
#     ("wrong_option", 1, [True, False], {True}),
# ]

_false_input_params = [
    (["not_supported"], 3.519, "wrong_option"),
    (["year", 1874], [1, -1.09, "var3"], 1),
    ("year", [3.5], [True, False]),
    (14198, [0.1, False], {True})
]


@pytest.mark.parametrize(
    "_features_to_extract, _variables, _other_params", _false_input_params
)
def test_raises_error_when_wrong_input_params(
        _features_to_extract, _variables, _other_params):
    with pytest.raises(ValueError):
        assert DatetimeFeatures(features_to_extract=_features_to_extract)
    with pytest.raises(ValueError):
        assert DatetimeFeatures(variables=_variables)
    with pytest.raises(ValueError):
        assert DatetimeFeatures(missing_values=_other_params)
    with pytest.raises(ValueError):
        assert DatetimeFeatures(drop_original=_other_params)
    with pytest.raises(ValueError):
        assert DatetimeFeatures(time_aware=_other_params)


def test_default_params():
    transformer = DatetimeFeatures()
    assert isinstance(transformer, DatetimeFeatures)
    assert transformer.variables is None
    assert transformer.features_to_extract is None
    assert transformer.drop_original
    assert transformer.time_aware is False
    assert transformer.dayfirst is False
    assert transformer.yearfirst is False
    assert transformer.missing_values == "raise"


_variables = [0, [0, 1, 9, 23], "var_str", ["var_str1", "var_str2"], [0, 1, "var3", 3]]


@pytest.mark.parametrize(
    "_variables", _variables
)
def test_variables_params(_variables):
    assert DatetimeFeatures(variables=_variables).variables == _variables


def test_features_to_extract_param():
    assert DatetimeFeatures(features_to_extract=None).features_to_extract is None
    assert DatetimeFeatures(features_to_extract=["year"]).features_to_extract == [
        "year"
    ]
    assert DatetimeFeatures(features_to_extract="all").features_to_extract == "all"


_not_a_df = [
    "not_a_df",
    [1, 2, 3, "some_data"],
    pd.Series([-2, 1.5, 8.94], name="not_a_df"),
]


@pytest.mark.parametrize(
    "_not_a_df", _not_a_df
)
def test_raises_error_when_fitting_not_a_df(_not_a_df):
    transformer = DatetimeFeatures()
    # trying to fit not a df
    with pytest.raises(TypeError):
        transformer.fit(_not_a_df)


def test_raises_error_when_variables_not_datetime(df_datetime):
    # asking for not datetime variable(s)
    with pytest.raises(TypeError):
        DatetimeFeatures(variables=["Age"]).fit(df_datetime)
    with pytest.raises(TypeError):
        DatetimeFeatures(variables=["Name", "Age", "date_obj1"]).fit(df_datetime)
    # passing a df that contains no datetime variables
    with pytest.raises(ValueError):
        DatetimeFeatures().fit(df_datetime[["Name", "Age"]])


def test_raises_error_when_df_has_nan():
    # dataset containing nans
    with pytest.raises(ValueError):
        DatetimeFeatures().fit(dates_nan)


def test_attributes_upon_fitting(df_datetime):
    assert DatetimeFeatures().fit(df_datetime).variables_ == vars_dt
    assert DatetimeFeatures(variables="date_obj1").fit(df_datetime).variables_ == [
        "date_obj1"
    ]
    assert DatetimeFeatures(variables=["date_obj1", "time_obj"]).fit(
        df_datetime
    ).variables_ == ["date_obj1", "time_obj"]
    assert DatetimeFeatures().fit(df_datetime).features_to_extract_ == FEATURES_DEFAULT
    assert (
            DatetimeFeatures(features_to_extract="all")
            .fit(df_datetime)
            .features_to_extract_
            == FEATURES_SUPPORTED
    )
    assert DatetimeFeatures(features_to_extract=["year", "quarter_end", "second"]).fit(
        df_datetime
    ).features_to_extract_ == ["year", "quarter_end", "second"]
    assert DatetimeFeatures().fit(df_datetime).n_features_in_ == df_datetime.shape[1]


def test_raises_error_when_transforming(df_datetime):
    # trying to transform before fitting
    with pytest.raises(NotFittedError):
        DatetimeFeatures().transform(df_datetime)
    transformer = DatetimeFeatures()
    transformer.fit(df_datetime)
    # trying to transform not a df
    with pytest.raises(TypeError):
        transformer.transform("not_a_df")
    with pytest.raises(TypeError):
        transformer.transform([1, 2, 3, "some_data"])
    with pytest.raises(TypeError):
        transformer.transform(pd.Series([-2, 1.5, 8.94], name="not_a_df"))
    # different number of columns than the df used to fit
    with pytest.raises(ValueError):
        transformer.transform(df_datetime[vars_dt])
    # dataset containing nans
    with pytest.raises(ValueError):
        DatetimeFeatures().fit_transform(dates_nan)


def test_extract_datetime_features_with_default_options(
        df_datetime, df_datetime_transformed
):
    transformer = DatetimeFeatures()
    X = transformer.fit_transform(df_datetime)
    pd.testing.assert_frame_equal(
        X,
        df_datetime_transformed[
            vars_non_dt + [var + feat for var in vars_dt for feat in feat_names_default]
            ],
    )


def test_extract_datetime_features_from_specified_variables(
        df_datetime, df_datetime_transformed
):
    # single datetime variable
    X = DatetimeFeatures(variables="date_obj1").fit_transform(df_datetime)
    pd.testing.assert_frame_equal(
        X,
        df_datetime_transformed[
            vars_non_dt
            + ["datetime_range", "date_obj2", "time_obj"]
            + ["date_obj1" + feat for feat in feat_names_default]
            ],
    )

    # multiple datetime variables
    X = DatetimeFeatures(variables=["datetime_range", "date_obj2"]).fit_transform(
        df_datetime
    )
    pd.testing.assert_frame_equal(
        X,
        df_datetime_transformed[
            vars_non_dt
            + ["date_obj1", "time_obj"]
            + [
                var + feat
                for var in ["datetime_range", "date_obj2"]
                for feat in feat_names_default
            ]
            ],
    )

    # multiple datetime variables in different order than they appear in the df
    X = DatetimeFeatures(variables=["date_obj2", "date_obj1"]).fit_transform(
        df_datetime
    )
    pd.testing.assert_frame_equal(
        X,
        df_datetime_transformed[
            vars_non_dt
            + ["datetime_range", "time_obj"]
            + [
                var + feat
                for var in ["date_obj2", "date_obj1"]
                for feat in feat_names_default
            ]
            ],
    )


def test_extract_all_datetime_features(df_datetime, df_datetime_transformed):
    X = DatetimeFeatures(features_to_extract="all").fit_transform(df_datetime)
    pd.testing.assert_frame_equal(X, df_datetime_transformed.drop(vars_dt, axis=1))


def test_extract_specified_datetime_features(df_datetime, df_datetime_transformed):
    X = DatetimeFeatures(
        features_to_extract=["semester", "week_of_the_year"]
    ).fit_transform(df_datetime)
    pd.testing.assert_frame_equal(
        X,
        df_datetime_transformed[
            vars_non_dt
            + [var + "_" + feat for var in vars_dt for feat in ["semester", "woty"]]
            ],
    )

    # different order than they appear in the glossary
    X = DatetimeFeatures(features_to_extract=["hour", "day_of_the_week"]).fit_transform(
        df_datetime
    )
    pd.testing.assert_frame_equal(
        X,
        df_datetime_transformed[
            vars_non_dt
            + [var + "_" + feat for var in vars_dt for feat in ["hour", "dotw"]]
            ],
    )


def test_extract_features_from_categorical_variable(
        df_datetime, df_datetime_transformed
):
    cat_date = pd.DataFrame(
        {"date_obj1": df_datetime["date_obj1"].astype("category")}
    )
    X = DatetimeFeatures(variables="date_obj1").fit_transform(cat_date)
    pd.testing.assert_frame_equal(
        X,
        df_datetime_transformed[["date_obj1" + feat for feat in feat_names_default]]
    )


def test_extract_features_from_different_timezones(
        df_datetime, df_datetime_transformed
):
    time_zones = [4, -1, 9, -7]
    tz_time = pd.DataFrame(
        {"time_obj": df_datetime["time_obj"].add(['+4', '-1', '+9', '-7'])}
    )
    X = DatetimeFeatures(
        variables="time_obj",
        features_to_extract=["hour"],
        time_aware=True) \
        .fit_transform(tz_time)
    pd.testing.assert_frame_equal(
        X,
        df_datetime_transformed[["time_obj_hour"]].apply(
            lambda x: x.subtract(time_zones)
        )
    )
    with pytest.raises(AttributeError):
        assert DatetimeFeatures(
            variables="time_obj",
            features_to_extract=["hour"],
            time_aware=False) \
            .fit_transform(tz_time)


def test_extract_features_without_dropping_original_variables(
        df_datetime, df_datetime_transformed
):
    X = DatetimeFeatures(
        variables=["datetime_range", "date_obj2"],
        features_to_extract=["week_of_the_year", "quarter"],
        drop_original=False,
    ).fit_transform(df_datetime)

    pd.testing.assert_frame_equal(
        X,
        pd.concat(
            [df_datetime_transformed[column] for column in vars_non_dt]
            + [df_datetime[var] for var in vars_dt]
            + [
                df_datetime_transformed[feat]
                for feat in [
                    var + "_" + feat
                    for var in ["datetime_range", "date_obj2"]
                    for feat in ["woty", "quarter"]
                ]
            ],
            axis=1,
        ),
    )


def test_extract_features_from_variables_containing_nans():
    X = DatetimeFeatures(
        features_to_extract=["year"], missing_values="ignore"
    ).fit_transform(dates_nan)
    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame({"dates_na_year": [2010, np.nan, 1922, np.nan]}),
    )


def test_extract_features_with_different_datetime_parsing_options(df_datetime):
    X = DatetimeFeatures(
        features_to_extract=["day_of_the_month"], dayfirst=True
    ).fit_transform(df_datetime[["date_obj2"]])
    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame({"date_obj2_dotm": [10, 31, 30, 17]}),
    )

    X = DatetimeFeatures(features_to_extract=["year"], yearfirst=True).fit_transform(
        df_datetime[["date_obj2"]]
    )
    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame({"date_obj2_year": [2010, 2009, 1995, 2004]}),
    )
