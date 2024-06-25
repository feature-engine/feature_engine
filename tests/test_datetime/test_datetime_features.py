import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from feature_engine.datetime import DatetimeFeatures
from feature_engine.datetime._datetime_constants import (
    FEATURES_DEFAULT,
    FEATURES_SUFFIXES,
    FEATURES_SUPPORTED,
)

vars_dt = ["datetime_range", "date_obj1", "date_obj2", "time_obj"]
vars_non_dt = ["Name", "Age"]
feat_names_default = [FEATURES_SUFFIXES[feat] for feat in FEATURES_DEFAULT]
dates_nan = pd.DataFrame({"dates_na": ["Feb-2010", np.nan, "Jun-1922", np.nan]})
dates_idx_nan = pd.DataFrame(
    [1, 2, 3, 4], index=["Feb-2010", np.nan, "Jun-1922", np.nan]
)
dates_idx_dt = pd.DataFrame(
    [4, 3, 2, 1],
    index=pd.date_range("2003-02-27", periods=4, freq="D"),
)


_false_input_params = [
    (["not_supported"], 3.519, "wrong_option"),
    (["year", 1874], [1, -1.09, "var3"], 1),
    ("year", [3.5], [True, False]),
    (14198, [0.1, False], {True}),
]


@pytest.mark.parametrize(
    "_features_to_extract, _variables, _other_params", _false_input_params
)
def test_raises_error_when_wrong_input_params(
    _features_to_extract, _variables, _other_params
):
    with pytest.raises(ValueError):
        assert DatetimeFeatures(features_to_extract=_features_to_extract)
    with pytest.raises(ValueError):
        assert DatetimeFeatures(variables=_variables)
    with pytest.raises(ValueError):
        assert DatetimeFeatures(missing_values=_other_params)
    with pytest.raises(ValueError):
        assert DatetimeFeatures(drop_original=_other_params)
    with pytest.raises(ValueError):
        assert DatetimeFeatures(utc=_other_params)


def test_default_params():
    transformer = DatetimeFeatures()
    assert isinstance(transformer, DatetimeFeatures)
    assert transformer.variables is None
    assert transformer.features_to_extract is None
    assert transformer.drop_original
    assert transformer.utc is None
    assert transformer.dayfirst is False
    assert transformer.yearfirst is False
    assert transformer.missing_values == "raise"


_variables = [0, [0, 1, 9, 23], "var_str", ["var_str1", "var_str2"], [0, 1, "var3", 3]]


@pytest.mark.parametrize("_variables", _variables)
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


@pytest.mark.parametrize("_not_a_df", _not_a_df)
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
    with pytest.raises(TypeError):
        DatetimeFeatures(variables="index").fit(df_datetime)
    # passing a df that contains no datetime variables
    with pytest.raises(ValueError):
        DatetimeFeatures().fit(df_datetime[["Name", "Age"]])


def test_raises_error_when_df_has_nan():
    # dataset containing nans
    with pytest.raises(ValueError):
        DatetimeFeatures().fit(dates_nan)
    with pytest.raises(ValueError):
        DatetimeFeatures(variables="index").fit(dates_idx_nan)


def test_attributes_upon_fitting(df_datetime):
    transformer = DatetimeFeatures()
    transformer.fit(df_datetime)

    assert transformer.variables_ == vars_dt
    assert transformer.features_to_extract_ == FEATURES_DEFAULT
    assert transformer.n_features_in_ == df_datetime.shape[1]

    transformer = DatetimeFeatures(variables="date_obj1", features_to_extract="all")
    transformer.fit(df_datetime)

    assert transformer.variables_ == ["date_obj1"]
    assert transformer.features_to_extract_ == FEATURES_SUPPORTED

    transformer = DatetimeFeatures(
        variables=["date_obj1", "time_obj"],
        features_to_extract=["year", "quarter_end", "second"],
    )
    transformer.fit(df_datetime)

    assert transformer.variables_ == ["date_obj1", "time_obj"]
    assert transformer.features_to_extract_ == ["year", "quarter_end", "second"]


@pytest.mark.parametrize("_not_a_df", _not_a_df)
def test_raises_error_when_transforming_not_a_df(_not_a_df, df_datetime):
    transformer = DatetimeFeatures()
    transformer.fit(df_datetime)
    # trying to transform not a df
    with pytest.raises(TypeError):
        transformer.transform(_not_a_df)


def test_raises_error_when_transform_df_with_different_n_variables(df_datetime):
    transformer = DatetimeFeatures()
    transformer.fit(df_datetime)
    # different number of columns than the df used to fit
    with pytest.raises(ValueError):
        transformer.transform(df_datetime[vars_dt])


def test_raises_error_when_nan_in_transform_df(df_datetime):
    transformer = DatetimeFeatures()
    transformer.fit(df_datetime)
    # dataset containing nans
    with pytest.raises(ValueError):
        transformer.transform(dates_nan)
    transformer = DatetimeFeatures(variables="index")
    transformer.fit(dates_idx_dt)
    with pytest.raises(ValueError):
        transformer.transform(dates_idx_nan)


def test_raises_non_fitted_error(df_datetime):
    # trying to transform before fitting
    with pytest.raises(NotFittedError):
        DatetimeFeatures().transform(df_datetime)


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
        check_dtype=False,
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
        check_dtype=False,
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
        check_dtype=False,
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
        check_dtype=False,
    )

    # datetime variable is index
    X = DatetimeFeatures(
        variables="index", features_to_extract=["month", "day_of_month"]
    ).fit_transform(dates_idx_dt)
    pd.testing.assert_frame_equal(
        X,
        pd.concat(
            [
                dates_idx_dt,
                pd.DataFrame(
                    [[2, 27], [2, 28], [3, 1], [3, 2]],
                    index=dates_idx_dt.index,
                    columns=["month", "day_of_month"],
                ),
            ],
            axis=1,
        ),
        check_dtype=False,
    )


def test_extract_all_datetime_features(df_datetime, df_datetime_transformed):
    X = DatetimeFeatures(features_to_extract="all").fit_transform(df_datetime)
    pd.testing.assert_frame_equal(
        X, df_datetime_transformed.drop(vars_dt, axis=1), check_dtype=False
    )


def test_extract_specified_datetime_features(df_datetime, df_datetime_transformed):
    X = DatetimeFeatures(features_to_extract=["semester", "week"]).fit_transform(
        df_datetime
    )
    pd.testing.assert_frame_equal(
        X,
        df_datetime_transformed[
            vars_non_dt
            + [var + "_" + feat for var in vars_dt for feat in ["semester", "week"]]
        ],
        check_dtype=False,
    )

    # different order than they appear in the glossary
    X = DatetimeFeatures(features_to_extract=["hour", "day_of_week"]).fit_transform(
        df_datetime
    )
    pd.testing.assert_frame_equal(
        X,
        df_datetime_transformed[
            vars_non_dt
            + [var + "_" + feat for var in vars_dt for feat in ["hour", "day_of_week"]]
        ],
        check_dtype=False,
    )


def test_extract_features_from_categorical_variable(
    df_datetime, df_datetime_transformed
):
    cat_date = pd.DataFrame({"date_obj1": df_datetime["date_obj1"].astype("category")})
    X = DatetimeFeatures(variables="date_obj1").fit_transform(cat_date)
    pd.testing.assert_frame_equal(
        X,
        df_datetime_transformed[["date_obj1" + feat for feat in feat_names_default]],
        check_dtype=False,
    )


def test_extract_features_from_different_timezones():
    df = pd.DataFrame()
    df["time"] = pd.concat(
        [
            pd.Series(
                pd.date_range(
                    start="2014-08-01 09:00", freq="h", periods=3, tz="Europe/Berlin"
                )
            ),
            pd.Series(
                pd.date_range(
                    start="2014-08-01 09:00", freq="h", periods=3, tz="US/Central"
                )
            ),
        ],
        axis=0,
    )
    df.reset_index(inplace=True, drop=True)

    transformer = DatetimeFeatures(
        variables="time", features_to_extract=["hour"], utc=True
    )
    X = transformer.fit_transform(df)

    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame({"time_hour": [7, 8, 9, 14, 15, 16]}),
        check_dtype=False,
    )
    exp_err_msg = (
        "Tz-aware datetime.datetime cannot be converted to datetime64 "
        "unless utc=True, at position 3"
    )
    with pytest.raises(ValueError) as errinfo:
        assert DatetimeFeatures(
            variables="time", features_to_extract=["hour"], utc=False
        ).fit_transform(df)
    assert str(errinfo.value) == exp_err_msg


def test_extract_features_from_different_timezones_when_string(
    df_datetime, df_datetime_transformed
):
    time_zones = [4, -1, 9, -7]
    tz_df = pd.DataFrame(
        {"time_obj": df_datetime["time_obj"].add(["+4", "-1", "+9", "-7"])}
    )
    transformer = DatetimeFeatures(
        variables="time_obj",
        features_to_extract=["hour"],
        utc=True,
        format="mixed",
    )
    X = transformer.fit_transform(tz_df)

    pd.testing.assert_frame_equal(
        X,
        df_datetime_transformed[["time_obj_hour"]].apply(
            lambda x: x.subtract(time_zones)
        ),
        check_dtype=False,
    )


def test_extract_features_from_localized_tz_variables():
    tz_df = pd.DataFrame(
        {
            "date_var": [
                "2018-10-28 01:30:00",
                "2018-10-28 02:00:00",
                "2018-10-28 02:30:00",
                "2018-10-28 02:00:00",
                "2018-10-28 02:30:00",
                "2018-10-28 03:00:00",
                "2018-10-28 03:30:00",
            ]
        }
    )

    tz_df["date_var"] = pd.to_datetime(tz_df["date_var"]).dt.tz_localize(
        tz="US/Eastern"
    )

    # when utc is None
    transformer = DatetimeFeatures(features_to_extract=["hour"]).fit(tz_df)

    # init params
    assert transformer.variables is None
    assert transformer.utc is None
    assert transformer.features_to_extract == ["hour"]
    # fit attr
    assert transformer.variables_ == ["date_var"]
    assert transformer.features_to_extract_ == ["hour"]
    assert transformer.n_features_in_ == 1
    # transform
    X = transformer.transform(tz_df)
    df_expected = pd.DataFrame({"date_var_hour": [1, 2, 2, 2, 2, 3, 3]})
    pd.testing.assert_frame_equal(X, df_expected, check_dtype=False)

    # when utc is True
    transformer = DatetimeFeatures(features_to_extract=["hour"], utc=True).fit(tz_df)

    # init params
    assert transformer.variables is None
    assert transformer.utc is True
    assert transformer.features_to_extract == ["hour"]
    # fit attr
    assert transformer.variables_ == ["date_var"]
    assert transformer.features_to_extract_ == ["hour"]
    assert transformer.n_features_in_ == 1
    # transform
    X = transformer.transform(tz_df)
    df_expected = pd.DataFrame({"date_var_hour": [5, 6, 6, 6, 6, 7, 7]})
    pd.testing.assert_frame_equal(X, df_expected, check_dtype=False)


def test_extract_features_without_dropping_original_variables(
    df_datetime, df_datetime_transformed
):
    X = DatetimeFeatures(
        variables=["datetime_range", "date_obj2"],
        features_to_extract=["week", "quarter"],
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
                    for feat in ["week", "quarter"]
                ]
            ],
            axis=1,
        ),
        check_dtype=False,
    )


def test_extract_features_from_variables_containing_nans():
    X = DatetimeFeatures(
        features_to_extract=["year"], missing_values="ignore"
    ).fit_transform(dates_nan)
    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame({"dates_na_year": [2010, np.nan, 1922, np.nan]}),
    )
    # dt variable is index
    X = DatetimeFeatures(
        variables="index", features_to_extract=["month"], missing_values="ignore"
    ).fit_transform(dates_idx_nan)
    pd.testing.assert_frame_equal(
        X,
        pd.concat(
            [
                dates_idx_nan,
                pd.DataFrame(
                    {"month": [2, np.nan, 6, np.nan]}, index=dates_idx_nan.index
                ),
            ],
            axis=1,
        ),
    )


def test_extract_features_with_different_datetime_parsing_options(df_datetime):
    X = DatetimeFeatures(
        features_to_extract=["day_of_month"], dayfirst=True
    ).fit_transform(df_datetime[["date_obj2"]])
    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame({"date_obj2_day_of_month": [10, 31, 30, 17]}),
        check_dtype=False,
    )

    X = DatetimeFeatures(features_to_extract=["year"], yearfirst=True).fit_transform(
        df_datetime[["date_obj2"]]
    )
    pd.testing.assert_frame_equal(
        X,
        pd.DataFrame({"date_obj2_year": [2010, 2009, 1995, 2004]}),
        check_dtype=False,
    )


def test_get_feature_names_out(df_datetime, df_datetime_transformed):
    # default features from all variables
    transformer = DatetimeFeatures()
    X = transformer.fit_transform(df_datetime)
    assert list(X.columns) == transformer.get_feature_names_out()
    assert list(X.columns) == transformer.get_feature_names_out(df_datetime.columns)

    with pytest.raises(ValueError):
        transformer.get_feature_names_out(input_features=vars_dt)

    with pytest.raises(ValueError):
        transformer.get_feature_names_out(input_features=["date_obj1"])
    # default features from 1 variable
    transformer = DatetimeFeatures(variables="date_obj1")
    X = transformer.fit_transform(df_datetime)
    assert list(X.columns) == transformer.get_feature_names_out()
    assert list(X.columns) == transformer.get_feature_names_out(df_datetime.columns)

    # all features
    transformer = DatetimeFeatures(features_to_extract="all")
    X = transformer.fit_transform(df_datetime)
    assert list(X.columns) == transformer.get_feature_names_out()

    # specified features
    transformer = DatetimeFeatures(features_to_extract=["semester", "week"])
    X = transformer.fit_transform(df_datetime)
    assert list(X.columns) == transformer.get_feature_names_out()

    # features were extracted from index
    transformer = DatetimeFeatures(
        variables="index", features_to_extract=["semester", "week"]
    )
    X = transformer.fit_transform(dates_idx_dt)
    assert list(X.columns) == transformer.get_feature_names_out()

    # user passes something else than index as input_features
    with pytest.raises(ValueError):
        transformer.get_feature_names_out(input_features="not_index")
    with pytest.raises(ValueError):
        transformer.get_feature_names_out(input_features=["still", "not", "index"])

    # when drop original is False
    transformer = DatetimeFeatures(drop_original=False)
    X = transformer.fit_transform(df_datetime)
    assert list(X.columns) == transformer.get_feature_names_out()
    with pytest.raises(ValueError):
        # assert error when user passes a string instead of list
        transformer.get_feature_names_out(input_features="date_obj1")

    with pytest.raises(ValueError):
        # assert error when uses passes features that were not lagged
        transformer.get_feature_names_out(input_features=["color"])


def test_get_feature_names_out_from_pipeline(df_datetime, df_datetime_transformed):
    transformer = Pipeline([("transformer", DatetimeFeatures())])
    X = transformer.fit_transform(df_datetime)
    assert list(X.columns) == transformer.get_feature_names_out()
    assert list(X.columns) == transformer.get_feature_names_out(df_datetime.columns)
