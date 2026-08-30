import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from feature_engine.datetime import DatetimeFeatures
from feature_engine.datetime._datetime_constants import (
    FEATURES_DEFAULT,
    FEATURES_FUNCTIONS,
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

# ISO-8601 strings parse identically on pandas and polars/narwhals (unlike the
# dateutil-style formats in df_datetime above, which are pandas-only), so these
# back the cross-backend tests. Covers a leap day and a year/quarter/month
# boundary, so "all" features exercise every derived (non-1:1) narwhals feature.
CROSS_BACKEND_DATES = [
    "2020-01-01 00:00:00",
    "2020-02-29 12:30:45",
    "2020-12-31 23:59:59",
    "2021-07-15 06:07:08",
]
CROSS_BACKEND_DATA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "Age": [20, 21, 19, 18],
    "date": CROSS_BACKEND_DATES,
}
feat_names_default_cb = [f"date{FEATURES_SUFFIXES[feat]}" for feat in FEATURES_DEFAULT]


def _expected_cross_backend_features(feats):
    """Reference feature values computed with pandas' native FEATURES_FUNCTIONS,
    the ground truth both the pandas and the narwhals extraction paths must match."""
    dt = pd.Series(pd.to_datetime(CROSS_BACKEND_DATES))
    return {
        f"date{FEATURES_SUFFIXES[feat]}": list(FEATURES_FUNCTIONS[feat](dt))
        for feat in feats
    }


def _to_py_values(column):
    # normalise pandas/numpy and polars scalar containers to plain Python ints
    # so the two backends' outputs compare equal regardless of dtype width.
    return [int(v) for v in column]


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
    with pytest.raises(TypeError):
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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_extract_datetime_features_with_default_options(make_df):
    X = make_df(CROSS_BACKEND_DATA)
    Xt = DatetimeFeatures().fit_transform(X)

    result = nw.from_native(Xt, eager_only=True)
    assert result.columns == vars_non_dt + feat_names_default_cb
    for col, expected in _expected_cross_backend_features(FEATURES_DEFAULT).items():
        assert _to_py_values(result.get_column(col)) == expected


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_extract_datetime_features_from_specified_variables(make_df):
    data = dict(CROSS_BACKEND_DATA)
    data["date2"] = CROSS_BACKEND_DATES
    X = make_df(data)

    # single datetime variable
    Xt = DatetimeFeatures(variables="date").fit_transform(X)
    result = nw.from_native(Xt, eager_only=True)
    assert result.columns == vars_non_dt + ["date2"] + feat_names_default_cb
    for col, expected in _expected_cross_backend_features(FEATURES_DEFAULT).items():
        assert _to_py_values(result.get_column(col)) == expected

    # multiple datetime variables, in different order than they appear in X
    Xt = DatetimeFeatures(variables=["date2", "date"]).fit_transform(X)
    result = nw.from_native(Xt, eager_only=True)
    expected_cols = vars_non_dt + [
        f"date2{FEATURES_SUFFIXES[feat]}" for feat in FEATURES_DEFAULT
    ] + feat_names_default_cb
    assert result.columns == expected_cols
    for col, expected in _expected_cross_backend_features(FEATURES_DEFAULT).items():
        assert _to_py_values(result.get_column(col)) == expected
        assert _to_py_values(result.get_column(col.replace("date", "date2"))) == (
            expected
        )


def test_extract_datetime_features_from_index():
    # "index" is pandas-only: polars and other narwhals backends have no index.
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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_variables_index_raises_on_non_pandas(make_df):
    X = make_df(CROSS_BACKEND_DATA)
    transformer = DatetimeFeatures(variables="index")
    if make_df is pd.DataFrame:
        with pytest.raises(TypeError, match="The dataframe index is not datetime."):
            transformer.fit(X)
    else:
        with pytest.raises(TypeError, match="variables='index' requires a pandas"):
            transformer.fit(X)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_extract_all_datetime_features(make_df):
    X = make_df(CROSS_BACKEND_DATA)
    Xt = DatetimeFeatures(features_to_extract="all").fit_transform(X)

    result = nw.from_native(Xt, eager_only=True)
    expected = _expected_cross_backend_features(FEATURES_SUPPORTED)
    assert result.columns == vars_non_dt + list(expected.keys())
    for col, values in expected.items():
        assert _to_py_values(result.get_column(col)) == values


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "features", [["semester", "week"], ["hour", "day_of_week"]]
)
def test_extract_specified_datetime_features(make_df, features):
    X = make_df(CROSS_BACKEND_DATA)
    Xt = DatetimeFeatures(features_to_extract=features).fit_transform(X)

    result = nw.from_native(Xt, eager_only=True)
    expected = _expected_cross_backend_features(features)
    assert result.columns == vars_non_dt + list(expected.keys())
    for col, values in expected.items():
        assert _to_py_values(result.get_column(col)) == values


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
    exp_err_msg = "cannot be converted to datetime64 unless utc=True"
    with pytest.raises(ValueError) as errinfo:
        assert DatetimeFeatures(
            variables="time", features_to_extract=["hour"], utc=False
        ).fit_transform(df)
    assert exp_err_msg in str(errinfo.value)


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
                "2018-06-15 01:30:00",
                "2018-06-15 02:00:00",
                "2018-06-15 02:30:00",
                "2018-06-15 02:00:00",
                "2018-06-15 02:30:00",
                "2018-06-15 03:00:00",
                "2018-06-15 03:30:00",
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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_extract_features_without_dropping_original_variables(make_df):
    data = dict(CROSS_BACKEND_DATA)
    data["date2"] = CROSS_BACKEND_DATES
    X = make_df(data)

    Xt = DatetimeFeatures(
        variables=["date", "date2"],
        features_to_extract=["week", "quarter"],
        drop_original=False,
    ).fit_transform(X)

    result = nw.from_native(Xt, eager_only=True)
    expected_cols = (
        vars_non_dt
        + ["date", "date2"]
        + [
            f"{var}{FEATURES_SUFFIXES[feat]}"
            for var in ["date", "date2"]
            for feat in ["week", "quarter"]
        ]
    )
    assert result.columns == expected_cols
    for col, values in _expected_cross_backend_features(["week", "quarter"]).items():
        assert _to_py_values(result.get_column(col)) == values
        assert _to_py_values(result.get_column(col.replace("date", "date2"))) == (
            values
        )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_extract_features_from_variables_containing_nans(make_df):
    X = make_df({"dates_na": ["2010-02-01", None, "1922-06-01", None]})
    Xt = DatetimeFeatures(
        features_to_extract=["year"], missing_values="ignore"
    ).fit_transform(X)

    result = nw.from_native(Xt, eager_only=True).get_column("dates_na_year")
    values = result.to_list()
    assert values[0] == 2010 or values[0] == 2010.0
    assert values[1] is None or (isinstance(values[1], float) and np.isnan(values[1]))
    assert values[2] == 1922 or values[2] == 1922.0
    assert values[3] is None or (isinstance(values[3], float) and np.isnan(values[3]))


def test_extract_features_from_index_containing_nans():
    # "index" is pandas-only: polars and other narwhals backends have no index.
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


def test_polars_string_parsing_needs_explicit_format_for_ambiguous_dates():
    # dayfirst/yearfirst are pandas.to_datetime-only: narwhals' generic
    # str.to_datetime() has no day/year-first heuristic, so an ambiguous,
    # non-ISO format needs an explicit `format` on non-pandas input.
    X = pl.DataFrame({"date_obj1": ["01-Jan-2010", "24-Feb-1945"]})
    transformer = DatetimeFeatures(variables="date_obj1", features_to_extract=["year"])
    transformer.fit(X)
    with pytest.raises(Exception, match="could not find an appropriate format"):
        transformer.transform(X)

    transformer = DatetimeFeatures(
        variables="date_obj1", features_to_extract=["year"], format="%d-%b-%Y"
    )
    transformer.fit(X)
    Xt = transformer.transform(X)
    assert nw.from_native(Xt, eager_only=True).get_column(
        "date_obj1_year"
    ).to_list() == [2010, 1945]


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


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_get_feature_names_out_cross_backend(make_df):
    X = make_df(CROSS_BACKEND_DATA)
    transformer = DatetimeFeatures()
    Xt = transformer.fit_transform(X)
    result = nw.from_native(Xt, eager_only=True)
    assert result.columns == transformer.get_feature_names_out()

    transformer = DatetimeFeatures(drop_original=False)
    Xt = transformer.fit_transform(X)
    result = nw.from_native(Xt, eager_only=True)
    assert result.columns == transformer.get_feature_names_out()


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
