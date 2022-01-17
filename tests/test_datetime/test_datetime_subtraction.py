import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.datetime.datetime_subtraction import DatetimeSubtraction

vars_dt = ["datetime_range", "date_obj1", "date_obj2", "time_obj"]
vars_non_dt = ["Name", "Age"]
dates_nan = pd.DataFrame({"dates_na": ["Feb-2010", np.nan, "Jun-1922", np.nan]})
var_pairs = \
    [
        ('datetime_range', 'date_obj1'),
        ('datetime_range', 'date_obj2'),
        ('datetime_range', 'time_obj'),
        ('date_obj1', 'datetime_range'),
        ('date_obj1', 'date_obj2'),
        ('date_obj1', 'time_obj'),
        ('date_obj2', 'datetime_range'),
        ('date_obj2', 'date_obj1'),
        ('date_obj2', 'time_obj'),
        ('time_obj', 'datetime_range'),
        ('time_obj', 'date_obj1'),
        ('time_obj', 'date_obj2')
     ]

var_pairs_deduped = \
    [
        ('date_obj1', 'datetime_range'),
        ('date_obj2', 'datetime_range'),
        ('datetime_range', 'time_obj'),
        ('date_obj1', 'date_obj2'),
        ('date_obj1', 'time_obj'),
        ('date_obj2', 'time_obj')
     ]



def test_default_params():
    transformer = DatetimeSubtraction()
    assert isinstance(transformer, DatetimeSubtraction)
    assert transformer.reference_variables is None
    assert transformer.variables_to_combine is None
    assert transformer.output_unit is 'D'
    assert transformer.dedupe_variable_pairs is False
    assert transformer.new_variables_names is None
    assert transformer.missing_values == "raise"
    assert transformer.drop_original is True
    assert transformer.utc is None
    assert transformer.dayfirst is False
    assert transformer.yearfirst is False


_false_input_params = [
    (["not_supported"], 3.519, "wrong_option"),
    (["year", 1874], [1, -1.09, "var3"], 1),
    ("year", [3.5], [True, False]),
    (14198, [0.1, False], {True}),
]

# @pytest.mark.parametrize(
#     "_features_to_extract, _variables, _other_params", _false_input_params
# )
# def test_raises_error_when_wrong_input_params(
#     _features_to_extract, _variables, _other_params
# ):
#     with pytest.raises(ValueError):
#         assert DatetimeSubtraction(reference_variables=_variables)
#     with pytest.raises(ValueError):
#         assert DatetimeSubtraction(variables_to_combine=_variables)
#     with pytest.raises(ValueError):
#         assert DatetimeSubtraction(missing_values=_other_params)
#     with pytest.raises(ValueError):
#         assert DatetimeSubtraction(drop_original=_other_params)
#     with pytest.raises(ValueError):
#         assert DatetimeSubtraction(utc=_other_params)



_variables = [0, [0, 1, 9, 23], ["var_str1", "var_str2"], [0, 1, "var3", 3]]


@pytest.mark.parametrize("_variables", _variables)
def test_reference_variables_params(_variables):
    assert DatetimeSubtraction(reference_variables=_variables).reference_variables ==\
           _variables


@pytest.mark.parametrize("_variables", _variables)
def test_reference_variables_params(_variables):
    assert DatetimeSubtraction(variables_to_combine=_variables).variables_to_combine ==\
           _variables


_not_a_df = [
    "not_a_df",
    [1, 2, 3, "some_data"],
    pd.Series([-2, 1.5, 8.94], name="not_a_df"),
]


@pytest.mark.parametrize("_not_a_df", _not_a_df)
def test_raises_error_when_fitting_not_a_df(_not_a_df):
    transformer = DatetimeSubtraction()
    # trying to fit not a df
    with pytest.raises(TypeError):
        transformer.fit(_not_a_df)


def test_raises_error_when_variables_not_datetime(df_datetime):
    # asking for not datetime variable(s)
    with pytest.raises(TypeError):
        DatetimeSubtraction(reference_variables=["Age"]).fit(df_datetime)
    with pytest.raises(TypeError):
        DatetimeSubtraction(reference_variables=["Name", "Age", "date_obj1"])\
            .fit(df_datetime)
    with pytest.raises(TypeError):
        DatetimeSubtraction(variables_to_combine=["Age"]).fit(df_datetime)
    with pytest.raises(TypeError):
        DatetimeSubtraction(variables_to_combine=["Name", "Age", "date_obj1"])\
            .fit(df_datetime)
    # passing a df that contains no datetime variables
    with pytest.raises(ValueError):
        DatetimeSubtraction().fit(df_datetime[["Name", "Age"]])


def test_raises_error_when_df_has_nan():
    # dataset containing nans
    with pytest.raises(ValueError):
        DatetimeSubtraction().fit(dates_nan)


def test_attributes_upon_fitting(df_datetime):
    transformer = DatetimeSubtraction()
    transformer.fit(df_datetime)

    assert transformer.reference_variables_ == vars_dt
    assert transformer.variables_to_combine_ == vars_dt
    assert transformer.variable_pairs_ == var_pairs
    assert transformer.n_features_in_ == df_datetime.shape[1]

    transformer = DatetimeSubtraction(dedupe_variable_pairs=True)
    transformer.fit(df_datetime)

    assert transformer.reference_variables_ == vars_dt
    assert transformer.variables_to_combine_ == vars_dt
    assert transformer.variable_pairs_ == var_pairs_deduped
    assert transformer.n_features_in_ == df_datetime.shape[1]

    transformer = DatetimeSubtraction(reference_variables=["date_obj1"])
    transformer.fit(df_datetime)

    assert transformer.reference_variables_ == ["date_obj1"]

    transformer = DatetimeSubtraction(variables_to_combine=["date_obj1"])
    transformer.fit(df_datetime)

    assert transformer.variables_to_combine_ == ["date_obj1"]

    transformer = DatetimeSubtraction(
        reference_variables=["date_obj1", "time_obj"],
        variables_to_combine=["date_obj2", "datetime_range"],
    )
    transformer.fit(df_datetime)

    assert transformer.reference_variables_ == ["date_obj1", "time_obj"]
    assert transformer.variables_to_combine_ == ["date_obj2", "datetime_range"]
    assert transformer.variable_pairs_ == \
            [
                ('date_obj1', 'date_obj2'),
                ('date_obj1', 'datetime_range'),
                ('time_obj', 'date_obj2'),
                ('time_obj', 'datetime_range')
             ]


@pytest.mark.parametrize("_not_a_df", _not_a_df)
def test_raises_error_when_transforming_not_a_df(_not_a_df, df_datetime):
    transformer = DatetimeSubtraction()
    transformer.fit(df_datetime)
    # trying to transform not a df
    with pytest.raises(TypeError):
        transformer.transform(_not_a_df)


def test_raises_error_when_transform_df_with_different_n_variables(df_datetime):
    transformer = DatetimeSubtraction()
    transformer.fit(df_datetime)
    # different number of columns than the df used to fit
    with pytest.raises(ValueError):
        transformer.transform(df_datetime[vars_dt])


def test_raises_error_when_nan_in_transform_df(df_datetime):
    transformer = DatetimeSubtraction()
    transformer.fit(df_datetime)
    # dataset containing nans
    with pytest.raises(ValueError):
        DatetimeSubtraction().transform(dates_nan)


def test_raises_non_fitted_error(df_datetime):
    # trying to transform before fitting
    with pytest.raises(NotFittedError):
        DatetimeSubtraction().transform(df_datetime)


def test_extract_all_datetime_features(df_datetime, df_datetime_sub_transformed):
    X = DatetimeSubtraction().fit_transform(df_datetime)
    pd.testing.assert_frame_equal(X, df_datetime_sub_transformed)

#
# def test_extract_features_from_categorical_variable(
#     df_datetime, df_datetime_sub_transformed
# ):
#     cat_date = pd.DataFrame({"date_obj1": df_datetime["date_obj1"].astype("category")})
#     X = DatetimeSubtraction(variables="date_obj1").fit_transform(cat_date)
#     pd.testing.assert_frame_equal(
#         X, df_datetime_sub_transformed[["date_obj1" + feat for feat in feat_names_default]]
#     )
#
#
# def test_extract_features_from_different_timezones(
#     df_datetime, df_datetime_sub_transformed
# ):
#     time_zones = [4, -1, 9, -7]
#     tz_df = pd.DataFrame(
#         {"time_obj": df_datetime["time_obj"].add(["+4", "-1", "+9", "-7"])}
#     )
#     transformer = DatetimeSubtraction(
#         variables="time_obj", features_to_extract=["hour"], utc=True
#     )
#     X = transformer.fit_transform(tz_df)
#
#     pd.testing.assert_frame_equal(
#         X,
#         df_datetime_sub_transformed[["time_obj_hour"]].apply(
#             lambda x: x.subtract(time_zones)
#         ),
#     )
#     exp_err_msg = "ValueError: variable(s) time_obj " \
#         "could not be converted to datetime. Try setting utc=True"
#     with pytest.raises(ValueError) as errinfo:
#         assert DatetimeSubtraction(
#             variables="time_obj", features_to_extract=["hour"], utc=False
#         ).fit_transform(tz_df)
#     assert str(errinfo.value) == exp_err_msg
#
#
# def test_extract_features_from_localized_tz_variables():
#     tz_df = pd.DataFrame(
#         {
#             "date_var": [
#                 "2018-10-28 01:30:00",
#                 "2018-10-28 02:00:00",
#                 "2018-10-28 02:30:00",
#                 "2018-10-28 02:00:00",
#                 "2018-10-28 02:30:00",
#                 "2018-10-28 03:00:00",
#                 "2018-10-28 03:30:00",
#             ]
#         }
#     )
#
#     tz_df["date_var"] = pd.to_datetime(tz_df["date_var"]).dt.tz_localize(
#         tz="US/Eastern"
#     )
#
#     # when utc is None
#     transformer = DatetimeSubtraction(features_to_extract=["hour"]).fit(tz_df)
#
#     # init params
#     assert transformer.variables is None
#     assert transformer.utc is None
#     assert transformer.features_to_extract == ["hour"]
#     # fit attr
#     assert transformer.variables_ == ["date_var"]
#     assert transformer.features_to_extract_ == ["hour"]
#     assert transformer.n_features_in_ == 1
#     # transform
#     X = transformer.transform(tz_df)
#     df_expected = pd.DataFrame({"date_var_hour": [1, 2, 2, 2, 2, 3, 3]})
#     pd.testing.assert_frame_equal(X, df_expected)
#
#     # when utc is True
#     transformer = DatetimeSubtraction(features_to_extract=["hour"], utc=True).fit(tz_df)
#
#     # init params
#     assert transformer.variables is None
#     assert transformer.utc is True
#     assert transformer.features_to_extract == ["hour"]
#     # fit attr
#     assert transformer.variables_ == ["date_var"]
#     assert transformer.features_to_extract_ == ["hour"]
#     assert transformer.n_features_in_ == 1
#     # transform
#     X = transformer.transform(tz_df)
#     df_expected = pd.DataFrame({"date_var_hour": [5, 6, 6, 6, 6, 7, 7]})
#     pd.testing.assert_frame_equal(X, df_expected)
#
#
# def test_extract_features_without_dropping_original_variables(
#     df_datetime, df_datetime_sub_transformed
# ):
#     X = DatetimeSubtraction(
#         variables=["datetime_range", "date_obj2"],
#         features_to_extract=["week", "quarter"],
#         drop_original=False,
#     ).fit_transform(df_datetime)
#
#     pd.testing.assert_frame_equal(
#         X,
#         pd.concat(
#             [df_datetime_sub_transformed[column] for column in vars_non_dt]
#             + [df_datetime[var] for var in vars_dt]
#             + [
#                 df_datetime_sub_transformed[feat]
#                 for feat in [
#                     var + "_" + feat
#                     for var in ["datetime_range", "date_obj2"]
#                     for feat in ["week", "quarter"]
#                 ]
#             ],
#             axis=1,
#         ),
#     )
#
#
# def test_extract_features_from_variables_containing_nans():
#     X = DatetimeSubtraction(
#         features_to_extract=["year"], missing_values="ignore"
#     ).fit_transform(dates_nan)
#     pd.testing.assert_frame_equal(
#         X,
#         pd.DataFrame({"dates_na_year": [2010, np.nan, 1922, np.nan]}),
#     )
#
#
# def test_extract_features_with_different_datetime_parsing_options(df_datetime):
#     X = DatetimeSubtraction(
#         features_to_extract=["day_of_month"], dayfirst=True
#     ).fit_transform(df_datetime[["date_obj2"]])
#     pd.testing.assert_frame_equal(
#         X,
#         pd.DataFrame({"date_obj2_day_of_month": [10, 31, 30, 17]}),
#     )
#
#     X = DatetimeSubtraction(features_to_extract=["year"], yearfirst=True).fit_transform(
#         df_datetime[["date_obj2"]]
#     )
#     pd.testing.assert_frame_equal(
#         X,
#         pd.DataFrame({"date_obj2_year": [2010, 2009, 1995, 2004]}),
#     )
