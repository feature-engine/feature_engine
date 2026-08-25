from datetime import datetime as _datetime

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from feature_engine.datetime import DatetimeSubtraction
from tests.estimator_checks.estimator_checks import (
    check_raises_error_when_input_not_a_df,
)
from tests.estimator_checks.fit_functionality_checks import (
    check_feature_names_in,
    check_return_empty,
)
from tests.estimator_checks.init_params_triggered_functionality_checks import (
    check_drop_original_variables,
)
from tests.estimator_checks.non_fitted_error_checks import check_raises_non_fitted_error

DATA_DATETIME = {
    "Name": ["tom", "nick", "krish", "jack"],
    "Age": [20, 21, 19, 18],
    "datetime_range": [
        _datetime(2020, 2, 24),
        _datetime(2020, 2, 25),
        _datetime(2020, 2, 26),
        _datetime(2020, 2, 27),
    ],
    "date_obj1": ["01-Jan-2010", "24-Feb-1945", "14-Jun-2100", "17-May-1999"],
    "date_obj2": ["10/11/12", "12/31/09", "06/30/95", "03/17/04"],
    "time_obj": ["21:45:23", "09:15:33", "12:34:59", "03:27:02"],
}

DATA_NAN = {
    "dates_na": ["Feb-2010", None, "Jun-1922", None],
    "dates_full": ["Feb-2010", "Mar-2010", "Jun-1922", "Feb-2011"],
}

DATA_NAN_FILLED = {
    "dates_na": ["Feb-2010", "Mar-2010", "Jun-1922", "Mar-2010"],
    "dates_full": ["Feb-2010", "Mar-2010", "Jun-1922", "Feb-2011"],
}


def assert_df_equal(X, expected: dict, abs_tol: float = 1e-5) -> None:
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(expected.keys())
    for col, values in expected.items():
        assert result[col] == pytest.approx(values, abs=abs_tol)


# ========= init functionality tests


@pytest.mark.parametrize(
    "_input_vars",
    [
        ("var1", "var2"),
        {"var1": 1, "var2": 2},
        ["var1", "var2", "var2", "var3"],
        [0, 1, 1, 2],
    ],
)
def test_init_parameters_variables_and_reference_raise_error(_input_vars):
    with pytest.raises(ValueError):
        assert DatetimeSubtraction(variables=_input_vars, reference=["var1"])
    with pytest.raises(ValueError):
        assert DatetimeSubtraction(variables=["var1"], reference=_input_vars)


@pytest.mark.parametrize("_input_vars", ["var1", ["var1"], ["var1", "var2"]])
def test_init_parameters_variables_and_reference_correct_assignment(_input_vars):
    transformer = DatetimeSubtraction(variables=_input_vars, reference=_input_vars)
    assert transformer.variables == _input_vars
    assert transformer.reference == _input_vars


def test_init_parameters_variables_and_reference_take_none():
    tr = DatetimeSubtraction()
    assert tr.variables is None
    assert tr.reference is None


@pytest.mark.parametrize(
    "_input_vars",
    [
        ("var1", "var2"),
        {"var1": 1, "var2": 2},
        "var1",
    ],
)
def test_new_variable_names_raise_errors(_input_vars):
    with pytest.raises(ValueError):
        assert DatetimeSubtraction(
            variables="var1", reference="var2", new_variables_names=_input_vars
        )


def test_new_variable_names_correct_assignment():
    tr = DatetimeSubtraction(variables="var1", reference="var2")
    assert tr.new_variables_names is None
    tr = DatetimeSubtraction(
        variables="var1", reference="var2", new_variables_names=["var1"]
    )
    assert tr.new_variables_names == ["var1"]
    tr = DatetimeSubtraction(
        variables="var1", reference="var2", new_variables_names=["var1", "var2"]
    )
    assert tr.new_variables_names == ["var1", "var2"]


@pytest.mark.parametrize(
    "output",
    [
        "D",
        "Y",
        "M",
        "W",
        "h",
        "m",
        "s",
        "ms",
        "us",
        "μs",
        "ns",
        "ps",
        "fs",
        "as",
    ],
)
def test_valid_output_unit_param(output):
    transformer = DatetimeSubtraction(
        variables=["var1"], reference=["var1"], output_unit=output
    )
    assert transformer.output_unit == output


@pytest.mark.parametrize("output", [["D"], "J", True, 1, 1.5])
def test_output_unit_raises_error_when_not_valid(output):
    with pytest.raises(ValueError):
        DatetimeSubtraction(variables=["var1"], reference=["var1"], output_unit=output)


@pytest.mark.parametrize("param", [True, False])
def test_drop_original_correct_assignment(param):
    transformer = DatetimeSubtraction(
        variables=["var1"], reference=["var1"], drop_original=param
    )
    assert transformer.drop_original is param


@pytest.mark.parametrize("param", [["D"], "J", 10, 1.5])
def test_drop_original_raises_error_when_not_valid(param):
    with pytest.raises(ValueError):
        DatetimeSubtraction(variables=["var1"], reference=["var1"], drop_original=param)


@pytest.mark.parametrize("param", ["ignore", "raise"])
def test_missing_values_correct_assignment(param):
    transformer = DatetimeSubtraction(
        variables=["var1"], reference=["var1"], missing_values=param
    )
    assert transformer.missing_values is param


@pytest.mark.parametrize("param", [["D"], "J", 10, 1.5])
def test_missing_values_raises_error_when_not_valid(param):
    with pytest.raises(ValueError):
        DatetimeSubtraction(
            variables=["var1"], reference=["var1"], missing_values=param
        )


# ==== fit functionality
@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("input_vars", [["Age", "date_obj2"], "Age"])
def test_raises_error_when_variables_not_datetime(make_df, input_vars):
    df = make_df(DATA_DATETIME)
    tr = DatetimeSubtraction(variables=input_vars, reference="date_obj1")
    with pytest.raises(TypeError):
        tr.fit(df)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("input_vars", [["Age", "date_obj2"], "Age"])
def test_raises_error_when_reference_not_datetime(make_df, input_vars):
    df = make_df(DATA_DATETIME)
    tr = DatetimeSubtraction(variables=["date_obj1"], reference=input_vars)
    with pytest.raises(TypeError):
        tr.fit(df)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("input_vars", [["time_obj", "date_obj2"], "date_obj2", None])
def test_sets_variables_if_datetime(make_df, input_vars):
    df = make_df(DATA_DATETIME)
    tr = DatetimeSubtraction(variables=input_vars, reference=input_vars)
    tr.fit(df)
    if input_vars is None:
        dt_vars = ["datetime_range", "date_obj1", "date_obj2", "time_obj"]
        assert tr.variables_ == dt_vars
        assert tr.reference_ == dt_vars
    elif input_vars == "date_obj2":
        assert tr.variables_ == ["date_obj2"]
        assert tr.reference_ == ["date_obj2"]
    else:
        assert tr.variables_ == ["time_obj", "date_obj2"]
        assert tr.reference_ == ["time_obj", "date_obj2"]


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("new", [["new1", "new2"], ["new1", "new2", "new3"]])
def test_new_variables_raise_error_if_not_adequate_number(make_df, new):
    df = make_df(DATA_DATETIME)
    tr = DatetimeSubtraction(
        variables="date_obj1", reference="date_obj1", new_variables_names=new
    )
    with pytest.raises(ValueError):
        tr.fit(df)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("input_vars_1", ["dates_full", None])
@pytest.mark.parametrize("input_vars_2", ["dates_na", ["dates_full", "dates_na"], None])
def test_raises_error_when_nan_in_variables_in_fit(
    make_df, input_vars_1, input_vars_2
):
    df_nan = make_df(DATA_NAN)
    tr = DatetimeSubtraction(
        variables=input_vars_2, reference=input_vars_1, missing_values="raise"
    )
    with pytest.raises(ValueError):
        tr.fit(df_nan)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("input_vars_1", ["dates_full", None])
@pytest.mark.parametrize("input_vars_2", ["dates_na", ["dates_full", "dates_na"], None])
def test_raises_error_when_nan_in_reference_in_fit(
    make_df, input_vars_1, input_vars_2
):
    df_nan = make_df(DATA_NAN)
    tr = DatetimeSubtraction(
        variables=input_vars_1, reference=input_vars_2, missing_values="raise"
    )
    with pytest.raises(ValueError):
        tr.fit(df_nan)


# transform tests
@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("input_vars_1", ["dates_full", None])
@pytest.mark.parametrize("input_vars_2", ["dates_na", ["dates_full", "dates_na"], None])
def test_raises_error_when_nan_in_variables_in_transform(
    make_df, input_vars_1, input_vars_2
):
    tr = DatetimeSubtraction(
        variables=input_vars_2, reference=input_vars_1, missing_values="raise"
    )
    tr.fit(make_df(DATA_NAN_FILLED))
    with pytest.raises(ValueError):
        tr.transform(make_df(DATA_NAN))


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("input_vars_1", ["dates_full", None])
@pytest.mark.parametrize("input_vars_2", ["dates_na", ["dates_full", "dates_na"], None])
def test_raises_error_when_nan_in_reference_in_transform(
    make_df, input_vars_1, input_vars_2
):
    tr = DatetimeSubtraction(
        variables=input_vars_1, reference=input_vars_2, missing_values="raise"
    )
    tr.fit(make_df(DATA_NAN_FILLED))
    with pytest.raises(ValueError):
        tr.transform(make_df(DATA_NAN))


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "unit, expected",
    [
        ("D", [31, 61, 183]),
        ("M", [1.018501, 2.004148, 6.012444]),
        ("h", [744.0, 1464.0, 4392.0]),
    ],
)
def test_subtraction_units(make_df, unit, expected):
    data = {
        "date1": ["2022-09-18", "2022-10-27", "2022-12-24"],
        "date2": ["2022-08-18", "2022-08-27", "2022-06-24"],
    }
    df_input = make_df(data)

    dtf = DatetimeSubtraction(
        variables=["date1"], reference=["date2"], output_unit=unit
    )
    df_output = dtf.fit_transform(df_input)

    expected_dict = dict(data)
    expected_dict["date1_sub_date2"] = expected
    assert_df_equal(df_output, expected_dict)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_multiple_subtractions(make_df):
    data = {
        "date1": ["2022-09-01", "2022-10-01", "2022-12-01"],
        "date2": ["2022-09-15", "2022-10-15", "2022-12-15"],
        "date3": ["2022-08-01", "2022-09-01", "2022-11-01"],
        "date4": ["2022-08-15", "2022-09-15", "2022-11-15"],
    }
    df_input = make_df(data)

    expected = dict(data)
    expected["date1_sub_date3"] = [31, 30, 30]
    expected["date2_sub_date3"] = [45, 44, 44]
    expected["date1_sub_date4"] = [17, 16, 16]
    expected["date2_sub_date4"] = [31, 30, 30]

    dtf = DatetimeSubtraction(
        variables=["date1", "date2"], reference=["date3", "date4"]
    )
    df_output = dtf.fit_transform(df_input)
    assert_df_equal(df_output, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_assigns_new_variable_names(make_df):
    data = {
        "date1": ["2022-09-01", "2022-10-01", "2022-12-01"],
        "date2": ["2022-09-15", "2022-10-15", "2022-12-15"],
        "date3": ["2022-08-01", "2022-09-01", "2022-11-01"],
        "date4": ["2022-08-15", "2022-09-15", "2022-11-15"],
    }
    df_input = make_df(data)

    expected = dict(data)
    expected["new1"] = [31, 30, 30]
    expected["new2"] = [45, 44, 44]
    expected["new3"] = [17, 16, 16]
    expected["new4"] = [31, 30, 30]

    dtf = DatetimeSubtraction(
        variables=["date1", "date2"],
        reference=["date3", "date4"],
        new_variables_names=["new1", "new2", "new3", "new4"],
    )
    df_output = dtf.fit_transform(df_input)
    assert_df_equal(df_output, expected)


# additional methods


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_get_feature_names_out(make_df):
    df = make_df(
        {
            "d1": ["Feb-2010", "Mar-2010", "Jun-1922", "Feb-2011"],
            "d2": ["Feb-2010", "Mar-2010", "Jun-1922", "Feb-2011"],
            "d3": ["Feb-2010", "Mar-2010", "Jun-1922", "Feb-2011"],
            "d4": ["Feb-2010", "Mar-2010", "Jun-1922", "Feb-2011"],
        }
    )
    input_vars = list(nw.from_native(df, eager_only=True).columns)

    tr = DatetimeSubtraction(variables="d1", reference="d2")
    tr.fit(df)
    assert tr.get_feature_names_out() == input_vars + ["d1_sub_d2"]

    tr = DatetimeSubtraction(variables=["d1", "d2"], reference="d3")
    tr.fit(df)
    assert tr.get_feature_names_out() == input_vars + ["d1_sub_d3", "d2_sub_d3"]

    tr = DatetimeSubtraction(variables="d3", reference=["d1", "d2"])
    tr.fit(df)
    assert tr.get_feature_names_out() == input_vars + ["d3_sub_d1", "d3_sub_d2"]

    tr = DatetimeSubtraction(variables=["d1", "d2"], reference=["d3", "d4"])
    tr.fit(df)
    assert tr.get_feature_names_out() == input_vars + [
        "d1_sub_d3",
        "d2_sub_d3",
        "d1_sub_d4",
        "d2_sub_d4",
    ]

    new = ["new1", "new2", "new3", "new4"]
    tr = DatetimeSubtraction(
        variables=["d1", "d2"], reference=["d3", "d4"], new_variables_names=new
    )
    tr.fit(df)
    assert tr.get_feature_names_out() == input_vars + new


# common tests
estimator = [DatetimeSubtraction(variables=["date1"], reference=["date2"])]


@pytest.mark.parametrize("estimator", estimator)
def test_common_tests(estimator):
    check_raises_non_fitted_error(estimator)
    check_raises_error_when_input_not_a_df(estimator)
    check_feature_names_in(estimator)
    check_drop_original_variables(estimator)
    check_return_empty(estimator)
