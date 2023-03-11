import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from feature_engine.datetime import DatetimeSubtraction


@pytest.mark.parametrize(
    "_input_vars",
    [
        ("var1", "var2"),
        {"var1": 1, "var2": 2},
        ["var1", "var2", "var2", "var3"],
        [0, 1, 1, 2],
    ],
)
def test_init_parameters_variables_and_reference_raises_errors(_input_vars):
    with pytest.raises(ValueError):
        assert DatetimeSubtraction(variables=_input_vars, reference=["var1"])
    with pytest.raises(ValueError):
        assert DatetimeSubtraction(reference=_input_vars, variables=["var1"])


@pytest.mark.parametrize("_input_vars", ["var1", ["var1"], ["var1", "var2"]])
def test_init_parameters_variables_and_reference(_input_vars):
    transformer = DatetimeSubtraction(variables=_input_vars, reference=_input_vars)
    assert transformer.variables == _input_vars
    assert transformer.reference == _input_vars


@pytest.mark.parametrize("_input_vars", ["var1", ["var1"], ["var1", "var2"]])
def test_mandatory_init_parameters(_input_vars):
    with pytest.raises(TypeError):
        DatetimeSubtraction(reference=["var1"])
    with pytest.raises(TypeError):
        DatetimeSubtraction(variables=["var1"])


@pytest.mark.parametrize("output", [ "D", "Y", "M", "W", "h", "m", "s", "ms", "us", "μs", "ns", "ps", "fs", "as",])
def test_valid_output_unit_param(output):
    transformer = DatetimeSubtraction(variables=["var1"], reference=["var1"], output_unit=output)
    assert transformer.output_unit == output


@pytest.mark.parametrize("output", [ ["D"], "J", True, 1, 1.5])
def test_output_unit_raises_error_when_not_valid(output):
    with pytest.raises(ValueError):
        DatetimeSubtraction(variables=["var1"], reference=["var1"], output_unit=output)


@pytest.mark.parametrize("output", [ ["D"], "J", True, 1, 1.5])
def test_output_unit_raises_error_when_not_valid(output):
    with pytest.raises(ValueError):
        DatetimeSubtraction(variables=["var1"], reference=["var1"], output_unit=output)


def test_raises_error_when_variables_not_datetime(df_datetime):
    with pytest.raises(TypeError):
        DatetimeSubtraction(variables="Age", reference="date_obj1").fit(df_datetime)
    with pytest.raises(TypeError):
        DatetimeSubtraction(variables=["date_obj1"], reference=["Age"]).fit(df_datetime)


def test_sets_variables_if_datetime(df_datetime):
    tr = DatetimeSubtraction(variables="date_obj1", reference="date_obj1").fit(df_datetime)
    assert tr.variables_ == ["date_obj1"]
    assert tr.reference_ == ["date_obj1"]


def test_raises_error_when_nan_in_fit():
    df = pd.DataFrame({
        "dates_na": ["Feb-2010", np.nan, "Jun-1922", np.nan],
        "dates_full":["Feb-2010", "Mar-2010", "Jun-1922", "Feb-2011"],
    })

    tr = DatetimeSubtraction(variables="dates_na", reference="dates_full", missing_values="raise")
    with pytest.raises(ValueError):
        tr.fit(df)

    tr = DatetimeSubtraction(variables="dates_full", reference="dates_na", missing_values="raise")
    with pytest.raises(ValueError):
        tr.fit(df)


def test_raises_error_when_nan_in_transform():
    df_fit = pd.DataFrame({
        "dates_na": ["Feb-2010", "Mar-2010", "Jun-1922", "Feb-2011"],
        "dates_full":["Feb-2010", "Mar-2010", "Jun-1922", "Feb-2011"],
    })
    df_transform = pd.DataFrame({
        "dates_na": ["Feb-2010", np.nan, "Jun-1922", np.nan],
        "dates_full":["Feb-2010", "Mar-2010", "Jun-1922", "Feb-2011"],
    })

    tr = DatetimeSubtraction(variables="dates_na", reference="dates_full", missing_values="raise")
    tr.fit(df_fit)
    with pytest.raises(ValueError):
        tr.fit(df_transform)

    tr = DatetimeSubtraction(variables="dates_full", reference="dates_na", missing_values="raise")
    tr.fit(df_fit)
    with pytest.raises(ValueError):
        tr.fit(df_transform)



