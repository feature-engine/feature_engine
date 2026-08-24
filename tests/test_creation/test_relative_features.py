import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from feature_engine.creation import RelativeFeatures

DATA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
}


def _none_to_nan(values):
    # Missing values print as None for polars, NaN for pandas float columns
    # - both mean "missing" here, so normalize both sides before comparing.
    return [np.nan if v is None else v for v in values]


def assert_df_equal(X, expected: dict, abs_tol: float = 1e-5) -> None:
    result = nw.from_native(X, eager_only=True).to_dict(as_series=False)
    assert list(result.keys()) == list(expected.keys())
    for col, values in expected.items():
        assert _none_to_nan(result[col]) == pytest.approx(
            _none_to_nan(values), abs=abs_tol, nan_ok=True
        )


def test_mandatory_init_parameters():
    with pytest.raises(TypeError):
        RelativeFeatures(reference=["var1"], func=["add"])
    with pytest.raises(TypeError):
        RelativeFeatures(variables=["var1"], func=["add"])
    with pytest.raises(TypeError):
        RelativeFeatures(variables=["var1"], reference=["var2"])


_variables = ["var1", ["var1", "var1", "var2"], ["var1", 0.5], ("Age", "Name")]


@pytest.mark.parametrize("_variables", _variables)
def test_error_when_param_variables_not_permitted(_variables):
    with pytest.raises(ValueError):
        RelativeFeatures(
            variables=_variables, reference=["Age", "Name"], func=["add", "mul"]
        )


@pytest.mark.parametrize("_variables", _variables)
def test_error_when_param_reference_not_permitted(_variables):
    with pytest.raises(ValueError):
        RelativeFeatures(
            reference=_variables, variables=["Age", "Name"], func=["add", "mul"]
        )


_operations = [
    "add",
    ["add", "add", "mul"],
    ["add", "multiply"],
    ("add", "mul"),
    [np.mean, "add"],
]


@pytest.mark.parametrize("_func", _operations)
def test_error_if_func_not_supported(_func):
    with pytest.raises(ValueError):
        RelativeFeatures(
            variables=["Age", "Name"],
            reference=["Age", "Name"],
            func=_func,
        )


@pytest.mark.parametrize("_fill_value", [(2, 3.3), ["test"], "python"])
def test_error_if_fill_value_not_permitted(_fill_value):
    with pytest.raises(ValueError):
        RelativeFeatures(
            variables=["Age"],
            reference=["Marks"],
            func=["sub", "div", "add", "mul"],
            fill_value=_fill_value,
        )


def test_error_when_drop_original_not_bool():
    for drop_original in ["True", [True]]:
        with pytest.raises(ValueError):
            RelativeFeatures(
                variables=["Age"],
                reference=["Marks"],
                func=["add", "mul"],
                drop_original=drop_original,
            )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_when_variables_not_numeric(make_df):
    df = make_df(DATA)
    transformer = RelativeFeatures(
        variables=["Name", "Age", "Marks"],
        reference=["Age", "Name"],
        func=["sub"],
    )
    with pytest.raises(TypeError):
        transformer.fit_transform(df)

    transformer = RelativeFeatures(
        reference=["Name", "Age", "Marks"],
        variables=["Age", "Name"],
        func=["sub"],
    )
    with pytest.raises(TypeError):
        transformer.fit_transform(df)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_when_entered_variables_not_in_df(make_df):
    df = make_df(DATA)
    transformer = RelativeFeatures(
        variables=["FeatOutsideDataset", "Age"],
        reference=["Age", "Name"],
        func=["sub"],
    )
    with pytest.raises(KeyError):
        transformer.fit_transform(df)

    transformer = RelativeFeatures(
        reference=["FeatOutsideDataset", "Age"],
        variables=["Age", "Name"],
        func=["sub"],
    )
    with pytest.raises(TypeError):
        transformer.fit_transform(df)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_classic_binary_operation(make_df):
    df = make_df(DATA)
    transformer = RelativeFeatures(
        variables=["Age"],
        reference=["Marks"],
        func=["sub", "div", "add", "mul"],
    )
    Xt = transformer.fit_transform(df)

    expected = dict(DATA)
    expected["Age_sub_Marks"] = [19.1, 20.2, 18.3, 17.4]
    expected["Age_div_Marks"] = [22.22222222222222, 26.25, 27.142857142857146, 30.0]
    expected["Age_add_Marks"] = [20.9, 21.8, 19.7, 18.6]
    expected["Age_mul_Marks"] = [18.0, 16.8, 13.3, 10.8]

    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_alternative_operation(make_df):
    df = make_df(DATA)
    transformer = RelativeFeatures(
        variables=["Age"],
        reference=["Marks"],
        func=["truediv", "floordiv", "mod", "pow"],
    )
    Xt = transformer.fit_transform(df)

    expected = dict(DATA)
    expected["Age_truediv_Marks"] = [22.22222222222222, 26.25, 27.142857142857146, 30.0]
    expected["Age_floordiv_Marks"] = [22.0, 26.0, 27.0, 30.0]
    expected["Age_mod_Marks"] = [
        0.1999999999999995,
        0.19999999999999885,
        0.1000000000000012,
        6.661338147750939e-16,
    ]
    expected["Age_pow_Marks"] = [
        14.822688982138954,
        11.42287530066645,
        7.85466234994081,
        5.664525067769412,
    ]

    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_operations_with_multiple_variables(make_df):
    df = make_df(DATA)
    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["sub"],
    )
    Xt = transformer.fit_transform(df)

    expected = dict(DATA)
    expected["Age_sub_Age"] = [0, 0, 0, 0]
    expected["Marks_sub_Age"] = [-19.1, -20.2, -18.3, -17.4]
    expected["Age_sub_Marks"] = [19.1, 20.2, 18.3, 17.4]
    expected["Marks_sub_Marks"] = [0.0, 0.0, 0.0, 0.0]

    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_multiple_operations_with_multiple_variables(make_df):
    df = make_df(DATA)

    # column order follows func order: sub's 4 columns, then add's 4
    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["sub", "add"],
    )
    Xt = transformer.fit_transform(df)

    expected = dict(DATA)
    expected["Age_sub_Age"] = [0, 0, 0, 0]
    expected["Marks_sub_Age"] = [-19.1, -20.2, -18.3, -17.4]
    expected["Age_sub_Marks"] = [19.1, 20.2, 18.3, 17.4]
    expected["Marks_sub_Marks"] = [0.0, 0.0, 0.0, 0.0]
    expected["Age_add_Age"] = [40, 42, 38, 36]
    expected["Marks_add_Age"] = [20.9, 21.8, 19.7, 18.6]
    expected["Age_add_Marks"] = [20.9, 21.8, 19.7, 18.6]
    expected["Marks_add_Marks"] = [1.8, 1.6, 1.4, 1.2]

    assert_df_equal(Xt, expected)

    # reversing func order reverses the corresponding column block order
    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["add", "sub"],
    )
    Xt = transformer.fit_transform(df)

    expected = dict(DATA)
    expected["Age_add_Age"] = [40, 42, 38, 36]
    expected["Marks_add_Age"] = [20.9, 21.8, 19.7, 18.6]
    expected["Age_add_Marks"] = [20.9, 21.8, 19.7, 18.6]
    expected["Marks_add_Marks"] = [1.8, 1.6, 1.4, 1.2]
    expected["Age_sub_Age"] = [0, 0, 0, 0]
    expected["Marks_sub_Age"] = [-19.1, -20.2, -18.3, -17.4]
    expected["Age_sub_Marks"] = [19.1, 20.2, 18.3, 17.4]
    expected["Marks_sub_Marks"] = [0.0, 0.0, 0.0, 0.0]

    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_when_missing_values_is_ignore(make_df):
    data_na = dict(DATA)
    data_na["Age"] = [20, None, 19, 18]
    df_na = make_df(data_na)

    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["sub"],
        missing_values="ignore",
    )
    Xt = transformer.fit_transform(df_na)

    expected = dict(data_na)
    expected["Age_sub_Age"] = [0, np.nan, 0, 0]
    expected["Marks_sub_Age"] = [-19.1, np.nan, -18.3, -17.4]
    expected["Age_sub_Marks"] = [19.1, np.nan, 18.3, 17.4]
    expected["Marks_sub_Marks"] = [0.0, 0.0, 0.0, 0.0]

    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_when_null_values_in_variable(make_df):
    data_na = dict(DATA)
    data_na["Age"] = [20, None, 19, 18]
    df_na = make_df(data_na)

    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["add", "mul"],
        missing_values="raise",
    )

    with pytest.raises(ValueError):
        transformer.fit(df_na)

    transformer.fit(make_df(DATA))
    with pytest.raises(ValueError):
        transformer.transform(df_na)


def test_when_df_cols_are_integers():
    # polars requires string column names, so int-named columns are
    # pandas-only - no polars equivalent to parametrize against here.
    df = pd.DataFrame(DATA)
    df.columns = [0, 1, 2, 3]

    transformer = RelativeFeatures(
        variables=[2, 3],
        reference=[2, 3],
        func=["sub", "add"],
    )

    X = transformer.fit_transform(df)

    ref = pd.DataFrame.from_dict(
        {
            0: ["tom", "nick", "krish", "jack"],
            1: ["London", "Manchester", "Liverpool", "Bristol"],
            2: [20, 21, 19, 18],
            3: [0.9, 0.8, 0.7, 0.6],
            "2_sub_2": [0, 0, 0, 0],
            "3_sub_2": [-19.1, -20.2, -18.3, -17.4],
            "2_sub_3": [19.1, 20.2, 18.3, 17.4],
            "3_sub_3": [0.0, 0.0, 0.0, 0.0],
            "2_add_2": [40, 42, 38, 36],
            "3_add_2": [20.9, 21.8, 19.7, 18.6],
            "2_add_3": [20.9, 21.8, 19.7, 18.6],
            "3_add_3": [1.8, 1.6, 1.4, 1.2],
        }
    )

    pd.testing.assert_frame_equal(X, ref)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("_func", [["div"], ["truediv"], ["floordiv"], ["mod"]])
def test_error_when_division_by_zero_and_fill_value_is_none(make_df, _func):
    data_zero = dict(DATA)
    data_zero["Marks"] = [0.9, 0, 0.7, 0.6]
    df_zero = make_df(data_zero)

    transformer = RelativeFeatures(
        variables=["Age"],
        reference=["Marks"],
        func=_func,
    )
    transformer.fit(make_df(DATA))

    msg = (
        "Some of the reference variables contain zeroes. Division by zero "
        "does not exist. Replace zeros before using this transformer for division "
        "or set `fill_value` to a number."
    )
    with pytest.raises(ValueError, match=msg):
        transformer.transform(df_zero)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize(
    "_fill_value, _func",
    [
        (111.111, ["div"]),
        (999, ["div"]),
        (111.111, ["truediv"]),
        (999, ["truediv"]),
        (111.111, ["floordiv"]),
        (999, ["floordiv"]),
        (111.111, ["mod"]),
        (999, ["mod"]),
    ],
)
def test_fill_values_when_division_by_zero(make_df, _fill_value, _func):
    data_zero = dict(DATA)
    data_zero["Marks"] = [0.9, 0.8, 0, 0.6]
    # Age must be float from the start: polars can't build an Int64 column
    # from a mix of ints and NaN/inf the way pandas silently upcasts to.
    data_zero["Age"] = [20.0, np.nan, 19.0, np.inf]
    df_zero = make_df(data_zero)

    transformer = RelativeFeatures(
        variables=["Age"],
        reference=["Marks"],
        fill_value=_fill_value,
        func=_func,
        missing_values="ignore",
    )
    Xt = transformer.fit_transform(df_zero)

    new_var = f"Age_{_func[0]}_Marks"
    result = nw.from_native(Xt, eager_only=True).to_dict(as_series=False)

    assert result[new_var][2] == pytest.approx(_fill_value)
    np.testing.assert_equal(result["Age"][1], np.nan)
    np.testing.assert_equal(result["Age"][3], np.inf)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("_drop", [True, False])
def test_get_feature_names_out(make_df, _drop):
    df = make_df(DATA)
    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["add", "sub"],
        drop_original=_drop,
    )
    varnames = [
        "Age_add_Age",
        "Marks_add_Age",
        "Age_add_Marks",
        "Marks_add_Marks",
        "Age_sub_Age",
        "Marks_sub_Age",
        "Age_sub_Marks",
        "Marks_sub_Marks",
    ]
    Xt = transformer.fit_transform(df)
    feat_out = list(nw.from_native(Xt, eager_only=True).columns)
    assert feat_out == transformer.get_feature_names_out(input_features=None)
    assert all([f for f in varnames if f in feat_out])
    if _drop is True:
        # drop_original only drops columns that are in variables/reference
        # (here Age, Marks) - Name and City are neither, so they remain.
        assert feat_out == ["Name", "City"] + varnames
    else:
        assert feat_out == list(DATA.keys()) + varnames


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("_drop", [True, False])
def test_get_feature_names_out_from_pipeline(make_df, _drop):
    df = make_df(DATA)
    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["add", "sub"],
        drop_original=_drop,
    )
    pipe = Pipeline([("transformer", transformer)])

    Xt = pipe.fit_transform(df)
    feat_out = list(nw.from_native(Xt, eager_only=True).columns)
    assert feat_out == pipe.get_feature_names_out(input_features=None)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("_input_features", ["hola", ["Age", "Marks"]])
def test_get_feature_names_out_raises_error_when_wrong_param(make_df, _input_features):
    df = make_df(DATA)
    transformer = RelativeFeatures(
        variables=["Age", "Marks"],
        reference=["Age", "Marks"],
        func=["add", "sub"],
    )
    transformer.fit(df)

    with pytest.raises(ValueError):
        transformer.get_feature_names_out(input_features=_input_features)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_mixed_int_float_variables_preserve_own_dtype(make_df):
    # a regression check: extracting variables as one batched 2D array
    # upcasts everything to a common dtype, losing e.g. an int column's
    # own int result for subtraction. Each variable must keep its own
    # dtype promotion, independent of the other variables in the list.
    df = make_df(DATA)
    transformer = RelativeFeatures(
        variables=["Age", "Marks"], reference=["Age"], func=["sub"]
    )
    Xt = transformer.fit_transform(df)
    nw_Xt = nw.from_native(Xt, eager_only=True)
    assert nw_Xt.get_column("Age_sub_Age").dtype.is_integer()
    assert not nw_Xt.get_column("Marks_sub_Age").dtype.is_integer()


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_floordiv_zero_with_float_fill_value_widens_dtype(make_df):
    # floordiv on integer input stays integer-typed; a float fill_value
    # must widen the result column rather than truncating or erroring,
    # matching pandas' own automatic dtype promotion here.
    df = make_df({"v": [7, 8], "ref": [0, 2]})
    transformer = RelativeFeatures(
        variables=["v"], reference=["ref"], func=["floordiv"], fill_value=-1.5
    )
    Xt = transformer.fit_transform(df)
    result = nw.from_native(Xt, eager_only=True).get_column("v_floordiv_ref").to_list()
    assert result == pytest.approx([-1.5, 4.0])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_drop_original_both_backends(make_df):
    df = make_df({"x1": [1, 2, 3], "x2": [4, 5, 6], "x3": [3, 4, 5]})
    transformer = RelativeFeatures(
        variables=["x1", "x2"], reference=["x3"], func=["div"], drop_original=True
    )
    Xt = transformer.fit_transform(df)
    assert list(nw.from_native(Xt, eager_only=True).columns) == [
        "x1_div_x3",
        "x2_div_x3",
    ]
