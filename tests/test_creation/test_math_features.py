import warnings

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from feature_engine.creation import MathFeatures

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


# test param variables_to_combine
def test_error_when_required_params_not_entered():
    with pytest.raises(TypeError):
        MathFeatures(func=["mean", "sum"])
    with pytest.raises(TypeError):
        MathFeatures(variables=["vara", "varb"])


@pytest.mark.parametrize(
    "_variables",
    [
        ["vara", "vara", "varb"],
        ["vara", "vara"],
        "vara",
        ["vara"],
        1,
        [1, 1, 2],
        [2],
        ["vara", 0.5],
    ],
)
def test_error_when_variables_not_permitted(_variables):
    with pytest.raises(ValueError):
        MathFeatures(variables=_variables, func=["sum", "mean"])


def test_error_if_func_is_dictionary():
    with pytest.raises(NotImplementedError):
        MathFeatures(variables=["Age", "Name"], func={"A": "sum", "B": "mean"})


@pytest.mark.parametrize("_variables", [[4], ("vara", "vara"), "vara"])
def test_error_if_new_variable_names_not_permitted(_variables):
    with pytest.raises(ValueError):
        MathFeatures(
            variables=["Age", "Name"], func=["sum"], new_variables_names=_variables
        )


def test_error_new_variable_names_not_permitted():
    variables = ["Age", "Name"]
    with pytest.raises(ValueError):
        MathFeatures(
            variables=variables,
            func=["sum", "mean"],
            new_variables_names=[
                "sum_of_two_vars",
                "mean_of_two_vars",
                "another_alias",
            ],
        )

    with pytest.raises(ValueError):
        MathFeatures(
            variables=variables,
            func=["sum"],
            new_variables_names=["sum_of_two_vars", "mean_of_two_vars"],
        )

    with pytest.raises(ValueError):
        MathFeatures(
            variables=variables,
            func="sum",
            new_variables_names=["sum_of_two_vars", "mean_of_two_vars"],
        )
    with pytest.raises(ValueError):
        MathFeatures(
            variables=variables,
            func=["sum", "mean"],
            new_variables_names=["sum_of_two_vars", "sum_of_two_vars"],
        )


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_aggregations_with_strings(make_df):
    df = make_df(DATA)
    transformer = MathFeatures(
        variables=["Age", "Marks"], func=["sum", "prod", "mean", "std", "max", "min"]
    )
    Xt = transformer.fit_transform(df)

    expected = dict(DATA)
    expected["sum_Age_Marks"] = [20.9, 21.8, 19.7, 18.6]
    expected["prod_Age_Marks"] = [18.0, 16.8, 13.3, 10.8]
    expected["mean_Age_Marks"] = [10.45, 10.9, 9.85, 9.3]
    expected["std_Age_Marks"] = [13.505740, 14.283557, 12.940054, 12.303658]
    expected["max_Age_Marks"] = [20.0, 21.0, 19.0, 18.0]
    expected["min_Age_Marks"] = [0.9, 0.8, 0.7, 0.6]

    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_aggregations_with_functions(make_df):
    df = make_df(DATA)
    transformer = MathFeatures(
        variables=["Age", "Marks"], func=[np.sum, np.mean, np.std]
    )
    Xt = transformer.fit_transform(df)

    expected = dict(DATA)
    expected["sum_Age_Marks"] = [20.9, 21.8, 19.7, 18.6]
    expected["mean_Age_Marks"] = [10.45, 10.9, 9.85, 9.3]

    # np.std uses ddof=0 (population std) everywhere now, except pandas < 3,
    # where agg() still routes np.std through pandas' own ddof=1 Series.std().
    # TODO: remove the pandas < 3 branch when dropping older pandas support.
    if make_df is pd.DataFrame and int(pd.__version__.split(".")[0]) < 3:
        expected["std_Age_Marks"] = [13.505740, 14.283557, 12.940054, 12.303658]
    else:
        arr = np.array([DATA["Age"], DATA["Marks"]], dtype=float)
        expected["std_Age_Marks"] = np.std(arr, axis=0).tolist()

    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_user_enters_two_operations(make_df):
    df = make_df(DATA)
    transformer = MathFeatures(variables=["Age", "Marks"], func=["sum", np.mean])
    Xt = transformer.fit_transform(df)

    expected = dict(DATA)
    expected["sum_Age_Marks"] = [20.9, 21.8, 19.7, 18.6]
    expected["mean_Age_Marks"] = [10.45, 10.9, 9.85, 9.3]

    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_new_variable_names(make_df):
    df = make_df(DATA)
    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
        new_variables_names=["sum_of_two_vars", "mean_of_two_vars"],
    )
    Xt = transformer.fit_transform(df)

    expected = dict(DATA)
    expected["sum_of_two_vars"] = [20.9, 21.8, 19.7, 18.6]
    expected["mean_of_two_vars"] = [10.45, 10.9, 9.85, 9.3]

    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_one_mathematical_operation(make_df):
    df = make_df(DATA)
    expected = dict(DATA)
    expected["sum_Age_Marks"] = [20.9, 21.8, 19.7, 18.6]

    transformer = MathFeatures(variables=["Age", "Marks"], func="sum")
    assert_df_equal(transformer.fit_transform(df), expected)

    transformer = MathFeatures(variables=["Age", "Marks"], func=["sum"])
    assert_df_equal(transformer.fit_transform(df), expected)


def test_variable_names_when_df_cols_are_integers(df_numeric_columns):
    # polars requires string column names, so int-named columns are
    # pandas-only - no polars equivalent to parametrize against here.
    transformer = MathFeatures(
        variables=[2, 3], func=["sum", "prod", "mean", "std", "max", "min"]
    )

    X = transformer.fit_transform(df_numeric_columns)

    ref = pd.DataFrame.from_dict(
        {
            0: ["tom", "nick", "krish", "jack"],
            1: ["London", "Manchester", "Liverpool", "Bristol"],
            2: [20, 21, 19, 18],
            3: [0.9, 0.8, 0.7, 0.6],
            4: pd.date_range("2020-02-24", periods=4, freq="min"),
            "sum_2_3": [20.9, 21.8, 19.7, 18.6],
            "prod_2_3": [18.0, 16.8, 13.299999999999999, 10.799999999999999],
            "mean_2_3": [10.45, 10.9, 9.85, 9.3],
            "std_2_3": [
                13.505739520663058,
                14.28355697996826,
                12.94005409571382,
                12.303657992645928,
            ],
            "max_2_3": [20.0, 21.0, 19.0, 18.0],
            "min_2_3": [0.9, 0.8, 0.7, 0.6],
        }
    )

    pd.testing.assert_frame_equal(X, ref)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_error_when_null_values_in_variable(make_df):
    data_na = dict(DATA)
    data_na["Age"] = [20, None, 19, 18]
    df_na = make_df(data_na)

    math_combinator = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
        missing_values="raise",
    )

    with pytest.raises(ValueError):
        math_combinator.fit(df_na)

    math_combinator.fit(make_df(DATA))
    with pytest.raises(ValueError):
        math_combinator.transform(df_na)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_no_error_when_null_values_in_variable(make_df):
    data_na = dict(DATA)
    data_na["Age"] = [20, None, 19, 18]
    df_na = make_df(data_na)

    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
        missing_values="ignore",
    )
    Xt = transformer.fit_transform(df_na)

    expected = dict(data_na)
    expected["sum_Age_Marks"] = [20.9, 0.8, 19.7, 18.6]
    expected["mean_Age_Marks"] = [10.45, 0.8, 9.85, 9.3]

    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_standard_aggregations_match_pandas_with_missing_values(make_df):
    data = {
        "a": [1.0, np.nan, np.nan, 4.0],
        "b": [3.0, 4.0, np.nan, 6.0],
        "c": [5.0, 8.0, np.nan, np.nan],
    }
    functions = ["sum", "mean", "std", "var", "min", "max", "prod", "median"]
    names = [f"result_{function}" for function in functions]

    # pandas' own agg() is the ground truth both backends are checked against.
    X_pd = pd.DataFrame(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        expected_df = X_pd.agg(functions, axis=1)
    expected = {name: expected_df[fn].tolist() for name, fn in zip(names, functions)}

    df = make_df(data)
    transformer = MathFeatures(
        variables=list(data.keys()),
        func=functions,
        new_variables_names=names,
        missing_values="ignore",
    )
    result = transformer.fit_transform(df)

    result_dict = nw.from_native(result, eager_only=True).to_dict(as_series=False)
    for name in names:
        assert result_dict[name] == pytest.approx(expected[name], nan_ok=True)


def test_nullable_dtypes_use_backwards_compatible_aggregation():
    # pandas' nullable "Int64" dtype is pandas-specific - no polars
    # equivalent to parametrize against here.
    X = pd.DataFrame(
        {
            "a": pd.Series([1, pd.NA, 3], dtype="Int64"),
            "b": pd.Series([2, 4, pd.NA], dtype="Int64"),
        }
    )
    functions = ["sum", "mean"]
    names = ["row_sum", "row_mean"]
    expected = X.agg(functions, axis=1)
    expected.columns = names

    transformer = MathFeatures(
        variables=list(X.columns),
        func=functions,
        new_variables_names=names,
        missing_values="ignore",
    )
    result = transformer.fit_transform(X)

    pd.testing.assert_frame_equal(result[names], expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_custom_function_fallback(make_df):
    # max()/min()/sum() are built-ins, so they work identically whether
    # func receives a pandas Series (pandas' agg(axis=1) fallback) or a
    # plain tuple (polars' map_rows fallback) - one callable, one test.
    def peak_to_peak(row):
        return max(row) - min(row)

    df = make_df(DATA)
    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func=peak_to_peak,
        new_variables_names=["age_marks_range"],
    )
    Xt = transformer.fit_transform(df)

    expected = dict(DATA)
    expected["age_marks_range"] = [
        a - m for a, m in zip(DATA["Age"], DATA["Marks"])
    ]
    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_multiple_custom_functions_fallback(make_df):
    def total(row):
        return sum(row)

    def spread(row):
        return max(row) - min(row)

    df = make_df(DATA)
    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func=[total, spread],
        new_variables_names=["total", "spread"],
    )
    Xt = transformer.fit_transform(df)

    expected = dict(DATA)
    expected["total"] = [a + m for a, m in zip(DATA["Age"], DATA["Marks"])]
    expected["spread"] = [a - m for a, m in zip(DATA["Age"], DATA["Marks"])]
    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_uncommon_aggregation_string_only_supported_for_pandas(make_df):
    # a genuine, documented backend asymmetry, not an oversight: pandas'
    # agg() accepts any of its own aggregation strings (even ones outside
    # our NumPy-vectorized table), but polars has no way to resolve an
    # arbitrary pandas-specific string without pandas itself, so it raises
    # instead of silently doing the wrong thing.
    df = make_df(DATA)
    transformer = MathFeatures(variables=["Age", "Marks"], func="sem")

    if make_df is pd.DataFrame:
        Xt = transformer.fit_transform(df)
        expected = dict(DATA)
        expected["sem_Age_Marks"] = [9.55, 10.10, 9.15, 8.70]
        assert_df_equal(Xt, expected)
    else:
        with pytest.raises(NotImplementedError, match="has no NumPy-vectorized"):
            transformer.fit_transform(df)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_drop_original_variables(make_df):
    df = make_df(DATA)
    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
        drop_original=True,
    )
    Xt = transformer.fit_transform(df)

    expected = {
        "Name": DATA["Name"],
        "City": DATA["City"],
        "sum_Age_Marks": [20.9, 21.8, 19.7, 18.6],
        "mean_Age_Marks": [10.45, 10.9, 9.85, 9.3],
    }
    assert_df_equal(Xt, expected)


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("_varnames", [None, ["var1", "var2"]])
@pytest.mark.parametrize("_drop", [True, False])
def test_get_feature_names_out(make_df, _varnames, _drop):
    df = make_df(DATA)
    tr = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
        new_variables_names=_varnames,
        drop_original=_drop,
    )
    Xt = tr.fit_transform(df)
    feat_out = list(nw.from_native(Xt, eager_only=True).columns)
    assert tr.get_feature_names_out(input_features=None) == feat_out


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("_varnames", [None, ["var1", "var2"]])
@pytest.mark.parametrize("_drop", [True, False])
def test_get_feature_names_out_from_pipeline(make_df, _varnames, _drop):
    df = make_df(DATA)
    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
        new_variables_names=_varnames,
        drop_original=_drop,
    )

    pipe = Pipeline([("transformer", transformer)])
    Xt = pipe.fit_transform(df)

    feat_out = list(nw.from_native(Xt, eager_only=True).columns)
    assert pipe.get_feature_names_out(input_features=None) == feat_out


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("_input_features", ["hola", ["Age", "Marks"]])
def test_get_feature_names_out_raises_error_when_wrong_param(make_df, _input_features):
    df = make_df(DATA)
    transformer = MathFeatures(
        variables=["Age", "Marks"],
        func=["sum", "mean"],
    )
    transformer.fit(df)

    with pytest.raises(ValueError):
        transformer.get_feature_names_out(input_features=_input_features)
