import numpy as np
import pandas as pd
import polars as pl
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from polars.testing import assert_frame_equal as pl_assert_frame_equal
from polars.testing import assert_series_equal as pl_assert_series_equal
from scipy.sparse import csr_matrix

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
    _check_X_matches_training_df,
    check_X,
    check_X_y,
    check_y,
)

# ------------------------
# test check_X
# ------------------------


@pytest.mark.parametrize(
    "make_df, assert_equal_fn",
    [(pd.DataFrame, assert_frame_equal), (pl.DataFrame, pl_assert_frame_equal)],
)
def test_check_X_returns_df_unchanged(make_df, assert_equal_fn):
    df = make_df({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    X = check_X(df)
    assert isinstance(X, type(df))
    assert_equal_fn(X, df)


@pytest.mark.parametrize(
    "make_df, assert_equal_fn",
    [(pd.DataFrame, assert_frame_equal), (pl.DataFrame, pl_assert_frame_equal)],
)
def test_check_X_returns_df_with_mixed_dtypes(make_df, assert_equal_fn):
    data = {
        "Name": ["tom", "nick", "krish", "jack"],
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
        "dob": pd.date_range("2020-02-24", periods=4, freq="min"),
    }
    df = make_df(data)
    assert_equal_fn(check_X(df), df)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame([]),
        pd.DataFrame({"a": []}),
        pl.DataFrame({"a": []}),
    ],
)
def test_raises_error_if_empty_df(df):
    with pytest.raises(ValueError):
        check_X(df)


def test_check_X_raises_error_if_0_columns():
    # A dataframe with rows but no columns is not caught by `is_empty()`, which
    # only looks at the row count, so it needs its own explicit check. Polars has
    # no representation for "rows with 0 columns", so this case is pandas-only.
    df = pd.DataFrame(index=range(3))
    assert df.shape == (3, 0)
    with pytest.raises(ValueError):
        check_X(df)


def test_check_X_raises_error_on_duplicated_column_names():
    # only relevant for pandas
    df = pd.DataFrame(
        {
            "Name": ["tom", "nick", "krish", "jack"],
            "City": ["London", "Manchester", "Liverpool", "Bristol"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, 0.8, 0.7, 0.6],
        }
    )
    df.columns = ["var_A", "var_A", "var_B", "var_C"]
    with pytest.raises(ValueError) as err_txt:
        check_X(df)
    assert err_txt.match("Expected unique column names")


@pytest.mark.parametrize(
    "X",
    [
        np.array([[1, 2], [3, 4]]),
        np.array([1, 2, 3]),
        np.array(1),
        [1, 2, 3],
        {"a": [1, 2, 3]},
        "not a dataframe",
        None,
        csr_matrix([[1, 2], [3, 4]]),
    ],
)
def test_check_X_raises_error_on_non_dataframe_input(X):
    with pytest.raises(TypeError) as record:
        check_X(X)
    assert record.match("X must be a dataframe from a library supported by narwhals")


# ------------------------
# test check_y
# ------------------------

# --- series input ---


@pytest.mark.parametrize(
    "make_series, assert_equal_fn",
    [(pd.Series, assert_series_equal), (pl.Series, pl_assert_series_equal)],
)
def test_check_y_series_returns_values_unchanged(make_series, assert_equal_fn):
    s = make_series([0, 1, 2, 3, 4])
    assert_equal_fn(check_y(s), s)


@pytest.mark.parametrize(
    "make_series",
    [pd.Series, pl.Series],
)
def test_check_y_series_raises_nan_error(make_series):
    s = make_series([0.0, None, 2.0])
    with pytest.raises(ValueError, match="y contains NaN values."):
        check_y(s)


@pytest.mark.parametrize(
    "make_series",
    [pd.Series, pl.Series],
)
def test_check_y_series_raises_inf_error(make_series):
    s = make_series([0.0, float("inf"), 2.0])
    with pytest.raises(ValueError, match="y contains infinity values."):
        check_y(s)


@pytest.mark.parametrize(
    "make_series, assert_equal_fn",
    [(pd.Series, assert_series_equal), (pl.Series, pl_assert_series_equal)],
)
def test_check_y_series_converts_string_to_number_when_y_numeric(
    make_series, assert_equal_fn
):
    s = make_series(["0", "1", "2"])
    y = check_y(s, y_numeric=True)
    expected = make_series([0.0, 1.0, 2.0])
    assert_equal_fn(y, expected)


@pytest.mark.parametrize(
    "make_series, assert_equal_fn",
    [(pd.Series, assert_series_equal), (pl.Series, pl_assert_series_equal)],
)
def test_check_y_series_leaves_non_numeric_unchanged_by_default(
    make_series, assert_equal_fn
):
    # y_numeric defaults to False: a non-numeric series (e.g. classification
    # labels) should be returned as-is, without being cast to float.
    s = make_series(["a", "b", "c"])
    assert_equal_fn(check_y(s), s)


# --- dataframe (multioutput) input ---


@pytest.mark.parametrize(
    "make_df, assert_equal_fn",
    [(pd.DataFrame, assert_frame_equal), (pl.DataFrame, pl_assert_frame_equal)],
)
def test_check_y_dataframe_returns_values_unchanged(make_df, assert_equal_fn):
    d = make_df({"t1": [0, 1, 2, 3, 4], "t2": [5, 6, 7, 8, 9]})
    assert_equal_fn(check_y(d), d)


@pytest.mark.parametrize(
    "make_df",
    [pd.DataFrame, pl.DataFrame],
)
def test_check_y_dataframe_raises_nan_error(make_df):
    d = make_df({"t1": [0.0, None, 2.0], "t2": [5.0, 6.0, 7.0]})
    with pytest.raises(ValueError, match="y contains NaN values."):
        check_y(d)


@pytest.mark.parametrize(
    "make_df",
    [pd.DataFrame, pl.DataFrame],
)
def test_check_y_dataframe_raises_inf_error(make_df):
    d = make_df({"t1": [0.0, 0.4, 2.0], "t2": [5.0, float("inf"), 7.0]})
    with pytest.raises(ValueError, match="y contains infinity values."):
        check_y(d)


# --- array-like input ---


@pytest.mark.parametrize(
    "a",
    [
        np.array([1, 2, 3, 4]),
        np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 4),
    ],
)
def test_check_y_array_returns_unchanged(a):
    y = check_y(a)
    assert isinstance(y, np.ndarray)
    np.testing.assert_array_equal(a, y)


def test_check_y_raises_none_error():
    with pytest.raises(ValueError):
        check_y(None)


# ------------------------
# test check_X_y
# ------------------------


@pytest.mark.parametrize(
    "make_df, assert_frame_fn, make_series, assert_series_fn",
    [
        (pd.DataFrame, assert_frame_equal, pd.Series, assert_series_equal),
        (pl.DataFrame, pl_assert_frame_equal, pl.Series, pl_assert_series_equal),
    ],
)
def test_check_X_y_returns_df_and_series_unchanged(
    make_df, assert_frame_fn, make_series, assert_series_fn
):
    df = make_df({"a": [1, 2, 3], "b": [4, 5, 6]})
    s = make_series([0, 1, 2])
    X, y = check_X_y(df, s)
    assert isinstance(X, type(df)) and isinstance(y, type(s))
    assert_frame_fn(X, df)
    assert_series_fn(y, s)


@pytest.mark.parametrize(
    "make_df, assert_frame_fn",
    [(pd.DataFrame, assert_frame_equal), (pl.DataFrame, pl_assert_frame_equal)],
)
def test_check_X_y_returns_df_and_multioutput_y_unchanged(make_df, assert_frame_fn):
    df = make_df({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    d = make_df({"t1": [1, 2, 3, 4], "t2": [5, 6, 7, 8]})
    X, y = check_X_y(df, d)
    assert_frame_fn(X, df)
    assert_frame_fn(y, d)


def test_check_X_y_returns_pandas_with_non_typical_index():
    # only relevant for pandas: polars has no index to reconcile
    df = pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212])
    s = pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212])
    x, y = check_X_y(df, s)
    assert_frame_equal(df, x)
    assert_series_equal(s, y)


def test_check_X_y_raises_error_when_pandas_index_dont_match():
    # only relevant for pandas: polars has no index to reconcile
    msg = "The indexes of X and y do not match."

    df = pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212])
    s = pd.Series([1, 2, 3, 4], index=[22, 99, 101, 999])
    with pytest.raises(ValueError, match=msg):
        check_X_y(df, s)

    # when y is multioutput
    d = pd.DataFrame(
        np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(4, 2), index=[22, 99, 101, 999]
    )
    with pytest.raises(ValueError, match=msg):
        check_X_y(df, d)


@pytest.mark.parametrize(
    "make_df, make_series",
    [(pd.DataFrame, pd.Series), (pl.DataFrame, pl.Series)],
)
def test_check_x_y_raises_error_when_inconsistent_length(make_df, make_series):
    df = make_df({"a": [1, 2, 3]})
    s = make_series([0, 1])
    with pytest.raises(ValueError):
        check_X_y(df, s)


# -----------------------------------
# test _check_X_matches_training_df
# -----------------------------------


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_check_X_matches_training_df_passes_when_columns_match(make_df):
    df = make_df({"a": [1, 2], "b": [3, 4]})
    assert _check_X_matches_training_df(df, 2) is None


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_check_X_matches_training_df_raises_error_when_columns_dont_match(make_df):
    msg = "The number of columns in this dataset is different from"
    df = make_df({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match=msg):
        _check_X_matches_training_df(df, 3)


# -------------------------
# test _check_contains_na
# -------------------------


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_contains_na_raises_when_nan(make_df):
    msg1 = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    msg2 = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )

    df = make_df({"Name": ["tom", None], "City": ["London", "Manchester"]})
    with pytest.raises(ValueError, match=msg1):
        _check_contains_na(df, ["Name", "City"])

    with pytest.raises(ValueError, match=msg2):
        _check_contains_na(df, ["Name", "City"], error_msg="other")


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_contains_na_passes_when_no_nan(make_df):
    df = make_df({"Name": ["tom", "nick"], "City": ["London", "Manchester"]})
    assert _check_contains_na(df, ["Name", "City"]) is None


# --------------------------
# test _check_contains_inf
# --------------------------


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_contains_inf_raises_on_inf(make_df):
    msg = (
        "Some of the variables to transform contain inf values. Check and "
        "remove those before using this transformer."
    )
    df = make_df({"A": [1.1, np.inf, 3.3]})
    with pytest.raises(ValueError, match=msg):
        _check_contains_inf(df, ["A"])


@pytest.mark.parametrize("make_df", [pd.DataFrame, pl.DataFrame])
def test_contains_inf_passes_without_inf(make_df):
    df = make_df({"A": [1.1, 2.2, 3.3]})
    assert _check_contains_inf(df, ["A"]) is None
