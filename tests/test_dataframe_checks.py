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
    _check_optional_contains_na,
    _check_X_matches_training_df,
    check_X,
    check_X_y,
    check_y,
)


def test_check_X_returns_df(df_vartypes):
    assert_frame_equal(check_X(df_vartypes), df_vartypes)


def test_check_X_converts_numpy_to_pandas():
    a1D = np.array([1, 2, 3, 4])
    a2D = np.array([[1, 2], [3, 4]])
    a3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    df_2D = pd.DataFrame(a2D, columns=["x0", "x1"])
    assert_frame_equal(df_2D, check_X(a2D))

    with pytest.raises(ValueError):
        check_X(a3D)
    with pytest.raises(ValueError):
        check_X(a1D)


def test_check_X_raises_error_sparse_matrix():
    sparse_mx = csr_matrix([[5]])
    with pytest.raises(TypeError):
        assert check_X(sparse_mx)


def test_check_X_raises_error_with_complex_data():
    msg = "Complex data not supported"
    rng = np.random.RandomState(0)
    X = rng.uniform(size=10) + 1j * rng.uniform(size=10)
    X = X.reshape(-1, 1)
    with pytest.raises(ValueError, match=msg):
        assert check_X(X)


def test_raises_error_if_empty_df():
    df = pd.DataFrame([])
    with pytest.raises(ValueError):
        check_X(df)


def test_check_y_returns_series():
    s = pd.Series([0, 1, 2, 3, 4])
    assert_series_equal(check_y(s), s)


def test_check_y_returns_dataframe():
    d = pd.DataFrame({"t1": [0, 1, 2, 3, 4], "t2": [5, 6, 7, 8, 9]})
    assert_frame_equal(check_y(d), d)


def test_check_y_converts_np_array():
    a1D = np.array([1, 2, 3, 4])
    s = pd.Series(a1D)
    assert_series_equal(check_y(a1D), s)


def test_check_y_converts_np_array_2D():
    a2D = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(2, 4)
    d = pd.DataFrame(a2D)
    assert_frame_equal(check_y(a2D), d)


def test_check_y_raises_none_error():
    with pytest.raises(ValueError):
        check_y(None)


def test_check_y_raises_nan_error():
    msg = "y contains NaN values."

    # y is series
    s = pd.Series([0, np.nan, 2, 3, 4])
    with pytest.raises(ValueError) as record:
        check_y(s)
    assert str(record.value) == msg

    # y is multioutput
    d = pd.DataFrame(np.array([1, np.nan, 3, 4, 5, 6, np.nan, 8]).reshape(2, 4))
    with pytest.raises(ValueError) as record:
        check_y(d)
    assert str(record.value) == msg


def test_check_y_raises_inf_error():
    msg = "y contains infinity values."

    # y is series
    s = pd.Series([0, np.inf, 2, 3, 4])
    with pytest.raises(ValueError) as record:
        check_y(s)
    assert str(record.value) == msg

    # y is multioutput
    d = pd.DataFrame(np.array([1, np.inf, 3, 4, 5, 6, np.inf, 8]).reshape(2, 4))
    with pytest.raises(ValueError) as record:
        check_y(d)
    assert str(record.value) == msg


def test_check_y_converts_string_to_number():
    s = pd.Series(["0", "1", "2", "3", "4"])
    assert_series_equal(check_y(s, y_numeric=True), s.astype("float"))


def test_check_x_y_returns_pandas_from_pandas(df_vartypes):
    # when s is series
    s = pd.Series([0, 1, 2, 3])
    x, y = check_X_y(df_vartypes, s)
    assert_frame_equal(df_vartypes, x)
    assert_series_equal(s, y)

    # when y is multioutput
    d = pd.DataFrame(np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(4, 2))
    x, y = check_X_y(df_vartypes, d)
    assert_frame_equal(df_vartypes, x)
    assert_frame_equal(d, y)


def test_check_X_y_returns_pandas_from_pandas_with_non_typical_index():
    df = pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212])
    s = pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212])
    x, y = check_X_y(df, s)
    assert_frame_equal(df, x)
    assert_series_equal(s, y)


def test_check_X_y_raises_error_when_pandas_index_dont_match():
    msg = "The indexes of X and y do not match."

    df = pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212])
    s = pd.Series([1, 2, 3, 4], index=[22, 99, 101, 999])
    with pytest.raises(ValueError) as record:
        check_X_y(df, s)
    assert str(record.value) == msg

    # when y is multioutput
    d = pd.DataFrame(
        np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(4, 2), index=[22, 99, 101, 999]
    )
    with pytest.raises(ValueError) as record:
        check_X_y(df, d)
    assert str(record.value) == msg


def test_check_x_y_reassings_index_when_only_one_input_is_pandas():
    # X is dataframe, y is 1D array
    df = pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212])
    s = np.array([1, 2, 3, 4])
    s_exp = pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212])
    x, y = check_X_y(df, s)
    assert_frame_equal(df, x)
    assert_series_equal(s_exp.astype(int), y.astype(int))

    # X is dataframe, y is 2d array
    s = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(4, 2)
    s_exp = pd.DataFrame(s, index=[22, 99, 101, 212])
    x, y = check_X_y(df, s)
    assert_frame_equal(df, x)
    assert_frame_equal(s_exp.astype(int), y.astype(int))

    # X is not a df, y is a series
    df = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
    s = pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212])
    df_exp = pd.DataFrame(df, columns=["x0", "x1"])
    df_exp.index = s.index
    x, y = check_X_y(df, s)
    assert_frame_equal(df_exp, x)
    assert_series_equal(s, y)

    # X is not a df, y is a dataframe
    s = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(4, 2)
    s = pd.DataFrame(s, index=[22, 99, 101, 212])
    df = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
    df_exp = pd.DataFrame(df, columns=["x0", "x1"])
    df_exp.index = s.index
    x, y = check_X_y(df, s)
    assert_frame_equal(df_exp, x)
    assert_frame_equal(s, y)


def test_check_x_y_converts_numpy_to_pandas():
    a2D = np.array([[1, 2], [3, 4], [3, 4], [3, 4]])
    df2D = pd.DataFrame(a2D, columns=["x0", "x1"])

    a1D = np.array([1, 2, 3, 4])
    s1D = pd.Series(a1D)

    # X is df and y is array
    x, y = check_X_y(df2D, a1D)
    assert_frame_equal(df2D, x)
    assert_series_equal(s1D, y)

    # X is array and y is series
    x, y = check_X_y(a2D, s1D)
    assert_frame_equal(df2D, x)
    assert_series_equal(s1D, y)

    # X is df and y is 2d array
    y2D = pd.DataFrame(a2D, columns=[0, 1])
    x, y = check_X_y(df2D, a2D)
    assert_frame_equal(df2D, x)
    assert_frame_equal(y2D, y)

    # X is array and y multioutput df
    x, y = check_X_y(a2D, df2D)
    assert_frame_equal(df2D, x)
    assert_frame_equal(df2D, y)


def test_check_x_y_raises_error_when_inconsistent_length(df_vartypes):
    s = pd.Series([0, 1, 2, 3, 5])
    with pytest.raises(ValueError):
        check_X_y(df_vartypes, s)


def test_check_X_matches_training_df(df_vartypes):
    with pytest.raises(ValueError):
        assert _check_X_matches_training_df(df_vartypes, 4)


def test_contains_na(df_na):
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )

    with pytest.raises(ValueError) as record:
        assert _check_contains_na(df_na, ["Name", "City"])
    assert str(record.value) == msg


def test_optional_contains_na(df_na):
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )

    with pytest.raises(ValueError) as record:
        assert _check_optional_contains_na(df_na, ["Name", "City"])
    assert str(record.value) == msg


def test_contains_inf_raises_on_inf():
    msg = (
        "Some of the variables to transform contain inf values. Check and "
        "remove those before using this transformer."
    )
    df = pd.DataFrame({"A": [1.1, np.inf, 3.3]})
    with pytest.raises(ValueError, match=msg):
        _check_contains_inf(df, ["A"])


def test_contains_inf_passes_without_inf():
    df = pd.DataFrame({"A": [1.1, 2.2, 3.3]})
    assert _check_contains_inf(df, ["A"]) is None


def test_check_X_raises_error_on_duplicated_column_names():
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


def test_check_X_errors():
    # Test scalar array error
    with pytest.raises(ValueError) as record:
        check_X(np.array(1))
    assert record.match("Expected 2D array, got scalar array instead")

    # Test 1D array error
    with pytest.raises(ValueError) as record:
        check_X(np.array([1, 2, 3]))
    assert record.match("Expected 2D array, got 1D array instead")

    # Test incorrect type error
    with pytest.raises(TypeError) as record:
        check_X("not a dataframe")
    assert record.match("X must be a numpy array or a dataframe from a library")


# ------------------------
# polars support
# ------------------------


def test_check_X_accepts_polars_and_returns_a_copy():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    X = check_X(df)
    assert isinstance(X, pl.DataFrame)
    pl_assert_frame_equal(X, df)
    assert X is not df


def test_check_X_polars_raises_error_if_empty():
    with pytest.raises(ValueError):
        check_X(pl.DataFrame({"a": []}))


def test_check_y_accepts_polars_series_and_returns_a_copy():
    s = pl.Series("target", [0, 1, 2, 3, 4])
    y = check_y(s)
    assert isinstance(y, pl.Series)
    pl_assert_series_equal(y, s)
    assert y is not s


def test_check_y_polars_raises_nan_error():
    s = pl.Series("target", [0.0, None, 2.0])
    with pytest.raises(ValueError) as record:
        check_y(s)
    assert str(record.value) == "y contains NaN values."


def test_check_y_polars_raises_inf_error():
    s = pl.Series("target", [0.0, float("inf"), 2.0])
    with pytest.raises(ValueError) as record:
        check_y(s)
    assert str(record.value) == "y contains infinity values."


def test_check_y_polars_converts_string_to_number():
    s = pl.Series("target", ["0", "1", "2"])
    y = check_y(s, y_numeric=True)
    assert y.dtype.is_numeric()
    pl_assert_series_equal(y, pl.Series("target", [0.0, 1.0, 2.0]))


def test_check_x_y_polars_returns_polars():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    s = pl.Series("target", [0, 1, 2])
    X, y = check_X_y(df, s)
    assert isinstance(X, pl.DataFrame) and isinstance(y, pl.Series)
    pl_assert_frame_equal(X, df)
    pl_assert_series_equal(y, s)


def test_check_x_y_polars_raises_error_when_inconsistent_length():
    df = pl.DataFrame({"a": [1, 2, 3]})
    s = pl.Series([0, 1])
    with pytest.raises(ValueError):
        check_X_y(df, s)


def test_check_X_matches_training_df_with_polars():
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError):
        _check_X_matches_training_df(df, 3)


def test_contains_na_with_polars():
    df = pl.DataFrame({"Name": ["tom", None], "City": ["London", "Manchester"]})
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError) as record:
        _check_contains_na(df, ["Name", "City"])
    assert str(record.value) == msg


def test_optional_contains_na_with_polars():
    df = pl.DataFrame({"Name": ["tom", None], "City": ["London", "Manchester"]})
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    with pytest.raises(ValueError) as record:
        _check_optional_contains_na(df, ["Name", "City"])
    assert str(record.value) == msg


def test_contains_inf_with_polars():
    msg = (
        "Some of the variables to transform contain inf values. Check and "
        "remove those before using this transformer."
    )
    df = pl.DataFrame({"A": [1.1, float("inf"), 3.3]})
    with pytest.raises(ValueError, match=msg):
        _check_contains_inf(df, ["A"])

    df_ok = pl.DataFrame({"A": [1.1, 2.2, 3.3]})
    assert _check_contains_inf(df_ok, ["A"]) is None
