import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from scipy.sparse import csr_matrix

from feature_engine.dataframe_checks import (
    _check_contains_inf,
    _check_contains_na,
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


def test_raises_error_if_empty_df():
    df = pd.DataFrame([])
    with pytest.raises(ValueError):
        check_X(df)


def test_check_y_returns_series():
    s = pd.Series([0, 1, 2, 3, 4])
    assert_series_equal(check_y(s), s)


def test_check_y_converts_np_array():
    a1D = np.array([1, 2, 3, 4])
    s = pd.Series(a1D)
    assert_series_equal(check_y(a1D), s)


def test_check_y_raises_none_error():
    with pytest.raises(ValueError):
        check_y(None)


def test_check_y_raises_nan_error():
    s = pd.Series([0, np.nan, 2, 3, 4])
    with pytest.raises(ValueError):
        check_y(s)


def test_check_y_raises_inf_error():
    s = pd.Series([0, np.inf, 2, 3, 4])
    with pytest.raises(ValueError):
        check_y(s)


def test_check_y_converts_string_to_number():
    s = pd.Series(["0", "1", "2", "3", "4"])
    assert_series_equal(check_y(s, y_numeric=True), s.astype("float"))


def test_check_x_y_returns_pandas_from_pandas(df_vartypes):
    s = pd.Series([0, 1, 2, 3])
    x, y = check_X_y(df_vartypes, s)
    assert_frame_equal(df_vartypes, x)
    assert_series_equal(s, y)


def test_check_X_y_returns_pandas_from_pandas_with_non_typical_index():
    df = pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212])
    s = pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212])
    x, y = check_X_y(df, s)
    assert_frame_equal(df, x)
    assert_series_equal(s, y)


def test_check_X_y_raises_error_when_pandas_index_dont_match():
    df = pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212])
    s = pd.Series([1, 2, 3, 4], index=[22, 99, 101, 999])
    with pytest.raises(ValueError):
        check_X_y(df, s)


def test_check_x_y_reassings_index_when_only_one_input_is_pandas():
    # case 1: X is dataframe, y is something else
    df = pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212])
    s = np.array([1, 2, 3, 4])
    s_exp = pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212])
    x, y = check_X_y(df, s)
    assert_frame_equal(df, x)
    assert_series_equal(s_exp.astype(int), y.astype(int))

    # case 2: X is not a df, y is a series
    df = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T
    s = pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212])
    df_exp = pd.DataFrame(df, columns=["x0", "x1"])
    df_exp.index = s.index

    x, y = check_X_y(df, s)
    assert_frame_equal(df_exp, x)
    assert_series_equal(s, y)


def test_check_x_y_converts_numpy_to_pandas():
    a2D = np.array([[1, 2], [3, 4], [3, 4], [3, 4]])
    df_2D = pd.DataFrame(a2D, columns=["x0", "x1"])

    a1D = np.array([1, 2, 3, 4])
    s = pd.Series(a1D)

    x, y = check_X_y(df_2D, s)
    assert_frame_equal(df_2D, x)
    assert_series_equal(s, y)


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

    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )

    with pytest.raises(ValueError) as record:
        assert _check_contains_na(df_na, ["Name", "City"], switch_param=True)
    assert str(record.value) == msg


def test_contains_inf(df_na):
    df_na.fillna(np.inf, inplace=True)
    with pytest.raises(ValueError):
        assert _check_contains_inf(df_na, ["Age", "Marks"])
