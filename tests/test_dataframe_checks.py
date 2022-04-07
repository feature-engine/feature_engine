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
    check_y,
)


def test_check_X_returns_df(df_vartypes):
    assert_frame_equal(check_X(df_vartypes), df_vartypes)


def test_check_X_converts_numpy_to_pandas():
    a1D = np.array([1, 2, 3, 4])
    a2D = np.array([[1, 2], [3, 4]])
    a3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    df_2D = pd.DataFrame(a2D, columns=["0", "1"])
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
    s = pd.Series([0,1,2,3,4])
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


def test_check_X_matches_training_df(df_vartypes):
    with pytest.raises(ValueError):
        assert _check_X_matches_training_df(df_vartypes, 4)


def test_contains_na(df_na):
    with pytest.raises(ValueError):
        assert _check_contains_na(df_na, ["Name", "City"])


def test_contains_inf(df_na):
    df_na.fillna(np.inf, inplace=True)
    with pytest.raises(ValueError):
        assert _check_contains_inf(df_na, ["Age", "Marks"])
