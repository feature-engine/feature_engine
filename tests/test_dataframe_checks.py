import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_input_matches_training_df,
    _check_pd_X_y,
    _is_dataframe,
)


def test_is_dataframe(df_vartypes):
    assert_frame_equal(_is_dataframe(df_vartypes), df_vartypes)
    with pytest.raises(TypeError):
        assert _is_dataframe([1, 2, 4])


def test_check_input_matches_training_df(df_vartypes):
    with pytest.raises(ValueError):
        assert _check_input_matches_training_df(df_vartypes, 4)


def test_contains_na(df_na):
    with pytest.raises(ValueError):
        assert _check_contains_na(df_na, ["Name", "City"])


@pytest.mark.parametrize(
    "X_in, y_in, expected_1, expected_2",
    [
        # * If both parameters are numpy objects,
        # they are converted to pandas objects.
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T,
            np.array([1, 2, 3, 4]),
            pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}),
            pd.Series([1, 2, 3, 4]),
        ),
    ],
)
def test_check_pd_X_y_both_numpy(X_in, y_in, expected_1, expected_2):
    # Execute
    X_out, y_out = _check_pd_X_y(X_in, y_in)

    # Test X output
    if expected_1 is None:
        assert X_out is X_in
    elif isinstance(expected_1, pd.DataFrame):
        assert_frame_equal(X_out, expected_1)
    elif isinstance(expected_1, (np.generic, np.ndarray)):
        assert all(X_out == expected_1)

    # Test y output
    if expected_2 is None:
        assert y_out is y_in
    elif isinstance(expected_2, pd.Series):
        assert_series_equal(y_out, expected_2)
    elif isinstance(expected_2, (np.generic, np.ndarray)):
        assert all(y_out == expected_2)


@pytest.mark.parametrize(
    "X_in, y_in, expected_1, expected_2",
    [
        # * If both parameters are pandas objects and their indexes match, they are
        # copied and returned.
        (
            pd.DataFrame(
                {"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212]
            ),
            pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212]),
            pd.DataFrame(
                {"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212]
            ),
            pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212]),
        ),
    ],
)
def test_check_pd_X_y_both_pandas(X_in, y_in, expected_1, expected_2):
    # Execute
    X_out, y_out = _check_pd_X_y(X_in, y_in)

    # Test X output
    if isinstance(expected_1, pd.DataFrame):
        assert_frame_equal(X_out, expected_1)
        assert X_out is not expected_1  # make sure copied
    elif isinstance(expected_1, (np.generic, np.ndarray)):
        assert all(X_out == expected_1)

    # Test y output
    if isinstance(expected_2, pd.Series):
        assert_series_equal(y_out, expected_2)
        assert y_out is not expected_2  # make sure copied
    elif isinstance(expected_2, (np.generic, np.ndarray)):
        assert all(y_out == expected_2)


@pytest.mark.parametrize(
    "X_in, y_in, expected_1, expected_2",
    [
        # * If one parameter is a numpy object and the
        # other is a pandas object, the former will be
        # converted to a pandas object, with the indexes
        # of the latter.
        (
            pd.DataFrame(
                {"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212]
            ),
            np.array([1, 2, 3, 4]),
            pd.DataFrame(
                {"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212]
            ),
            pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212]),
        ),
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T,
            pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212]),
            pd.DataFrame(
                {"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212]
            ),
            pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212]),
        ),
    ],
)
def test_check_pd_X_y_np_to_pd(X_in, y_in, expected_1, expected_2):
    # Execute
    X_out, y_out = _check_pd_X_y(X_in, y_in)

    # Test X output
    if expected_1 is None:
        assert X_out is X_in
    elif isinstance(expected_1, pd.DataFrame):
        assert_frame_equal(X_out, expected_1)
    elif isinstance(expected_1, (np.generic, np.ndarray)):
        assert all(X_out == expected_1)

    # Test y output
    if expected_2 is None:
        assert y_out is y_in
    elif isinstance(expected_2, pd.Series):
        assert_series_equal(y_out, expected_2)
    elif isinstance(expected_2, (np.generic, np.ndarray)):
        assert all(y_out == expected_2)


@pytest.mark.parametrize(
    "X_in, y_in, exception_type, exception_match",
    [
        # * If both parameters are pandas objects, and their
        # indexes are inconsistent, an exception is raised
        # (i.e.this is the caller's error.)
        (
            pd.DataFrame(
                {"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212]
            ),
            pd.Series([1, 2, 3, 4], index=[22, 99, 101, 999]),
            ValueError,
            ".*Index.*",
        ),

        # Show that incompatible dimensions causes same error
        (
                pd.DataFrame(
                    {"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212]
                ),
                pd.Series([1, 2, 3], index=[22, 99, 101]),
                ValueError,
                ".*Lengths.*",
        ),
    ],
)
def test_check_pd_X_y_errors(X_in, y_in, exception_type, exception_match):
    with (pytest.raises(exception_type, match=exception_match)):
        # Execute - can throw here (non-null exception_type will expect exception)
        X_out, y_out = _check_pd_X_y(X_in, y_in)
