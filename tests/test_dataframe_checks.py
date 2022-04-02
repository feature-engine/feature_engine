import pytest
import contextlib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from feature_engine.dataframe_checks import (
    _check_contains_na,
    _check_input_matches_training_df,
    _is_dataframe,
    _check_X_y
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
    "X_in, y_in, expected_1, expected_2, exception_type, exception_match",
    [
        # * If both parameters are numpy objects,
        # they are converted to pandas objects.
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T,
            np.array([1, 2, 3, 4]),
            pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}),
            pd.Series([1, 2, 3, 4]),
            None,
            None
        ),

        # * If one parameter is a numpy object and the
        # other is a pandas object, the former will be
        # converted to a pandas object, with the indexes
        # of the latter.
        (
            pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212]),
            np.array([1, 2, 3, 4]),
            pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212]),
            pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212]),
            None,
            None
        ),
        (
            np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).T,
            pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212]),
            pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212]),
            pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212]),
            None,
            None
        ),

        # * If both parameters are pandas objects, and their
        # indexes are inconsistent, an exception is raised
        # (i.e.this is the caller's error.)
        (
            pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212]),
            pd.Series([1, 2, 3, 4], index=[22, 99, 101, 999]),
            None,
            None,
            ValueError,
            ".*Index.*"
        ),

        # * If both parameters are pandas objects and their indexes match, they are
        # returned unchanged.
        (
            pd.DataFrame({"0": [1, 2, 3, 4], "1": [5, 6, 7, 8]}, index=[22, 99, 101, 212]),
            pd.Series([1, 2, 3, 4], index=[22, 99, 101, 212]),
            None,
            None,
            None,
            None
        ),
    ]
)
def test_check_X_y(X_in, y_in, expected_1, expected_2, exception_type, exception_match):
    with (
        contextlib.nullcontext() if not exception_type
        else pytest.raises(exception_type, match=exception_match)
    ):
        # Execute - can throw here (non-null exception_type will expect exception)
        X_out, y_out = _check_X_y(X_in, y_in)

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