from typing import Any
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal


from feature_engine.numpy_to_pandas import (
    _is_numpy,
    _numpy_to_series,
    _numpy_to_dataframe
)


@pytest.mark.parametrize(
    "obj, expected",
    [
        (np.array([1, 2, 3, 4]), True),
        (pd.Series([1, 2, 3, 4]), False),
        ("something", False)
    ]
)
def test_is_numpy(obj: Any, expected: bool):
    assert _is_numpy(obj) == expected


def test_numpy_to_dataframe():
    np_array: np.ndarray = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected: pd.DataFrame = pd.DataFrame(
        {"0": [1, 4, 7], "1": [2, 5, 8], "2": [3, 6, 9]}
    )
    assert_frame_equal(_numpy_to_dataframe(np_array), expected)

    expected.index = ["a", "b", "c"]
    assert_frame_equal(_numpy_to_dataframe(np_array, index=["a", "b", "c"]), expected)


def test_numpy_to_series():
    np_array: np.ndarray = np.array([1, 2, 3])
    expected: pd.Series = pd.Series([1, 2, 3])
    assert_series_equal(_numpy_to_series(np_array), expected)

    expected.index = ["a", "b", "c"]
    assert_series_equal(_numpy_to_series(np_array, index=["a", "b", "c"]), expected)
