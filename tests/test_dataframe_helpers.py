import warnings

import narwhals as nw
import numpy as np
import pandas as pd

from feature_engine.dataframe_checks import check_X
from feature_engine.dataframe_helpers import (
    add_arrays_to_df,
    drop_variables_from_df,
    subset_df,
    to_numpy,
)
from tests.backend_helpers import frame_to_dict

DATA = {
    "var_a": [1, 2, 3],
    "var_b": [0.5, 1.5, 2.5],
    "var_c": ["x", "y", "z"],
}


def to_native(X):
    return nw.to_native(X, pass_through=True)


# ------------------------
# to_numpy
# ------------------------


def test_to_numpy_returns_one_array_per_variable(make_df):
    X = check_X(make_df(DATA))

    arrays = to_numpy(X, ["var_b", "var_a"])

    assert len(arrays) == 2
    assert all(isinstance(values, np.ndarray) for values in arrays)
    np.testing.assert_array_equal(arrays[0], [0.5, 1.5, 2.5])
    np.testing.assert_array_equal(arrays[1], [1, 2, 3])
    assert arrays[0].dtype.kind == "f"
    assert arrays[1].dtype.kind == "i"


def test_to_numpy_returns_nan_for_missing_values(make_df):
    X = check_X(make_df({"var_a": [1, None, 3], "var_b": [0.5, 1.5, None]}))

    arrays = to_numpy(X, ["var_a", "var_b"])

    np.testing.assert_array_equal(arrays[0], [1, np.nan, 3])
    np.testing.assert_array_equal(arrays[1], [0.5, 1.5, np.nan])


def test_to_numpy_with_empty_variables(make_df):
    assert to_numpy(check_X(make_df(DATA)), []) == []


def test_to_numpy_with_pandas_integer_column_names():
    X = check_X(pd.DataFrame({0: [1, 2], 1: [3.0, 4.0]}))

    arrays = to_numpy(X, [1, 0])

    np.testing.assert_array_equal(arrays[0], [3.0, 4.0])
    np.testing.assert_array_equal(arrays[1], [1, 2])


def test_to_numpy_with_pandas_nullable_dtypes():
    X = check_X(
        pd.DataFrame(
            {
                "var_a": pd.Series([1, pd.NA, 3], dtype="Int64"),
                "var_b": pd.Series([1, 2, 3], dtype="Int64"),
            }
        )
    )

    arrays = to_numpy(X, ["var_a", "var_b"])

    assert arrays[0].dtype.kind == "f"
    np.testing.assert_array_equal(arrays[0], [1, np.nan, 3])
    assert arrays[1].dtype.kind == "i"
    np.testing.assert_array_equal(arrays[1], [1, 2, 3])


# ------------------------
# add_arrays_to_df
# ------------------------


def test_add_arrays_to_df_adds_new_columns_at_the_end(make_df):
    X = check_X(make_df(DATA))

    Xt = to_native(
        add_arrays_to_df(
            X, {"new_1": np.array([10, 20, 30]), "new_2": np.array([0.1, 0.2, 0.3])}
        )
    )

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_a", "var_b", "var_c", "new_1", "new_2"]
    assert frame_to_dict(Xt) == {
        **DATA,
        "new_1": [10, 20, 30],
        "new_2": [0.1, 0.2, 0.3],
    }


def test_add_arrays_to_df_replaces_existing_columns_in_place(make_df):
    X = check_X(make_df(DATA))

    Xt = to_native(
        add_arrays_to_df(
            X, {"var_b": np.array([7.0, 8.0, 9.0]), "new_1": np.array([10, 20, 30])}
        )
    )

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_a", "var_b", "var_c", "new_1"]
    assert frame_to_dict(Xt) == {
        "var_a": [1, 2, 3],
        "var_b": [7.0, 8.0, 9.0],
        "var_c": ["x", "y", "z"],
        "new_1": [10, 20, 30],
    }


def test_add_arrays_to_df_does_not_modify_the_input(make_df):
    X = make_df(DATA)

    add_arrays_to_df(
        check_X(X),
        {"var_b": np.array([7.0, 8.0, 9.0]), "new_1": np.array([10, 20, 30])},
    )

    assert frame_to_dict(X) == DATA


def test_add_arrays_to_df_with_no_arrays(make_df):
    Xt = to_native(add_arrays_to_df(check_X(make_df(DATA)), {}))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == DATA


def test_add_arrays_to_df_keeps_pandas_index_and_integer_column_names():
    X = pd.DataFrame({0: [1, 2, 3], 1: [0.5, 1.5, 2.5]}, index=[10, 10, 30])

    Xt = add_arrays_to_df(
        check_X(X), {1: np.array([7.0, 8.0, 9.0]), "new_1": np.array([10, 20, 30])}
    )

    expected = pd.DataFrame(
        {0: [1, 2, 3], 1: [7.0, 8.0, 9.0], "new_1": [10, 20, 30]},
        index=[10, 10, 30],
    )
    pd.testing.assert_frame_equal(Xt, expected)


def test_add_arrays_to_df_does_not_fragment_pandas_dataframe():
    # pandas warns when many columns are inserted one at a time
    X = pd.DataFrame(DATA)
    arrays = {f"new_{i}": np.array([1.0, 2.0, 3.0]) for i in range(150)}

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        Xt = add_arrays_to_df(check_X(X), arrays)

    assert Xt.shape == (3, 153)


# ------------------------
# subset_df
# ------------------------


def test_subset_df_returns_variables_in_order(make_df):
    Xt = to_native(subset_df(check_X(make_df(DATA)), ["var_c", "var_a"]))

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_c", "var_a"]
    assert frame_to_dict(Xt) == {"var_c": ["x", "y", "z"], "var_a": [1, 2, 3]}


def test_subset_df_keeps_pandas_index_and_integer_column_names():
    X = pd.DataFrame({0: [1, 2, 3], 1: [0.5, 1.5, 2.5]}, index=[10, 20, 30])

    Xt = subset_df(check_X(X), [1, 0])

    expected = pd.DataFrame({1: [0.5, 1.5, 2.5], 0: [1, 2, 3]}, index=[10, 20, 30])
    pd.testing.assert_frame_equal(Xt, expected)


# ------------------------
# drop_variables_from_df
# ------------------------


def test_drop_variables_from_df_removes_the_variables(make_df):
    X = make_df(DATA)

    Xt = to_native(drop_variables_from_df(check_X(X), ["var_a", "var_c"]))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"var_b": [0.5, 1.5, 2.5]}
    assert frame_to_dict(X) == DATA


def test_drop_variables_from_df_keeps_pandas_index_and_integer_column_names():
    X = pd.DataFrame({0: [1, 2, 3], 1: [0.5, 1.5, 2.5]}, index=[10, 20, 30])

    Xt = drop_variables_from_df(check_X(X), [0])

    expected = pd.DataFrame({1: [0.5, 1.5, 2.5]}, index=[10, 20, 30])
    pd.testing.assert_frame_equal(Xt, expected)
