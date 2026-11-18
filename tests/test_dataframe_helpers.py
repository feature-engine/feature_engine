import warnings

import narwhals as nw
import numpy as np
import pandas as pd

from feature_engine._dataframe_helpers import add_arrays, select_columns, to_numpy
from tests.backend_helpers import frame_to_dict

DATA = {
    "var_a": [1, 2, 3],
    "var_b": [0.5, 1.5, 2.5],
    "var_c": ["x", "y", "z"],
}


# ------------------------
# to_numpy
# ------------------------


def test_to_numpy_returns_one_array_per_variable(make_df):
    nw_X = nw.from_native(make_df(DATA), eager_only=True)

    arrays = to_numpy(nw_X, ["var_b", "var_a"])

    assert len(arrays) == 2
    assert all(isinstance(values, np.ndarray) for values in arrays)
    np.testing.assert_array_equal(arrays[0], [0.5, 1.5, 2.5])
    np.testing.assert_array_equal(arrays[1], [1, 2, 3])
    assert arrays[0].dtype.kind == "f"
    assert arrays[1].dtype.kind == "i"


def test_to_numpy_returns_nan_for_missing_values(make_df):
    nw_X = nw.from_native(
        make_df({"var_a": [1, None, 3], "var_b": [0.5, 1.5, None]}), eager_only=True
    )

    arrays = to_numpy(nw_X, ["var_a", "var_b"])

    np.testing.assert_array_equal(arrays[0], [1, np.nan, 3])
    np.testing.assert_array_equal(arrays[1], [0.5, 1.5, np.nan])


def test_to_numpy_with_empty_variables(make_df):
    nw_X = nw.from_native(make_df(DATA), eager_only=True)
    assert to_numpy(nw_X, []) == []


def test_to_numpy_with_pandas_integer_column_names():
    nw_X = nw.from_native(pd.DataFrame({0: [1, 2], 1: [3.0, 4.0]}), eager_only=True)

    arrays = to_numpy(nw_X, [1, 0])

    np.testing.assert_array_equal(arrays[0], [3.0, 4.0])
    np.testing.assert_array_equal(arrays[1], [1, 2])


def test_to_numpy_returns_nan_for_pandas_nullable_dtypes():
    X = pd.DataFrame({"var_a": pd.Series([1, pd.NA, 3], dtype="Int64")})

    arrays = to_numpy(nw.from_native(X, eager_only=True), ["var_a"])

    assert arrays[0].dtype.kind == "f"
    np.testing.assert_array_equal(arrays[0], [1, np.nan, 3])


# ------------------------
# add_arrays
# ------------------------


def test_add_arrays_adds_new_columns_at_the_end(make_df):
    nw_X = nw.from_native(make_df(DATA), eager_only=True)

    Xt = add_arrays(
        nw_X, {"new_1": np.array([10, 20, 30]), "new_2": np.array([0.1, 0.2, 0.3])}
    ).to_native()

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_a", "var_b", "var_c", "new_1", "new_2"]
    assert frame_to_dict(Xt) == {
        **DATA,
        "new_1": [10, 20, 30],
        "new_2": [0.1, 0.2, 0.3],
    }


def test_add_arrays_replaces_existing_columns_in_place(make_df):
    nw_X = nw.from_native(make_df(DATA), eager_only=True)

    Xt = add_arrays(
        nw_X, {"var_b": np.array([7.0, 8.0, 9.0]), "new_1": np.array([10, 20, 30])}
    ).to_native()

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_a", "var_b", "var_c", "new_1"]
    assert frame_to_dict(Xt) == {
        "var_a": [1, 2, 3],
        "var_b": [7.0, 8.0, 9.0],
        "var_c": ["x", "y", "z"],
        "new_1": [10, 20, 30],
    }


def test_add_arrays_does_not_modify_the_input(make_df):
    X = make_df(DATA)

    add_arrays(
        nw.from_native(X, eager_only=True),
        {"var_b": np.array([7.0, 8.0, 9.0]), "new_1": np.array([10, 20, 30])},
    )

    assert frame_to_dict(X) == DATA


def test_add_arrays_with_no_arrays(make_df):
    nw_X = nw.from_native(make_df(DATA), eager_only=True)

    Xt = add_arrays(nw_X, {}).to_native()

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == DATA


def test_add_arrays_keeps_pandas_index_and_integer_column_names():
    X = pd.DataFrame({0: [1, 2, 3], 1: [0.5, 1.5, 2.5]}, index=[10, 20, 30])

    Xt = add_arrays(
        nw.from_native(X, eager_only=True),
        {1: np.array([7.0, 8.0, 9.0]), "new_1": np.array([10, 20, 30])},
    ).to_native()

    expected = pd.DataFrame(
        {0: [1, 2, 3], 1: [7.0, 8.0, 9.0], "new_1": [10, 20, 30]},
        index=[10, 20, 30],
    )
    pd.testing.assert_frame_equal(Xt, expected)


def test_add_arrays_does_not_fragment_pandas_dataframe():
    # pandas warns when many columns are inserted one at a time
    X = pd.DataFrame(DATA)
    arrays = {f"new_{i}": np.array([1.0, 2.0, 3.0]) for i in range(150)}

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        Xt = add_arrays(nw.from_native(X, eager_only=True), arrays).to_native()

    assert Xt.shape == (3, 153)


# ------------------------
# select_columns
# ------------------------


def test_select_columns_returns_variables_in_order(make_df):
    nw_X = nw.from_native(make_df(DATA), eager_only=True)

    Xt = select_columns(nw_X, ["var_c", "var_a"]).to_native()

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_c", "var_a"]
    assert frame_to_dict(Xt) == {"var_c": ["x", "y", "z"], "var_a": [1, 2, 3]}


def test_select_columns_keeps_pandas_index_and_integer_column_names():
    X = pd.DataFrame({0: [1, 2, 3], 1: [0.5, 1.5, 2.5]}, index=[10, 20, 30])

    Xt = select_columns(nw.from_native(X, eager_only=True), [1, 0]).to_native()

    expected = pd.DataFrame({1: [0.5, 1.5, 2.5], 0: [1, 2, 3]}, index=[10, 20, 30])
    pd.testing.assert_frame_equal(Xt, expected)
