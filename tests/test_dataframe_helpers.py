import re
import warnings

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

from feature_engine.dataframe_checks import check_X
from feature_engine.dataframe_helpers import (
    add_category_to_df,
    add_columns_to_df,
    aggregate_columns,
    drop_variables_from_df,
    fill_missing_in_df,
    filter_rows,
    map_categories_in_df,
    missing_values_count,
    missing_values_mask,
    replace_columns_in_df,
    rows_with_missing_values,
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


# ------------------------
# add_columns_to_df
# ------------------------


def test_add_columns_to_df_adds_new_columns_at_the_end(make_df):
    X = check_X(make_df(DATA))

    Xt = to_native(
        add_columns_to_df(
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


def test_add_columns_to_df_does_not_modify_the_input(make_df):
    X = make_df(DATA)

    add_columns_to_df(check_X(X), {"new_1": np.array([10, 20, 30])})

    assert frame_to_dict(X) == DATA


def test_add_columns_to_df_with_no_arrays(make_df):
    Xt = to_native(add_columns_to_df(check_X(make_df(DATA)), {}))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == DATA


def test_add_columns_to_df_keeps_pandas_index_and_integer_column_names():
    X = pd.DataFrame({0: [1, 2, 3], 1: [0.5, 1.5, 2.5]}, index=[10, 10, 30])

    Xt = add_columns_to_df(
        check_X(X), {2: np.array([7.0, 8.0, 9.0]), "new_1": np.array([10, 20, 30])}
    )

    expected = pd.DataFrame(
        {0: [1, 2, 3], 1: [0.5, 1.5, 2.5], 2: [7.0, 8.0, 9.0], "new_1": [10, 20, 30]},
        index=[10, 10, 30],
    )
    pd.testing.assert_frame_equal(Xt, expected)


def test_add_columns_to_df_does_not_fragment_pandas_dataframe():
    # pandas warns when many columns are inserted one at a time
    X = pd.DataFrame(DATA)
    arrays = {f"new_{i}": np.array([1.0, 2.0, 3.0]) for i in range(150)}

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        Xt = add_columns_to_df(check_X(X), arrays)

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


# ------------------------
# replace_columns_in_df
# ------------------------


def test_replace_columns_in_df_replaces_values_and_keeps_column_order(make_df):
    X = check_X(make_df(DATA))

    Xt = to_native(
        replace_columns_in_df(
            X, {"var_b": np.array([7, 8, 9]), "var_a": np.array([0.1, 0.2, 0.3])}
        )
    )

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_a", "var_b", "var_c"]
    assert frame_to_dict(Xt) == {
        "var_a": [0.1, 0.2, 0.3],
        "var_b": [7, 8, 9],
        "var_c": ["x", "y", "z"],
    }


def test_replace_columns_in_df_does_not_modify_the_input(make_df):
    X = make_df(DATA)

    replace_columns_in_df(check_X(X), {"var_a": np.array([7, 8, 9])})

    assert frame_to_dict(X) == DATA


def test_replace_columns_in_df_keeps_pandas_index_and_integer_column_names():
    X = pd.DataFrame(
        {0: [1, 2, 3], "var_c": ["x", "y", "z"], 1: [0.5, 1.5, 2.5]},
        index=[10, 10, 30],
    )

    Xt = replace_columns_in_df(check_X(X), {0: np.array([7.0, 8.0, 9.0])})

    expected = pd.DataFrame(
        {0: [7.0, 8.0, 9.0], "var_c": ["x", "y", "z"], 1: [0.5, 1.5, 2.5]},
        index=[10, 10, 30],
    )
    pd.testing.assert_frame_equal(Xt, expected)


def test_replace_columns_in_df_does_not_fragment_pandas_dataframe():
    # pandas warns when many columns are inserted one at a time
    X = pd.DataFrame({f"var_{i}": [1.0, 2.0, 3.0] for i in range(150)})
    arrays = {f"var_{i}": np.array([4, 5, 6]) for i in range(150)}

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        Xt = replace_columns_in_df(check_X(X), arrays)

    assert list(Xt.columns) == list(X.columns)
    assert Xt["var_0"].tolist() == [4, 5, 6]


# ------------------------
# map_categories_in_df
# ------------------------


def test_map_categories_in_df_replaces_categories_and_keeps_column_order(make_df):
    X = make_df(DATA)

    Xt = to_native(
        map_categories_in_df(check_X(X), {"var_c": {"x": 1, "y": 2, "z": 3}})
    )

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["var_a", "var_b", "var_c"]
    assert frame_to_dict(Xt) == {**DATA, "var_c": [1, 2, 3]}
    assert frame_to_dict(X) == DATA


@pytest.mark.parametrize(
    "default, expected", [(None, [1, None, None]), (-1, [1, -1, -1])]
)
def test_map_categories_in_df_with_unseen_categories_and_nulls(
    make_df, default, expected
):
    X = make_df({"var_c": ["x", "w", None]})

    Xt = to_native(
        map_categories_in_df(check_X(X), {"var_c": {"x": 1, "y": 2}}, default=default)
    )

    assert frame_to_dict(Xt) == {"var_c": expected}


def test_map_categories_in_df_returns_numbers_for_pandas_categorical_columns():
    X = pd.DataFrame(
        {"var_c": pd.Categorical(["x", "y", "x"]), "var_d": pd.Categorical(["x"] * 3)},
        index=[10, 10, 30],
    )

    Xt = map_categories_in_df(
        check_X(X), {"var_c": {"x": 1, "y": 2}, "var_d": {"x": 0.5}}
    )

    expected = pd.DataFrame(
        {"var_c": [1, 2, 1], "var_d": [0.5, 0.5, 0.5]}, index=[10, 10, 30]
    )
    pd.testing.assert_frame_equal(Xt, expected)


def test_map_categories_in_df_does_not_fragment_pandas_dataframe():
    X = pd.DataFrame({f"var_{i}": ["x", "y", "x"] for i in range(150)})
    mappings = {f"var_{i}": {"x": 0, "y": 1} for i in range(150)}

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        Xt = map_categories_in_df(check_X(X), mappings)

    assert list(Xt.columns) == list(X.columns)
    assert Xt["var_0"].tolist() == [0, 1, 0]


# ------------------------
# add_category_to_df
# ------------------------


def test_add_category_to_df_adds_category_to_pandas_categorical_columns():
    X = pd.DataFrame(
        {"var_c": pd.Categorical(["x", "y"]), "var_d": ["x", "y"]}, index=[10, 10]
    )

    Xt = add_category_to_df(check_X(X), ["var_c", "var_d"], "Rare")

    assert list(Xt["var_c"].cat.categories) == ["x", "y", "Rare"]
    assert Xt["var_d"].tolist() == ["x", "y"]
    assert list(X["var_c"].cat.categories) == ["x", "y"]


def test_add_category_to_df_skips_columns_that_have_the_category():
    X = pd.DataFrame(
        {"var_a": pd.Categorical(["x", "Rare"]), "var_b": pd.Categorical(["x", "y"])}
    )

    Xt = add_category_to_df(check_X(X), ["var_a", "var_b"], "Rare")

    assert list(Xt["var_a"].cat.categories) == ["Rare", "x"]
    assert list(Xt["var_b"].cat.categories) == ["x", "y", "Rare"]


def test_add_category_to_df_returns_the_same_dataframe_without_pandas_categories(
    make_df,
):
    X = check_X(make_df(DATA))

    assert add_category_to_df(X, ["var_c"], "Rare") is X


DATA_NA = {
    "var_a": [1.0, None, 3.0, 4.0],
    "var_b": ["x", None, "x", "y"],
    "var_c": [1, 2, 3, 4],
}


# ------------------------
# missing_values_mask
# ------------------------


def test_missing_values_mask_returns_one_column_per_variable(make_df):
    X = check_X(make_df(DATA_NA))

    mask = missing_values_mask(X, ["var_b", "var_c", "var_a"])

    assert mask.tolist() == [
        [False, False, False],
        [True, False, True],
        [False, False, False],
        [False, False, False],
    ]


def test_missing_values_mask_with_empty_variables(make_df):
    X = check_X(make_df(DATA_NA))

    assert missing_values_mask(X, []).shape == (4, 0)


# ------------------------
# rows_with_missing_values
# ------------------------


def test_rows_with_missing_values(make_df):
    X = check_X(make_df({**DATA_NA, "var_d": [None, 1.0, 2.0, 3.0]}))

    assert rows_with_missing_values(X, ["var_a", "var_d"]).tolist() == [
        True,
        True,
        False,
        False,
    ]
    assert rows_with_missing_values(X, ["var_c"]).tolist() == [False] * 4


def test_rows_with_missing_values_with_empty_variables(make_df):
    X = check_X(make_df(DATA_NA))

    assert rows_with_missing_values(X, []).tolist() == [False] * 4


# ------------------------
# missing_values_count
# ------------------------


def test_missing_values_count_returns_one_count_per_variable(make_df):
    X = check_X(make_df(DATA_NA))

    assert missing_values_count(X, ["var_c", "var_b", "var_a"]) == [0, 1, 1]
    assert missing_values_count(X, []) == []


# ------------------------
# aggregate_columns
# ------------------------


@pytest.mark.parametrize(
    "method, quantile, expected",
    [
        ("mean", None, {"var_a": 8 / 3, "var_c": 2.5}),
        ("median", None, {"var_a": 3.0, "var_c": 2.5}),
        ("max", None, {"var_a": 4.0, "var_c": 4}),
        ("std", None, {"var_a": 1.5275252316519468, "var_c": 1.2909944487358056}),
        ("quantile", 0.25, {"var_a": 2.0, "var_c": 1.75}),
    ],
)
def test_aggregate_columns_ignores_missing_values(make_df, method, quantile, expected):
    X = check_X(make_df(DATA_NA))

    statistics = aggregate_columns(X, ["var_a", "var_c"], method, quantile)

    assert statistics == pytest.approx(expected)


def test_aggregate_columns_with_empty_variables(make_df):
    assert aggregate_columns(check_X(make_df(DATA_NA)), [], "mean") == {}


# ------------------------
# fill_missing_in_df
# ------------------------


def test_fill_missing_in_df_fills_the_variables(make_df):
    X = make_df(DATA_NA)

    Xt = to_native(fill_missing_in_df(check_X(X), {"var_a": 0.0, "var_b": "M"}))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_a": [1.0, 0.0, 3.0, 4.0],
        "var_b": ["x", "M", "x", "y"],
        "var_c": [1, 2, 3, 4],
    }
    assert frame_to_dict(X) == DATA_NA


def test_fill_missing_in_df_raises_error_with_polars_enum_without_the_value():
    X = pl.DataFrame({"var_b": ["x", None]}, schema={"var_b": pl.Enum(["x", "y"])})
    msg = (
        "Cannot fill variable 'var_b' with 'M': it is a polars Enum with fixed "
        "categories ('x', 'y') that do not include the fill value. Cast the "
        "column to Categorical or String before imputing."
    )

    with pytest.raises(ValueError, match=re.escape(msg)):
        fill_missing_in_df(check_X(X), {"var_b": "M"})


# ------------------------
# filter_rows
# ------------------------


def test_filter_rows_keeps_the_rows_where_mask_is_true(make_df):
    X = check_X(make_df(DATA_NA))

    Xt = to_native(filter_rows(X, np.array([True, False, False, True])))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "var_a": [1.0, 4.0],
        "var_b": ["x", "y"],
        "var_c": [1, 4],
    }


def test_filter_rows_keeps_pandas_index():
    X = pd.DataFrame({"var_a": [1, 2, 3]}, index=[10, 10, 30])

    Xt = filter_rows(check_X(X), np.array([False, True, True]))

    expected = pd.DataFrame({"var_a": [2, 3]}, index=[10, 30])
    pd.testing.assert_frame_equal(Xt, expected)
