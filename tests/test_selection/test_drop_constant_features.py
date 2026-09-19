import re
from datetime import datetime

import pandas as pd
import pytest

from feature_engine.selection import DropConstantFeatures
from tests.backend_helpers import frame_to_dict

DATA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
    "dob": [datetime(2020, 2, 24, 0, minute) for minute in range(4)],
    "const_feat_num": [1, 1, 1, 1],
    "const_feat_cat": ["a", "a", "a", "a"],
    "quasi_feat_num": [1, 1, 1, 2],
    "quasi_feat_cat": ["a", "a", "a", "b"],
}

DATA_NA = {
    "num": [1.0, 2.0, 3.0, 4.0, 5.0],
    "const_num_na": [1.0, 1.0, 1.0, None, 1.0],
    "const_cat_na": ["a", "a", None, "a", "a"],
    "quasi_num_na": [1.0, 1.0, 1.0, None, None],
    "quasi_cat_na": ["a", None, "b", None, None],
    "cat": ["a", "b", "c", "d", "e"],
}


# init parameters
@pytest.mark.parametrize("tol", [2, -0.1, 1.5, "hola", False, None, [0.5]])
def test_error_if_tol_not_allowed(tol):
    msg = f"tol must be a float or integer between 0 and 1. Got {tol} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropConstantFeatures(tol=tol)


@pytest.mark.parametrize("missing_values", [2, "hola", False, None, ["raise"]])
def test_error_if_missing_values_not_allowed(missing_values):
    msg = (
        "missing_values takes only values 'raise', 'ignore' or 'include'. "
        f"Got {missing_values} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropConstantFeatures(missing_values=missing_values)


@pytest.mark.parametrize(
    "tol, missing_values, confirm_variables",
    [(1, "raise", False), (0, "ignore", True), (0.7, "include", False)],
)
def test_init_param_assignment(tol, missing_values, confirm_variables):
    sel = DropConstantFeatures(
        tol=tol, missing_values=missing_values, confirm_variables=confirm_variables
    )
    assert sel.tol == tol
    assert sel.missing_values == missing_values
    assert sel.confirm_variables is confirm_variables


# fit and transform
def test_drop_constant_features(make_df):
    X = make_df(DATA)
    sel = DropConstantFeatures()
    Xt = sel.fit_transform(X)

    assert sel.variables_ == list(DATA.keys())
    assert sel.features_to_drop_ == ["const_feat_num", "const_feat_cat"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        k: v for k, v in DATA.items() if k not in ["const_feat_num", "const_feat_cat"]
    }


@pytest.mark.parametrize(
    "tol, expected",
    [
        (0.8, ["const_feat_num", "const_feat_cat"]),
        (
            0.75,
            ["const_feat_num", "const_feat_cat", "quasi_feat_num", "quasi_feat_cat"],
        ),
        (
            0.7,
            ["const_feat_num", "const_feat_cat", "quasi_feat_num", "quasi_feat_cat"],
        ),
    ],
)
def test_drop_quasi_constant_features(make_df, tol, expected):
    X = make_df(DATA)
    sel = DropConstantFeatures(tol=tol)
    Xt = sel.fit_transform(X)

    assert sel.features_to_drop_ == expected
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {k: v for k, v in DATA.items() if k not in expected}


def test_variables_list(make_df):
    X = make_df(DATA)
    sel = DropConstantFeatures(
        tol=0.7, variables=["Name", "const_feat_num", "quasi_feat_num"]
    )
    Xt = sel.fit_transform(X)

    assert sel.variables_ == ["Name", "const_feat_num", "quasi_feat_num"]
    assert sel.features_to_drop_ == ["const_feat_num", "quasi_feat_num"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        k: v for k, v in DATA.items() if k not in ["const_feat_num", "quasi_feat_num"]
    }


def test_single_variable(make_df):
    sel = DropConstantFeatures(variables="const_feat_cat").fit(make_df(DATA))
    assert sel.variables_ == ["const_feat_cat"]
    assert sel.features_to_drop_ == ["const_feat_cat"]


def test_confirm_variables(make_df):
    sel = DropConstantFeatures(
        variables=["const_feat_num", "Age", "not_in_df"], confirm_variables=True
    ).fit(make_df(DATA))
    assert sel.variables_ == ["const_feat_num", "Age"]
    assert sel.features_to_drop_ == ["const_feat_num"]


@pytest.mark.parametrize(
    "data, tol",
    [
        ({"col1": [1, 1, 1], "col2": ["a", "a", "a"]}, 1),
        ({"col1": [1, 1, 1, 1], "col2": [1, 1, 1, 2], "col3": [1, 2, 2, 2]}, 0.7),
        ({"col1": [1, 2, 3, 4], "col2": [1, 1, 2, 2]}, 0),
    ],
)
def test_error_if_all_features_are_dropped(make_df, data, tol):
    msg = (
        "The resulting dataframe will have no columns after dropping all "
        "constant or quasi-constant features. Try changing the tol value."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropConstantFeatures(tol=tol).fit(make_df(data))


def test_error_if_missing_values_raise(make_df):
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropConstantFeatures(missing_values="raise").fit(make_df(DATA_NA))


def test_missing_values_raise_checks_only_selected_variables(make_df):
    sel = DropConstantFeatures(variables=["num", "cat"], missing_values="raise")
    sel.fit(make_df(DATA_NA))
    assert sel.features_to_drop_ == []


@pytest.mark.parametrize(
    "tol, expected",
    [
        # with tol=1, variables with a single value besides missing data are dropped.
        (1, ["const_num_na", "const_cat_na", "quasi_num_na"]),
        (0.8, ["const_num_na", "const_cat_na"]),
        (0.6, ["const_num_na", "const_cat_na", "quasi_num_na"]),
    ],
)
def test_missing_values_ignore(make_df, tol, expected):
    X = make_df(DATA_NA)
    sel = DropConstantFeatures(tol=tol, missing_values="ignore")
    Xt = sel.fit_transform(X)

    assert sel.features_to_drop_ == expected
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {k: v for k, v in DATA_NA.items() if k not in expected}


@pytest.mark.parametrize(
    "tol, expected",
    [
        (1, []),
        (0.8, ["const_num_na", "const_cat_na"]),
        (0.6, ["const_num_na", "const_cat_na", "quasi_num_na", "quasi_cat_na"]),
    ],
)
def test_missing_values_include(make_df, tol, expected):
    X = make_df(DATA_NA)
    sel = DropConstantFeatures(tol=tol, missing_values="include")
    Xt = sel.fit_transform(X)

    assert sel.features_to_drop_ == expected
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {k: v for k, v in DATA_NA.items() if k not in expected}


@pytest.mark.parametrize(
    "missing_values, tol, expected",
    [
        ("ignore", 1, []),
        ("ignore", 0.5, []),
        ("include", 1, ["all_na"]),
        ("include", 0.5, ["all_na"]),
    ],
)
def test_variable_with_only_missing_values(make_df, missing_values, tol, expected):
    X = make_df({"all_na": [None, None, None], "num": [1.0, 2.0, 3.0]})
    sel = DropConstantFeatures(tol=tol, missing_values=missing_values).fit(X)
    assert sel.features_to_drop_ == expected


@pytest.mark.parametrize(
    "missing_values, tol, expected",
    [
        ("ignore", 1, ["nan_num"]),
        ("ignore", 0.5, ["nan_num"]),
        ("ignore", 0.6, []),
        ("include", 1, []),
        ("include", 0.5, ["nan_num"]),
    ],
)
def test_nan_is_treated_as_missing(make_df, missing_values, tol, expected):
    nan = float("nan")
    X = make_df({"nan_num": [nan, 1.0, nan, 1.0], "num": [1.0, 2.0, 3.0, 4.0]})
    sel = DropConstantFeatures(tol=tol, missing_values=missing_values).fit(X)
    assert sel.features_to_drop_ == expected


def test_error_if_nan_and_missing_values_raise(make_df):
    X = make_df({"nan_num": [float("nan"), 1.0, 2.0], "num": [1.0, 2.0, 3.0]})
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropConstantFeatures().fit(X)


@pytest.mark.parametrize("tol, expected", [(0.8, []), (0.4, ["literal"])])
def test_include_counts_missing_values_apart_from_other_values(make_df, tol, expected):
    X = make_df(
        {
            "literal": ["missing_values", "missing_values", None, None, "x"],
            "num": [1, 2, 3, 4, 5],
        }
    )
    sel = DropConstantFeatures(tol=tol, missing_values="include").fit(X)
    assert sel.features_to_drop_ == expected


def test_fit_does_not_modify_input(make_df):
    X = make_df(DATA_NA)
    DropConstantFeatures(tol=0.7, missing_values="include").fit_transform(X)
    assert frame_to_dict(X) == DATA_NA


def test_get_support(make_df):
    sel = DropConstantFeatures(tol=0.7).fit(make_df(DATA))
    assert sel.get_support() == [True] * 5 + [False] * 4


def test_integer_column_names():
    X = pd.DataFrame({0: [1, 1, 1], 1: [1, 2, 3], "c": ["a", "a", "b"]})
    sel = DropConstantFeatures(tol=0.6)
    Xt = sel.fit_transform(X)

    assert sel.features_to_drop_ == [0, "c"]
    pd.testing.assert_frame_equal(Xt, X[[1]])


def test_pandas_index_is_preserved():
    X = pd.DataFrame({"a": [1, 1, 1, 1], "b": [1, 2, 3, 4]}, index=[10, 3, 7, 1])
    Xt = DropConstantFeatures().fit_transform(X)
    pd.testing.assert_frame_equal(Xt, X[["b"]])


@pytest.mark.parametrize(
    "missing_values, tol, expected",
    [
        ("ignore", 1, []),
        ("ignore", 0.5, ["c"]),
        ("include", 1, []),
        ("include", 0.5, ["c"]),
    ],
)
def test_pandas_category_dtype(missing_values, tol, expected):
    # the unused category "z" must not count as a value.
    X = pd.DataFrame(
        {
            "c": pd.Categorical(["a", "a", None, "b"], categories=["a", "b", "z"]),
            "d": [1, 2, 3, 4],
        }
    )
    sel = DropConstantFeatures(tol=tol, missing_values=missing_values).fit(X)
    assert sel.features_to_drop_ == expected


@pytest.mark.parametrize(
    "missing_values, tol, expected",
    [
        ("ignore", 1, ["a"]),
        ("ignore", 0.75, ["a"]),
        ("include", 1, []),
        ("include", 0.75, ["a"]),
    ],
)
def test_pandas_nullable_integer_dtype(missing_values, tol, expected):
    X = pd.DataFrame(
        {"a": pd.array([1, 1, None, 1], dtype="Int64"), "b": [1, 2, 3, 4]}
    )
    sel = DropConstantFeatures(tol=tol, missing_values=missing_values).fit(X)
    assert sel.features_to_drop_ == expected
