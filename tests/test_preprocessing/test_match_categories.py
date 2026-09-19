import re

import narwhals as nw
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.preprocessing import MatchCategories
from tests.backend_helpers import frame_to_dict

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer or set the parameter "
    "`missing_values='ignore'` when initialising this transformer."
)

MSG_NA_INTRODUCED = (
    "During the encoding, NaN values were introduced in the feature(s) {}."
)

TRAIN = {"x1": ["b", "a", "c", "a"], "x2": [4, 5, 6, 7], "x3": ["z", "y", "z", "y"]}
TEST = {"x1": ["c", "d", "a", "b"], "x2": [5, 6, 4, 7], "x3": ["y", "w", "z", "y"]}


def dtype_categories(X, variable):
    if isinstance(X, pd.DataFrame):
        return list(X[variable].cat.categories)
    return list(X.schema[variable].categories)


# init parameters
@pytest.mark.parametrize(
    "missing_values", ["other", "Raise", "", 1, 0.5, True, None, ["raise"]]
)
def test_error_if_missing_values_not_allowed(missing_values):
    msg = (
        "missing_values takes only values 'raise' or 'ignore'. "
        f"Got {missing_values} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MatchCategories(missing_values=missing_values)


@pytest.mark.parametrize("ignore_format", ["True", 1, 0, None, [True]])
def test_error_if_ignore_format_not_bool(ignore_format):
    msg = (
        "ignore_format takes only booleans True and False. "
        f"Got {ignore_format} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MatchCategories(ignore_format=ignore_format)


@pytest.mark.parametrize("return_empty", ["True", 1, 0, None, [True]])
def test_error_if_return_empty_not_bool(return_empty):
    msg = (
        "return_empty takes only boolean values True and False. "
        f"Got {return_empty} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        MatchCategories(return_empty=return_empty)


@pytest.mark.parametrize(
    "missing_values, ignore_format", [("raise", False), ("ignore", True)]
)
def test_init_param_assignment(missing_values, ignore_format):
    transformer = MatchCategories(
        missing_values=missing_values, ignore_format=ignore_format
    )
    assert transformer.missing_values == missing_values
    assert transformer.ignore_format is ignore_format


# fit and transform
def test_learns_categories_and_casts_to_categorical(make_df):
    transformer = MatchCategories()
    transformer.fit(make_df(TRAIN))
    Xt = transformer.transform(make_df(TRAIN))

    assert transformer.variables_ == ["x1", "x3"]
    assert {k: list(v) for k, v in transformer.category_dict_.items()} == {
        "x1": ["a", "b", "c"],
        "x3": ["y", "z"],
    }
    assert transformer.n_features_in_ == 3
    assert transformer.feature_names_in_ == ["x1", "x2", "x3"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == TRAIN
    assert dtype_categories(Xt, "x1") == ["a", "b", "c"]
    assert dtype_categories(Xt, "x3") == ["y", "z"]


def test_categories_are_the_same_in_train_and_test(make_df):
    train = make_df({"x1": ["b", "a", "c"]})
    test = make_df({"x1": ["c", "b", "c"]})
    transformer = MatchCategories().fit(train)

    assert dtype_categories(transformer.transform(train), "x1") == ["a", "b", "c"]
    assert dtype_categories(transformer.transform(test), "x1") == ["a", "b", "c"]


def test_unseen_categories_become_nan_and_warn(make_df):
    transformer = MatchCategories(missing_values="ignore").fit(make_df(TRAIN))

    with pytest.warns(UserWarning, match=re.escape(MSG_NA_INTRODUCED.format("x1, x3"))):
        Xt = transformer.transform(make_df(TEST))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "x1": ["c", None, "a", "b"],
        "x2": [5, 6, 4, 7],
        "x3": ["y", None, "z", "y"],
    }
    assert dtype_categories(Xt, "x1") == ["a", "b", "c"]


def test_error_if_unseen_categories_when_missing_values_raise(make_df):
    transformer = MatchCategories().fit(make_df(TRAIN))
    with pytest.raises(ValueError, match=re.escape(MSG_NA_INTRODUCED.format("x1, x3"))):
        transformer.transform(make_df(TEST))


def test_error_if_nan_in_fit_when_missing_values_raise(make_df):
    X = make_df({"x1": ["a", None, "b"]})
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        MatchCategories().fit(X)


def test_error_if_nan_in_transform_when_missing_values_raise(make_df):
    transformer = MatchCategories().fit(make_df({"x1": ["a", "b", "b"]}))
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.transform(make_df({"x1": ["a", None, "b"]}))


def test_nan_is_not_a_category_when_missing_values_ignore(make_df):
    X = make_df({"x1": ["b", None, "a", "b"], "x2": [1.0, None, 2.0, 3.0]})
    transformer = MatchCategories(missing_values="ignore").fit(X)

    with pytest.warns(UserWarning, match=re.escape(MSG_NA_INTRODUCED.format("x1"))):
        Xt = transformer.transform(X)

    assert {k: list(v) for k, v in transformer.category_dict_.items()} == {
        "x1": ["a", "b"]
    }
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "x1": ["b", None, "a", "b"],
        "x2": [1.0, None, 2.0, 3.0],
    }


@pytest.mark.parametrize("variables", ["x3", ["x3"]])
def test_transforms_only_selected_variables(make_df, variables):
    transformer = MatchCategories(variables=variables, missing_values="ignore")
    transformer.fit(make_df(TRAIN))

    with pytest.warns(UserWarning, match=re.escape(MSG_NA_INTRODUCED.format("x3"))):
        Xt = transformer.transform(make_df(TEST))

    assert transformer.variables_ == ["x3"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "x1": ["c", "d", "a", "b"],
        "x2": [5, 6, 4, 7],
        "x3": ["y", None, "z", "y"],
    }
    assert nw.from_native(Xt).schema["x1"] == nw.String


def test_keeps_categories_of_categorical_input(make_df):
    # the categories come from the dtype, so unused ones and their order are kept.
    X = (
        nw.from_native(make_df({"x1": ["b", "a", "b"]}))
        .with_columns(nw.col("x1").cast(nw.Enum(["z", "b", "a"])))
        .to_native()
    )
    transformer = MatchCategories().fit(X)
    Xt = transformer.transform(make_df({"x1": ["a", "b", "a"]}))

    assert list(transformer.category_dict_["x1"]) == ["z", "b", "a"]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"x1": ["a", "b", "a"]}
    assert dtype_categories(Xt, "x1") == ["z", "b", "a"]


def test_ignore_format_casts_numerical_variables(make_df):
    # polars categorical dtypes only take strings, so numbers become strings.
    X = make_df({"x1": [3, 1, 2, 10], "x2": [1.5, float("nan"), 2.5, 10.0]})
    X_test = make_df({"x1": [1, 5, 10, 3], "x2": [2.5, 1.5, 7.0, 10.0]})
    transformer = MatchCategories(ignore_format=True, missing_values="ignore")
    transformer.fit(X)

    with pytest.warns(UserWarning, match=re.escape(MSG_NA_INTRODUCED.format("x1, x2"))):
        Xt = transformer.transform(X_test)

    expected_categories = {
        pd.DataFrame: {"x1": [1, 2, 3, 10], "x2": [1.5, 2.5, 10.0]},
        pl.DataFrame: {"x1": ["1", "2", "3", "10"], "x2": ["1.5", "2.5", "10.0"]},
    }
    expected_values = {
        pd.DataFrame: {"x1": [1.0, None, 10.0, 3.0], "x2": [2.5, 1.5, None, 10.0]},
        pl.DataFrame: {
            "x1": ["1", None, "10", "3"],
            "x2": ["2.5", "1.5", None, "10.0"],
        },
    }
    assert {
        k: list(v) for k, v in transformer.category_dict_.items()
    } == expected_categories[make_df]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == expected_values[make_df]
    assert dtype_categories(Xt, "x1") == expected_categories[make_df]["x1"]


def test_return_empty_when_no_categorical_variables(make_df):
    X = make_df({"x1": [1, 2, 3], "x2": [1.0, 2.0, 3.0]})
    transformer = MatchCategories(return_empty=True)

    with pytest.warns(
        UserWarning,
        match=re.escape(
            "No categorical variables found in this dataframe. "
            "Returning an empty list."
        ),
    ):
        transformer.fit(X)
    Xt = transformer.transform(X)

    assert transformer.variables_ == []
    assert transformer.category_dict_ == {}
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"x1": [1, 2, 3], "x2": [1.0, 2.0, 3.0]}


def test_error_if_no_categorical_variables(make_df):
    msg = (
        "No categorical variables found in this dataframe. Check variable "
        "dtypes or set return_empty to True to return an empty list instead."
    )
    with pytest.raises(TypeError, match=re.escape(msg)):
        MatchCategories().fit(make_df({"x1": [1, 2, 3]}))


def test_does_not_modify_input(make_df):
    X = make_df(TEST)
    transformer = MatchCategories(missing_values="ignore").fit(make_df(TRAIN))

    with pytest.warns(UserWarning, match=re.escape(MSG_NA_INTRODUCED.format("x1, x3"))):
        transformer.transform(X)

    assert frame_to_dict(X) == TEST
    assert nw.from_native(X).schema["x1"] == nw.String


def test_error_if_transform_before_fit(make_df):
    msg = (
        "This MatchCategories instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        MatchCategories().transform(make_df(TRAIN))


def test_output_dtype_is_pandas_category():
    Xt = MatchCategories(missing_values="ignore").fit(pd.DataFrame(TRAIN))

    with pytest.warns(UserWarning, match=re.escape(MSG_NA_INTRODUCED.format("x1, x3"))):
        Xt = Xt.transform(pd.DataFrame(TEST))

    expected = pd.DataFrame(
        {
            "x1": pd.Categorical(["c", None, "a", "b"], categories=["a", "b", "c"]),
            "x2": [5, 6, 4, 7],
            "x3": pd.Categorical(["y", None, "z", "y"], categories=["y", "z"]),
        }
    )
    pd.testing.assert_frame_equal(Xt, expected)


def test_output_dtype_is_polars_enum():
    Xt = MatchCategories().fit_transform(pl.DataFrame(TRAIN))
    assert Xt.schema == pl.Schema(
        {"x1": pl.Enum(["a", "b", "c"]), "x2": pl.Int64, "x3": pl.Enum(["y", "z"])}
    )


def test_integer_column_names():
    X = pd.DataFrame({0: ["a", "b", "c"], 1: ["x", "y", "x"], "n": [1, 2, 3]})
    X_test = pd.DataFrame({0: ["a", "q", "c"], 1: ["x", "y", "w"], "n": [1, 2, 3]})
    transformer = MatchCategories(missing_values="ignore").fit(X)

    with pytest.warns(UserWarning, match=re.escape(MSG_NA_INTRODUCED.format("0, 1"))):
        Xt = transformer.transform(X_test)

    expected = pd.DataFrame(
        {
            0: pd.Categorical(["a", None, "c"], categories=["a", "b", "c"]),
            1: pd.Categorical(["x", "y", None], categories=["x", "y"]),
            "n": [1, 2, 3],
        }
    )
    pd.testing.assert_frame_equal(Xt, expected)


def test_keeps_pandas_index():
    X = pd.DataFrame(TRAIN, index=[10, 20, 30, 40])
    Xt = MatchCategories().fit_transform(X)
    pd.testing.assert_index_equal(Xt.index, X.index)
