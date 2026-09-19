import re
from datetime import datetime

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.selection import DropDuplicateFeatures
from tests.backend_helpers import frame_to_dict

DOB = [datetime(2020, 2, 24, 0, minute) for minute in range(4)]

DATA = {
    "Name": ["tom", "nick", "krish", "jack"],
    "dob2": DOB,
    "City": ["London", "Manchester", "Liverpool", "Bristol"],
    "Age": [20, 21, 19, 18],
    "Marks": [0.9, 0.8, 0.7, 0.6],
    "dob": DOB,
    "City2": ["London", "Manchester", "Liverpool", "Bristol"],
    "dob3": DOB,
    "Age2": [20, 21, 19, 18],
}

DATA_NA = {
    "Name": ["tom", "nick", "krish", "jack", None],
    "City": ["London", "Manchester", "Liverpool", "Bristol", None],
    "Age": [20, 21, None, 18, 34],
    "Marks": [0.9, 0.8, 0.7, 0.6, 0.5],
    "City2": ["London", "Manchester", "Liverpool", "Bristol", None],
    "Age2": [20, 21, None, 18, 34],
}

MSG_NA = (
    "Some of the variables in the dataset contain NaN. Check and "
    "remove those before using this transformer."
)


# init parameters
# the errors of variables and confirm_variables are tested in test_base_selector.py
# and in the variable handling tests.
@pytest.mark.parametrize("missing_values", ["hola", "include", 1, None, ["raise"]])
def test_error_if_missing_values_not_permitted(missing_values):
    msg = (
        "missing_values takes only values 'raise' or 'ignore'. "
        f"Got {missing_values} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropDuplicateFeatures(missing_values=missing_values)


@pytest.mark.parametrize(
    "missing_values, confirm_variables", [("raise", True), ("ignore", False)]
)
def test_init_param_assignment(missing_values, confirm_variables):
    transformer = DropDuplicateFeatures(
        missing_values=missing_values, confirm_variables=confirm_variables
    )
    assert transformer.missing_values == missing_values
    assert transformer.confirm_variables == confirm_variables


# fit and transform
def test_drop_duplicated_features(make_df):
    transformer = DropDuplicateFeatures()
    Xt = transformer.fit_transform(make_df(DATA))

    assert transformer.variables_ == list(DATA)
    assert transformer.features_to_drop_ == {"dob", "dob3", "City2", "Age2"}
    assert transformer.duplicated_feature_sets_ == [
        {"dob", "dob2", "dob3"},
        {"City", "City2"},
        {"Age", "Age2"},
    ]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "Name": ["tom", "nick", "krish", "jack"],
        "dob2": DOB,
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, 0.8, 0.7, 0.6],
    }


def test_missing_values_are_equal(make_df):
    transformer = DropDuplicateFeatures()
    Xt = transformer.fit_transform(make_df(DATA_NA))

    assert transformer.features_to_drop_ == {"City2", "Age2"}
    assert transformer.duplicated_feature_sets_ == [{"City", "City2"}, {"Age", "Age2"}]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "Name": ["tom", "nick", "krish", "jack", None],
        "City": ["London", "Manchester", "Liverpool", "Bristol", None],
        "Age": [20, 21, None, 18, 34],
        "Marks": [0.9, 0.8, 0.7, 0.6, 0.5],
    }


def test_duplicates_across_numerical_and_boolean_types(make_df):
    data = {
        "float": [5.5, 5.5, 5.5],
        "int": [1, 1, 1],
        "string": ["foo", "foo", "foo"],
        "date": [datetime(2001, 1, 2)] * 3,
        "float_one": [1.0, 1.0, 1.0],
        "false": [False, False, False],
        "true": [True, True, True],
        "int8": [1, 1, 1],
    }
    X = nw.from_native(make_df(data))
    X = X.with_columns(nw.col("int8").cast(nw.Int8)).to_native()
    transformer = DropDuplicateFeatures()
    Xt = transformer.fit_transform(X)

    assert transformer.duplicated_feature_sets_ == [
        {"int", "float_one", "true", "int8"}
    ]
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "float": [5.5, 5.5, 5.5],
        "int": [1, 1, 1],
        "string": ["foo", "foo", "foo"],
        "date": [datetime(2001, 1, 2)] * 3,
        "false": [False, False, False],
    }


def test_duplicates_across_categorical_and_string_types(make_df):
    data = {"string": ["a", "b", "a"], "categorical": ["a", "b", "a"]}
    X = nw.from_native(make_df(data))
    X = X.with_columns(nw.col("categorical").cast(nw.Categorical)).to_native()
    transformer = DropDuplicateFeatures().fit(X)

    assert transformer.features_to_drop_ == {"categorical"}


def test_duplicates_across_datetime_units(make_df):
    data = {"ns": DOB, "ms": DOB, "shifted": [d.replace(second=1) for d in DOB]}
    X = nw.from_native(make_df(data))
    X = X.with_columns(
        nw.col("ns").cast(nw.Datetime("ns")), nw.col("ms").cast(nw.Datetime("ms"))
    ).to_native()
    transformer = DropDuplicateFeatures().fit(X)

    assert transformer.features_to_drop_ == {"ms"}


def test_empty_variables_are_duplicates_whatever_their_type(make_df):
    data = {
        "string": ["a", None, "b"],
        "number": [1.0, None, 2.0],
        "empty_float": [None, None, None],
    }
    X = nw.from_native(make_df(data))
    X = X.with_columns(
        nw.col("empty_float").cast(nw.Float64),
        nw.lit(None, dtype=nw.String).alias("empty_string"),
        nw.lit(None, dtype=nw.Datetime("us")).alias("empty_datetime"),
    ).to_native()
    transformer = DropDuplicateFeatures().fit(X)

    assert transformer.duplicated_feature_sets_ == [
        {"empty_float", "empty_string", "empty_datetime"}
    ]


def test_large_integers_are_compared_exactly(make_df):
    # as floats, 2**60 and 2**60 + 1 are the same number.
    data = {"a": [2**60, 1], "b": [2**60 + 1, 1], "c": [2**60, 1], "d": [0.5, 1.0]}
    transformer = DropDuplicateFeatures().fit(make_df(data))

    assert transformer.duplicated_feature_sets_ == [{"a", "c"}]


def test_first_variable_of_each_group_is_kept(make_df):
    transformer = DropDuplicateFeatures(variables=["Age2", "City2", "City", "Age"])
    Xt = transformer.fit_transform(make_df(DATA))

    assert transformer.features_to_drop_ == {"City", "Age"}
    assert isinstance(Xt, make_df)
    assert list(frame_to_dict(Xt)) == [
        "Name",
        "dob2",
        "Marks",
        "dob",
        "City2",
        "dob3",
        "Age2",
    ]


def test_only_selected_variables_are_examined(make_df):
    transformer = DropDuplicateFeatures(variables=["Name", "City", "City2", "Age"])
    Xt = transformer.fit_transform(make_df(DATA))

    assert transformer.variables_ == ["Name", "City", "City2", "Age"]
    assert transformer.features_to_drop_ == {"City2"}
    assert transformer.duplicated_feature_sets_ == [{"City", "City2"}]
    assert isinstance(Xt, make_df)
    assert list(frame_to_dict(Xt)) == [
        "Name",
        "dob2",
        "City",
        "Age",
        "Marks",
        "dob",
        "dob3",
        "Age2",
    ]


def test_confirm_variables(make_df):
    transformer = DropDuplicateFeatures(
        variables=["Age", "Age2", "Marks", "Height"], confirm_variables=True
    )
    transformer.fit(make_df(DATA))

    assert transformer.variables_ == ["Age", "Age2", "Marks"]
    assert transformer.features_to_drop_ == {"Age2"}


def test_no_duplicates(make_df):
    data = {"a": [1, 2, 3], "b": [3, 2, 1], "c": [2, 1, 3]}
    transformer = DropDuplicateFeatures()
    Xt = transformer.fit_transform(make_df(data))

    assert transformer.features_to_drop_ == set()
    assert transformer.duplicated_feature_sets_ == []
    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == data


def test_error_if_missing_values_raise_and_data_has_nan(make_df):
    transformer = DropDuplicateFeatures(missing_values="raise")
    with pytest.raises(ValueError, match=re.escape(MSG_NA)):
        transformer.fit(make_df(DATA_NA))


def test_error_if_fewer_than_2_variables(make_df):
    msg = (
        "The selector needs at least 2 or more variables to select from. "
        "Got only 1 variable: ['Age']."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropDuplicateFeatures(variables=["Age"]).fit(make_df(DATA))


def test_fit_does_not_modify_input(make_df):
    X = make_df(DATA_NA)
    DropDuplicateFeatures().fit_transform(X)

    assert frame_to_dict(X) == DATA_NA


def test_get_support(make_df):
    transformer = DropDuplicateFeatures().fit(make_df(DATA))

    assert transformer.get_support() == [
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
    ]


def test_error_if_transform_before_fit(make_df):
    msg = (
        "This DropDuplicateFeatures instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        DropDuplicateFeatures().transform(make_df(DATA))


def test_nan_and_null_are_equal_in_polars():
    X = pl.DataFrame({"nan": [1.0, np.nan, 2.0], "null": [1.0, None, 2.0]})
    transformer = DropDuplicateFeatures().fit(X)

    assert transformer.features_to_drop_ == {"null"}


def test_integer_column_names():
    X = pd.DataFrame({0: [1, 2, 3], 1: ["a", "b", "c"], 2: [1.0, 2.0, 3.0]})
    transformer = DropDuplicateFeatures()
    Xt = transformer.fit_transform(X)

    assert transformer.features_to_drop_ == {2}
    pd.testing.assert_frame_equal(Xt, X[[0, 1]])
