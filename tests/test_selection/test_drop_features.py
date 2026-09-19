import re

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.selection import DropFeatures
from tests.backend_helpers import frame_to_dict


@pytest.fixture
def data():
    return {
        "Name": ["tom", "nick", None, "jack"],
        "City": ["London", "Manchester", "Liverpool", "Bristol"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, None, 0.7, 0.6],
    }


# init parameters
@pytest.mark.parametrize("features_to_drop", [[], "", None, 1, 0.5, True, ("Age",)])
def test_error_if_features_to_drop_not_allowed(features_to_drop):
    msg = (
        "features_to_drop should be a list with the name of the variables you "
        f"wish to drop from the dataframe. Got {features_to_drop} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        DropFeatures(features_to_drop=features_to_drop)


@pytest.mark.parametrize("features_to_drop", ["Age", ["Age"], ["Age", "City"], [0, 1]])
def test_init_param_assignment(features_to_drop):
    transformer = DropFeatures(features_to_drop=features_to_drop)
    assert transformer.features_to_drop == features_to_drop


# fit and transform
@pytest.mark.parametrize(
    "features_to_drop, expected_features_to_drop",
    [("City", ["City"]), (["City"], ["City"]), (["City", "Marks"], ["City", "Marks"])],
)
def test_fit_attributes(make_df, data, features_to_drop, expected_features_to_drop):
    transformer = DropFeatures(features_to_drop=features_to_drop)
    transformer.fit(make_df(data))
    assert transformer.features_to_drop_ == expected_features_to_drop
    assert transformer.feature_names_in_ == ["Name", "City", "Age", "Marks"]
    assert transformer.n_features_in_ == 4


def test_drop_one_variable(make_df, data):
    transformer = DropFeatures(features_to_drop="City")
    Xt = transformer.fit_transform(make_df(data))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "Name": ["tom", "nick", None, "jack"],
        "Age": [20, 21, 19, 18],
        "Marks": [0.9, None, 0.7, 0.6],
    }


def test_drop_several_variables(make_df, data):
    transformer = DropFeatures(features_to_drop=["City", "Marks"])
    Xt = transformer.fit_transform(make_df(data))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "Name": ["tom", "nick", None, "jack"],
        "Age": [20, 21, 19, 18],
    }


def test_transform_returns_training_column_order(make_df, data):
    transformer = DropFeatures(features_to_drop=["City"])
    transformer.fit(make_df(data))
    reordered = {col: data[col] for col in ["Marks", "Age", "City", "Name"]}
    Xt = transformer.transform(make_df(reordered))

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["Name", "Age", "Marks"]


def test_get_feature_names_out_and_support(make_df, data):
    transformer = DropFeatures(features_to_drop=["City", "Marks"])
    transformer.fit(make_df(data))

    assert transformer.get_feature_names_out() == ["Name", "Age"]
    assert transformer.get_support() == [True, False, True, False]
    assert list(transformer.get_support(indices=True)) == [0, 2]


def test_repeated_features_to_drop(make_df, data):
    transformer = DropFeatures(features_to_drop=["City", "City", "City", "City"])
    Xt = transformer.fit_transform(make_df(data))

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["Name", "Age", "Marks"]


@pytest.mark.parametrize(
    "features_to_drop, msg",
    [
        ("Surname", "The variable Surname is not in the dataframe."),
        (["City", "Surname"], "Some of the variables are not in the dataframe."),
    ],
)
def test_error_if_variable_not_in_dataframe(make_df, data, features_to_drop, msg):
    transformer = DropFeatures(features_to_drop=features_to_drop)
    with pytest.raises(KeyError, match=re.escape(msg)):
        transformer.fit(make_df(data))


@pytest.mark.parametrize(
    "features_to_drop",
    [["Name", "City", "Age", "Marks"], ["Marks", "Name", "Age", "City", "Age"]],
)
def test_error_if_dropping_all_variables(make_df, data, features_to_drop):
    msg = (
        "The resulting dataframe will have no columns after dropping all existing "
        "variables"
    )
    transformer = DropFeatures(features_to_drop=features_to_drop)
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.fit(make_df(data))


def test_error_if_transform_before_fit(make_df, data):
    msg = (
        "This DropFeatures instance is not fitted yet. Call 'fit' with appropriate "
        "arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        DropFeatures(features_to_drop="City").transform(make_df(data))


def test_integer_column_names():
    # polars does not allow integer column names.
    X = pd.DataFrame({0: ["tom", "nick"], 1: [20, 21], 2: [0.9, 0.8]})
    transformer = DropFeatures(features_to_drop=[0, 1])
    Xt = transformer.fit_transform(X)

    assert transformer.features_to_drop_ == [0, 1]
    assert transformer.feature_names_in_ == [0, 1, 2]
    pd.testing.assert_frame_equal(Xt, pd.DataFrame({2: [0.9, 0.8]}))


def test_pandas_index_is_kept(data):
    X = pd.DataFrame(data, index=[10, 20, 30, 40])
    Xt = DropFeatures(features_to_drop="City").fit_transform(X)

    expected = pd.DataFrame(
        {
            "Name": ["tom", "nick", None, "jack"],
            "Age": [20, 21, 19, 18],
            "Marks": [0.9, None, 0.7, 0.6],
        },
        index=[10, 20, 30, 40],
    )
    pd.testing.assert_frame_equal(Xt, expected)
    assert list(X.columns) == ["Name", "City", "Age", "Marks"]
