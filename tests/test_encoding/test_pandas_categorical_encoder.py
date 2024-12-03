import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import PandasCategoricalEncoder


def test_fit_with_specified_variables():
    """
    Test fitting the transformer with specified variables.
    """
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    transformer = PandasCategoricalEncoder(variables=["A"])
    transformer.fit(df)

    assert transformer.variables == ["A"]
    assert transformer.encoder_dict_ == {"A": {"a": 0, "b": 1, "c": 2}}


def test_fit_with_all_object_variables():
    """
    Test fitting the transformer with all object variables.
    """
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    transformer = PandasCategoricalEncoder()
    transformer.fit(df)

    assert transformer.variables_ == ["A", "B"]
    assert transformer.encoder_dict_ == {
        "A": {"a": 0, "b": 1, "c": 2},
        "B": {"x": 0, "y": 1, "z": 2},
    }


def test_transform_alphabetically_unordered_category():
    """
    Test transforming a dataframe with a category that is not alphabetically ordered
    (c).
    """
    df = pd.DataFrame({"A": ["c", "a", "b", "a"], "B": ["x", "y", "x", "z"]})
    transformer = PandasCategoricalEncoder(variables=["A", "B"])
    transformer.fit(df)
    transformed_df = transformer.transform(df)

    assert transformed_df["A"].dtype.name == "category"
    assert transformed_df["B"].dtype.name == "category"
    assert list(transformed_df["A"].cat.categories) == ["a", "b", "c"]
    assert list(transformed_df["B"].cat.categories) == ["x", "y", "z"]
    assert transformed_df["A"].cat.codes.tolist() == [2, 0, 1, 0]
    assert transformed_df["B"].cat.codes.tolist() == [0, 1, 0, 2]
    assert transformer.variables_ == ["A", "B"]
    assert transformer.encoder_dict_ == {
        "A": {"a": 0, "b": 1, "c": 2},
        "B": {"x": 0, "y": 1, "z": 2},
    }


def test_transform_with_unseen_data_and_unseen_is_ignore():
    """
    Test transforming the dataframe with unseen data.
    """
    df_train = pd.DataFrame({"A": ["a", "c", "b", "a"], "B": ["x", "y", "x", "z"]})
    df_test = pd.DataFrame(
        {"A": ["a", "b", "c", "unseen"], "B": ["x", "y", "z", "unseen"]}
    )
    transformer = PandasCategoricalEncoder(variables=["A", "B"])
    transformed_train_df = transformer.fit_transform(df_train)

    with pytest.warns(UserWarning) as record:
        transformed_test_df = transformer.transform(df_test)

    msg = "During the encoding, NaN values were introduced in the feature(s) A, B."

    # check that only one warning was raised
    assert len(record) == 1
    # check that the message matches
    assert record[0].message.args[0] == msg

    assert transformed_test_df["A"].dtype.name == "category"
    assert transformed_test_df["B"].dtype.name == "category"
    assert list(transformed_test_df["A"].cat.categories) == ["a", "b", "c"]
    assert list(transformed_test_df["B"].cat.categories) == ["x", "y", "z"]
    assert transformed_test_df["A"].isnull().tolist() == [False, False, False, True]
    assert transformed_test_df["B"].isnull().tolist() == [False, False, False, True]

    # Check that the category codes are consistent between the training and test sets
    # Expected codes: a=0, b=1, c=2, unseen=-1
    assert transformed_train_df["A"].cat.codes.tolist() == [0, 2, 1, 0]
    assert transformed_test_df["A"].cat.codes.tolist() == [0, 1, 2, -1]


def test_transform_with_unseen_data_and_unseen_is_raise():
    """
    Test transforming the dataframe with unseen data.
    """
    df_train = pd.DataFrame({"A": ["a", "c", "b", "a"], "B": ["x", "y", "x", "z"]})
    df_test = pd.DataFrame(
        {"A": ["a", "b", "c", "unseen"], "B": ["x", "y", "z", "unseen"]}
    )
    transformer = PandasCategoricalEncoder(variables=["A", "B"], unseen="raise")
    msg = "During the encoding, NaN values were introduced in the feature(s) A, B."

    transformer.fit_transform(df_train)
    with pytest.raises(ValueError) as record:
        transformer.transform(df_test)

    assert str(record.value) == msg


def test_fit_raises_error_if_df_contains_na():
    """
    Test that the transform method raises an error if the dataframe contains missing
    values.
    """
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    df.loc[2, "A"] = None
    transformer = PandasCategoricalEncoder(variables=["A", "B"])

    with pytest.raises(ValueError) as record:
        transformer.fit(df)

    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


def test_transform_raises_error_if_df_contains_na():
    """
    Test that the transform method raises an error if the dataframe contains missing
    values.
    """
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    transformer = PandasCategoricalEncoder(variables=["A", "B"])
    transformer.fit(df)

    df.loc[2, "A"] = None

    with pytest.raises(ValueError) as record:
        transformer.transform(df)

    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    assert str(record.value) == msg


def test_arbitrary_encoding_automatically_find_variables_ignore_format():
    """
    Test the ignore_format parameter.
    """
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": [1, 2, 1, 3]})
    transformer = PandasCategoricalEncoder(ignore_format=True)
    transformer.fit(df)
    transformed_df = transformer.transform(df)

    assert transformer.variables_ == ["A", "B"]
    assert transformer.encoder_dict_ == {
        "A": {"a": 0, "b": 1, "c": 2},
        "B": {1: 0, 2: 1, 3: 2},
    }
    assert transformed_df["A"].dtype.name == "category"
    assert transformed_df["B"].dtype.name == "category"
    assert list(transformed_df["A"].cat.categories) == ["a", "b", "c"]
    assert list(transformed_df["B"].cat.categories) == [1, 2, 3]
    assert transformed_df["A"].cat.codes.tolist() == [0, 1, 0, 2]
    assert transformed_df["B"].cat.codes.tolist() == [0, 1, 0, 2]


def test_ordered_encoding_1_variable_ignore_format():
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": [1, 2, 1, 3]})
    transformer = PandasCategoricalEncoder(ignore_format=True, variables=["A"])
    transformer.fit(df)
    transformed_df = transformer.transform(df)

    assert transformer.variables_ == ["A"]
    assert transformer.encoder_dict_ == {"A": {"a": 0, "b": 1, "c": 2}}
    assert transformed_df["A"].dtype.name == "category"
    assert list(transformed_df["A"].cat.categories) == ["a", "b", "c"]
    assert transformed_df["A"].cat.codes.tolist() == [0, 1, 0, 2]


def test_transform_with_missing_values():
    """
    Test transforming the dataframe with missing values.
    """
    df = pd.DataFrame({"A": ["a", "b", None, "c"], "B": ["x", None, "x", "z"]})
    transformer = PandasCategoricalEncoder(
        variables=["A", "B"], missing_values="ignore"
    )
    transformer.fit(df)
    transformed_df = transformer.transform(df)

    assert transformed_df["A"].dtype.name == "category"
    assert transformed_df["B"].dtype.name == "category"
    assert list(transformed_df["A"].cat.categories) == ["a", "b", "c"]
    assert list(transformed_df["B"].cat.categories) == ["x", "z"]
    assert transformed_df["A"].isnull().sum() == 1
    assert transformed_df["B"].isnull().sum() == 1
    assert transformer.variables_ == ["A", "B"]
    assert transformer.encoder_dict_ == {
        "A": {"a": 0, "b": 1, "c": 2},
        "B": {"x": 0, "z": 1},
    }


def test_fit_transform():
    """
    Test the fit_transform method.
    """
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    transformer = PandasCategoricalEncoder(variables=["A", "B"])
    transformed_df = transformer.fit_transform(df)

    assert transformed_df["A"].dtype.name == "category"
    assert transformed_df["B"].dtype.name == "category"
    assert list(transformed_df["A"].cat.categories) == ["a", "b", "c"]
    assert list(transformed_df["B"].cat.categories) == ["x", "y", "z"]
    assert transformed_df["A"].cat.codes.tolist() == [0, 1, 0, 2]
    assert transformed_df["B"].cat.codes.tolist() == [0, 1, 0, 2]
    assert transformer.variables_ == ["A", "B"]
    assert transformer.encoder_dict_ == {
        "A": {"a": 0, "b": 1, "c": 2},
        "B": {"x": 0, "y": 1, "z": 2},
    }


@pytest.mark.parametrize(
    "unseen", ["pizza", "encode", False, 1, ("raise", "ignore"), ["ignore"]]
)
def test_error_if_unseen_not_permitted_value(unseen):
    with pytest.raises(ValueError):
        PandasCategoricalEncoder(unseen=unseen)


def test_inverse_transform():
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    transformer = PandasCategoricalEncoder(variables=["A", "B"])
    transformed_df = transformer.fit_transform(df)
    inverse_df = transformer.inverse_transform(transformed_df)

    pd.testing.assert_frame_equal(df, inverse_df)


def test_inverse_transform_when_no_unseen():
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    transformer = PandasCategoricalEncoder(variables=["A", "B"])
    transformer.fit(df)
    transformed_df = transformer.transform(df)
    inverse_df = transformer.inverse_transform(transformed_df)

    pd.testing.assert_frame_equal(df, inverse_df)


def test_inverse_transform_when_ignore_unseen():
    df1 = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    df2 = pd.DataFrame({"A": ["a", "b", "d", "c"], "B": ["x", "y", "x", "w"]})
    transformer = PandasCategoricalEncoder(variables=["A", "B"], unseen="ignore")
    transformer.fit(df1)
    transformed_df = transformer.transform(df2)

    inverse_df = transformer.inverse_transform(transformed_df)
    expected_df = pd.DataFrame({"A": ["a", "b", None, "c"], "B": ["x", "y", "x", None]})
    pd.testing.assert_frame_equal(inverse_df, expected_df)


def test_inverse_transform_raises_non_fitted_error():
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    transformer = PandasCategoricalEncoder(variables=["A", "B"])

    # Test when fit is not called prior to transform.
    with pytest.raises(NotFittedError):
        transformer.inverse_transform(df)
