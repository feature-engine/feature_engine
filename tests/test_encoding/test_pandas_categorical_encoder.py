import pandas as pd

from feature_engine.encoding import PandasCategoricalEncoder

def test_fit_with_specified_variables():
    """
    Test fitting the transformer with specified variables.
    """
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    transformer = PandasCategoricalEncoder(variables=["A"])
    transformer.fit(df)

    assert transformer.variables == ["A"]
    assert transformer.encoder_dict_ == {"A": ["a", "b", "c"]}


def test_fit_with_all_object_variables():
    """
    Test fitting the transformer with all object variables.
    """
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    transformer = PandasCategoricalEncoder()
    transformer.fit(df)

    assert transformer.variables == ["A", "B"]
    assert transformer.encoder_dict_ == {"A": ["a", "b", "c"], "B": ["x", "y", "z"]}


def test_transform():
    """
    Test transforming the dataframe with the fitted transformer.
    """
    df = pd.DataFrame({"A": ["a", "b", "a", "c"], "B": ["x", "y", "x", "z"]})
    transformer = PandasCategoricalEncoder(variables=["A", "B"])
    transformer.fit(df)
    transformed_df = transformer.transform(df)

    assert transformed_df["A"].dtype.name == "category"
    assert transformed_df["B"].dtype.name == "category"
    assert list(transformed_df["A"].cat.categories) == ["a", "b", "c"]
    assert list(transformed_df["B"].cat.categories) == ["x", "y", "z"]


def test_transform_with_unseen_data():
    """
    Test transforming the dataframe with unseen data.
    """
    df_train = pd.DataFrame({"A": ["a", "c", "b", "a"], "B": ["x", "y", "x", "z"]})
    df_test = pd.DataFrame({"A": ["a", "b", "c", "d"], "B": ["x", "y", "z", "w"]})
    transformer = PandasCategoricalEncoder(variables=["A", "B"])
    transformed_train_df = transformer.fit_transform(df_train)
    transformed_test_df = transformer.transform(df_test)

    assert transformed_test_df["A"].dtype.name == "category"
    assert transformed_test_df["B"].dtype.name == "category"
    assert list(transformed_test_df["A"].cat.categories) == ["a", "b", "c"]
    assert list(transformed_test_df["B"].cat.categories) == ["x", "y", "z"]
    assert transformed_test_df["A"].isnull().tolist() == [False, False, False, True]
    assert transformed_test_df["B"].isnull().tolist() == [False, False, False, True]

    # Check that the category codes are consistent between the training and test sets
    # Expected codes: a=0, b=1, c=2, d=-1
    assert transformed_train_df["A"].cat.codes.tolist() == [0, 2, 1, 0]
    assert transformed_test_df["A"].cat.codes.tolist() == [0, 1, 2, -1]


def test_transform_with_missing_values():
    """
    Test transforming the dataframe with missing values.
    """
    df = pd.DataFrame({"A": ["a", "b", None, "c"], "B": ["x", None, "x", "z"]})
    transformer = PandasCategoricalEncoder(variables=["A", "B"])
    transformer.fit(df)
    transformed_df = transformer.transform(df)

    assert transformed_df["A"].dtype.name == "category"
    assert transformed_df["B"].dtype.name == "category"
    assert list(transformed_df["A"].cat.categories) == ["a", "b", "c"]
    assert list(transformed_df["B"].cat.categories) == ["x", "z"]
    assert transformed_df["A"].isnull().sum() == 1
    assert transformed_df["B"].isnull().sum() == 1


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

