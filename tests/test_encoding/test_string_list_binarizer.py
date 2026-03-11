import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.encoding import StringListBinarizer

def test_string_list_binarizer_delimited_strings():
    df = pd.DataFrame(
        {
            "tags": ["action, comedy", "comedy", "action, thriller"],
            "other": [1, 2, 3],
        }
    )

    expected_df = pd.DataFrame(
        {
            "other": [1, 2, 3],
            "tags_action": [1, 0, 1],
            "tags_comedy": [1, 1, 0],
            "tags_thriller": [0, 0, 1],
        }
    )

    encoder = StringListBinarizer(variables=["tags"], separator=",")
    X = encoder.fit_transform(df)

    assert encoder.variables_ == ["tags"]
    assert encoder.encoder_dict_ == {"tags": ["action", "comedy", "thriller"]}
    pd.testing.assert_frame_equal(X, expected_df)


def test_string_list_binarizer_python_lists():
    df = pd.DataFrame(
        {
            "tags": [["action", "comedy"], ["comedy"], ["action", "thriller"]],
            "other": [1, 2, 3],
        }
    )

    expected_df = pd.DataFrame(
        {
            "other": [1, 2, 3],
            "tags_action": [1, 0, 1],
            "tags_comedy": [1, 1, 0],
            "tags_thriller": [0, 0, 1],
        }
    )

    encoder = StringListBinarizer(variables=["tags"])
    X = encoder.fit_transform(df)

    assert encoder.variables_ == ["tags"]
    assert encoder.encoder_dict_ == {"tags": ["action", "comedy", "thriller"]}
    pd.testing.assert_frame_equal(X, expected_df)


def test_find_categorical_variables():
    df = pd.DataFrame({"tags": ["A,B", "C"], "num": [1, 2]})

    encoder = StringListBinarizer(variables=None, separator=",")
    encoder.fit(df)

    assert encoder.variables_ == ["tags"]


def test_ignore_format():
    df = pd.DataFrame(
        {
            "tags": ["A,B", "C"],
            "num": ["1", "2"],  # Treated as object but maybe we want to encode it
        }
    )

    encoder = StringListBinarizer(variables=["num"], ignore_format=True)
    encoder.fit(df)

    assert encoder.variables_ == ["num"]
    assert encoder.encoder_dict_ == {"num": ["1", "2"]}


def test_error_if_not_categorical():
    df = pd.DataFrame({"num": [1, 2]})
    encoder = StringListBinarizer(variables=["num"])
    with pytest.raises(TypeError):
        encoder.fit(df)


def test_missing_values_error():
    df = pd.DataFrame({"tags": ["A,B", float("nan")]})
    encoder = StringListBinarizer(variables=["tags"])
    with pytest.raises(ValueError):
        encoder.fit(df)


def test_not_fitted_error():
    df = pd.DataFrame({"tags": ["A,B"]})
    encoder = StringListBinarizer()
    with pytest.raises(NotFittedError):
        encoder.transform(df)


def test_unseen_categories():
    df_train = pd.DataFrame({"tags": ["A,B", "C"]})
    df_test = pd.DataFrame({"tags": ["A,D", "B,C,E"]})

    encoder = StringListBinarizer(variables=["tags"], separator=",")
    encoder.fit(df_train)
    X = encoder.transform(df_test)

    # Expect D and E to be ignored (columns for A, B, C only)
    expected_df = pd.DataFrame(
        {"tags_A": [1, 0], "tags_B": [0, 1], "tags_C": [0, 1]}
    )

    pd.testing.assert_frame_equal(X, expected_df)
