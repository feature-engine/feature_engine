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


def test_init_separator_not_str():
    with pytest.raises(ValueError, match="separator takes only strings"):
        StringListBinarizer(variables=["tags"], separator=123)


def test_init_ignore_format_not_bool():
    with pytest.raises(ValueError, match="ignore_format takes only booleans"):
        StringListBinarizer(variables=["tags"], ignore_format="yes")


def test_ignore_format_true_variables_none():
    """Fit with ignore_format=True and variables=None uses find_all_variables."""
    df = pd.DataFrame(
        {"tags": ["a,b", "c"], "num": [1, 2], "other": ["x", "y"]}
    )
    encoder = StringListBinarizer(separator=",", ignore_format=True)
    encoder.fit(df)
    assert set(encoder.variables_) == {"tags", "num", "other"}
    X = encoder.transform(df)
    assert list(X.columns) == encoder.get_feature_names_out()


def test_no_categorical_variables_raises():
    """Raise when variables=None and no object/category/string columns."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    encoder = StringListBinarizer(variables=None)
    with pytest.raises(ValueError, match="No categorical variables found"):
        encoder.fit(df)


def test_fit_row_not_str_or_list():
    """Fit with a row that is neither str nor list (e.g. number) uses else branch."""
    df = pd.DataFrame({"tags": ["A,B", 42]})
    encoder = StringListBinarizer(variables=["tags"], separator=",")
    encoder.fit(df)
    assert "A" in encoder.encoder_dict_["tags"]
    assert "B" in encoder.encoder_dict_["tags"]
    assert "42" in encoder.encoder_dict_["tags"]


def test_transform_row_not_str_or_list():
    """Transform with non-str non-list row uses else branch."""
    df_train = pd.DataFrame({"tags": ["A", "B"]})
    encoder = StringListBinarizer(variables=["tags"])
    encoder.fit(df_train)
    df_test = pd.DataFrame({"tags": [123]})
    X = encoder.transform(df_test)
    assert "tags_A" in X.columns
    assert "tags_B" in X.columns


def test_get_feature_names_out():
    """get_feature_names_out returns binarized feature names in order."""
    df = pd.DataFrame(
        {"x": [1, 2], "tags": ["a,b", "c"], "y": [3, 4]}
    )
    encoder = StringListBinarizer(variables=["tags"], separator=",")
    encoder.fit(df)
    names = encoder.get_feature_names_out()
    assert names == ["x", "y", "tags_a", "tags_b", "tags_c"]


def test_more_tags():
    """_more_tags returns expected sklearn config."""
    encoder = StringListBinarizer(variables=["tags"])
    tags = encoder._more_tags()
    assert tags["variables"] == "categorical"
    assert "check_estimators_nan_inf" in tags["_xfail_checks"]
