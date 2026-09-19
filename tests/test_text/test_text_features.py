import re
from datetime import datetime
from types import SimpleNamespace

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.text import TextFeatures, text_features
from feature_engine.text.text_features import TEXT_FEATURES
from tests.backend_helpers import frame_to_dict, make_series

TEXT = [
    "Hello World!",
    "HELLO",
    "12345",
    "e.g. i.e.",
    "   ",
    " trailing ",
    "abc...",
    "",
    None,
    "A? B! C.",
    "HeLLo",
    "Hi! @#",
    "A1b2 C3d4!@#$",
    "???",
    "i.e., this is wrong",
    "Is 1 > 2? No, 100%!",
    "Hello. World",
    "Hello. World.",
    "Hello... World!?!",
    "This is a proper sentence containing "
    "supercalifragilisticexpialidocious and exceptionally long words.",
]

# non-ASCII letters and digits, non-breaking space (\xa0), file separator (\x1c),
# new lines around the final punctuation and a Greek final sigma
TEXT_EDGE_CASES = [
    "",
    None,
    "   ",
    "Hello World!",
    "HELLO",
    "\N{LATIN CAPITAL LETTER E WITH ACUTE}COLE "
    "na\N{LATIN SMALL LETTER I WITH DIAERESIS}ve 123",
    "\N{ARABIC-INDIC DIGIT THREE} digits",
    "a\xa0b\x1cc",
    "x.\n",
    "x.\n\n",
    "a\nb.",
    "Dog dog DOG",
    "\N{GREEK CAPITAL LETTER OMICRON}\N{GREEK CAPITAL LETTER DELTA}"
    "\N{GREEK CAPITAL LETTER OMICRON}\N{GREEK CAPITAL LETTER SIGMA} "
    "\N{GREEK SMALL LETTER OMICRON}\N{GREEK SMALL LETTER DELTA}"
    "\N{GREEK SMALL LETTER OMICRON}\N{GREEK SMALL LETTER FINAL SIGMA}",
    "Is 1 > 2? No, 100%!",
]

EXPECTED = {
    "char_count": [11, 5, 5, 8, 0, 8, 6, 0, 0, 6, 5, 5, 12, 3, 16, 14, 11, 12, 16, 91],
    "word_count": [2, 1, 1, 2, 0, 1, 1, 0, 0, 3, 1, 2, 2, 1, 4, 6, 2, 2, 2, 11],
    "sentence_count": [1, 0, 0, 4, 0, 0, 1, 0, 0, 3, 0, 1, 1, 1, 2, 2, 1, 2, 2, 1],
    "avg_word_length": [
        6, 5, 5, 9 / 2, 0, 8, 6, 0, 0, 8 / 3, 5, 3, 13 / 2, 3, 19 / 4, 19 / 6, 6,
        13 / 2, 17 / 2, 101 / 11,
    ],
    "digit_count": [0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 5, 0, 0, 0, 0],
    "letter_count": [10, 5, 0, 4, 0, 8, 3, 0, 0, 3, 5, 2, 4, 0, 13, 4, 10, 10, 10, 90],
    "uppercase_count": [2, 5, 0, 0, 0, 0, 0, 0, 0, 3, 3, 1, 2, 0, 0, 2, 2, 2, 2, 1],
    "lowercase_count": [8, 0, 0, 4, 0, 8, 3, 0, 0, 0, 2, 1, 2, 0, 13, 2, 8, 8, 8, 89],
    "special_char_count": [1, 0, 0, 4, 0, 0, 3, 0, 0, 3, 0, 3, 4, 3, 3, 5, 1, 2, 6, 1],
    "whitespace_count": [1, 0, 0, 1, 3, 2, 0, 0, 0, 2, 0, 1, 1, 0, 3, 5, 1, 1, 1, 10],
    "whitespace_ratio": [
        1 / 12, 0, 0, 1 / 9, 1, 2 / 10, 0, 0, 0, 2 / 8, 0, 1 / 6, 1 / 13, 0, 3 / 19,
        5 / 19, 1 / 12, 1 / 13, 1 / 17, 10 / 101,
    ],
    "digit_ratio": [
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4 / 12, 0, 0, 5 / 14, 0, 0, 0, 0,
    ],
    "uppercase_ratio": [
        2 / 11, 1, 0, 0, 0, 0, 0, 0, 0, 3 / 6, 3 / 5, 1 / 5, 2 / 12, 0, 0, 2 / 14,
        2 / 11, 2 / 12, 2 / 16, 1 / 91,
    ],
    "has_digits": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    "has_uppercase": [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1],
    "is_empty": [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "starts_with_uppercase": [
        1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1,
    ],
    "ends_with_punctuation": [
        1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1,
    ],
    "unique_word_count": [2, 1, 1, 2, 0, 1, 1, 0, 0, 3, 1, 2, 2, 1, 4, 6, 2, 2, 2, 11],
    "lexical_diversity": [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
}

EXPECTED_EDGE_CASES = {
    "char_count": [0, 0, 0, 11, 5, 13, 7, 3, 2, 2, 3, 9, 8, 14],
    "word_count": [0, 0, 0, 2, 1, 3, 2, 3, 1, 1, 2, 3, 2, 6],
    "sentence_count": [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 2],
    "avg_word_length": [0, 0, 0, 6, 5, 5, 4, 5 / 3, 2, 2, 2, 11 / 3, 9 / 2, 19 / 6],
    "digit_count": [0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 0, 0, 5],
    "letter_count": [0, 0, 0, 10, 5, 8, 6, 3, 1, 1, 2, 9, 0, 4],
    "uppercase_count": [0, 0, 0, 2, 5, 4, 0, 0, 0, 0, 0, 4, 0, 2],
    "lowercase_count": [0, 0, 0, 8, 0, 4, 6, 3, 1, 1, 2, 5, 0, 2],
    "special_char_count": [0, 0, 0, 1, 0, 2, 1, 0, 1, 1, 1, 0, 8, 5],
    "whitespace_count": [0, 0, 3, 1, 0, 2, 1, 2, 1, 2, 1, 2, 1, 5],
    "whitespace_ratio": [
        0, 0, 1, 1 / 12, 0, 2 / 15, 1 / 8, 2 / 5, 1 / 3, 2 / 4, 1 / 4, 2 / 11, 1 / 9,
        5 / 19,
    ],
    "digit_ratio": [0, 0, 0, 0, 0, 3 / 13, 1 / 7, 0, 0, 0, 0, 0, 0, 5 / 14],
    "uppercase_ratio": [0, 0, 0, 2 / 11, 1, 4 / 13, 0, 0, 0, 0, 0, 4 / 9, 0, 2 / 14],
    "has_digits": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
    "has_uppercase": [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    "is_empty": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "starts_with_uppercase": [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    "ends_with_punctuation": [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    "unique_word_count": [0, 0, 0, 2, 1, 3, 2, 3, 1, 1, 2, 1, 1, 6],
    "lexical_diversity": [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 / 3, 1 / 2, 1],
}


# init parameters
@pytest.mark.parametrize(
    "variables", [123, True, None, [1, 2], ["text", 123], ("text",), {"text": 1}]
)
def test_error_if_variables_not_string_or_list_of_strings(variables):
    msg = f"variables must be a string or a list of strings. Got {variables} instead."
    with pytest.raises(ValueError, match=re.escape(msg)):
        TextFeatures(variables=variables)


@pytest.mark.parametrize(
    "features",
    [
        "char_count",
        123,
        True,
        ("char_count",),
        {"char_count": 1},
        [1, 2],
        ["char_count", True],
        ["invalid_feature"],
        ["char_count", "invalid_feature"],
    ],
)
def test_error_if_features_not_permitted(features):
    msg = (
        f"features must be None or a list with any of {list(TEXT_FEATURES.keys())}. "
        f"Got {features} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        TextFeatures(variables=["text"], features=features)


@pytest.mark.parametrize("missing_values", ["empanada", True, 1, None, ["raise"]])
def test_error_if_missing_values_not_permitted(missing_values):
    msg = (
        "missing_values takes only values 'raise' or 'ignore'. "
        f"Got {missing_values} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        TextFeatures(variables=["text"], missing_values=missing_values)


@pytest.mark.parametrize("drop_original", ["True", 1, None, [True]])
def test_error_if_drop_original_not_bool(drop_original):
    msg = (
        "drop_original takes only boolean values True and False. "
        f"Got {drop_original} instead."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        TextFeatures(variables=["text"], drop_original=drop_original)


@pytest.mark.parametrize(
    "features, missing_values, drop_original",
    [
        (None, "ignore", False),
        (["char_count"], "raise", True),
        (["word_count", "lexical_diversity"], "ignore", True),
    ],
)
def test_init_param_assignment(features, missing_values, drop_original):
    transformer = TextFeatures(
        variables=["text"],
        features=features,
        missing_values=missing_values,
        drop_original=drop_original,
    )
    assert transformer.features == features
    assert transformer.missing_values == missing_values
    assert transformer.drop_original is drop_original


# fit and transform
@pytest.mark.parametrize(
    "variables, features, variables_, features_",
    [
        ("text", None, ["text"], list(TEXT_FEATURES.keys())),
        (["string"], ["char_count"], ["string"], ["char_count"]),
        (["text", "string"], ["word_count"], ["text", "string"], ["word_count"]),
    ],
)
def test_fit_attributes(make_df, variables, features, variables_, features_):
    X = make_df({"text": ["Hello"], "string": ["Bye"], "number": [1]})
    transformer = TextFeatures(variables=variables, features=features).fit(X)

    assert transformer.variables_ == variables_
    assert transformer.features_ == features_
    assert transformer.feature_names_in_ == ["text", "string", "number"]
    assert transformer.n_features_in_ == 3


@pytest.mark.parametrize("target", ["series", "list", "array"])
def test_fit_ignores_the_target(make_df, target):
    X = make_df({"text": ["Hello", "World"]})
    y = {
        "series": make_series(make_df, [0, 1]),
        "list": [0, 1],
        "array": np.array([0, 1]),
    }[target]
    transformer = TextFeatures(variables=["text"], features=["char_count"])
    Xt = transformer.fit(X, y).transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"text": ["Hello", "World"], "text_char_count": [5, 5]}


def test_error_if_variable_not_in_dataframe(make_df):
    X = make_df({"text": ["Hello"]})
    transformer = TextFeatures(variables=["nonexistent"])
    msg = "Variables {'nonexistent'} are not present in the dataframe."
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.fit(X)


@pytest.mark.parametrize(
    "variable, values",
    [
        ("Age", [20, 21]),
        ("Marks", [0.9, 0.8]),
        ("dob", [datetime(2020, 2, 24), datetime(2020, 2, 25)]),
    ],
)
def test_error_if_variable_not_text(make_df, variable, values):
    X = make_df({"Name": ["tom", "nick"], variable: values})
    transformer = TextFeatures(variables=["Name", variable])
    msg = (
        f"Variables ['{variable}'] are not object or string. "
        "Please provide text variables only."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.fit(X)


def test_categorical_variables_with_string_categories(make_df):
    X = make_df({"text": ["Hello World", "Hi", "Hello World"]})
    X = nw.from_native(X).with_columns(nw.col("text").cast(nw.Categorical))
    transformer = TextFeatures(variables=["text"], features=["word_count"])
    Xt = transformer.fit_transform(X.to_native())

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "text": ["Hello World", "Hi", "Hello World"],
        "text_word_count": [2, 1, 2],
    }


def test_error_if_categories_are_not_strings():
    # polars categories are always strings
    X = pd.DataFrame({"text": pd.Series([1, 2], dtype="category")})
    transformer = TextFeatures(variables=["text"])
    msg = (
        "Variables ['text'] are not object or string. "
        "Please provide text variables only."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.fit(X)


def test_error_if_missing_values_in_fit(make_df):
    X = make_df({"text": ["Hello", None, "World"]})
    transformer = TextFeatures(variables=["text"], missing_values="raise")
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.fit(X)


def test_error_if_missing_values_in_transform(make_df):
    transformer = TextFeatures(variables=["text"], missing_values="raise")
    transformer.fit(make_df({"text": ["Hello", "World"]}))
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.transform(make_df({"text": ["Hello", None, "World"]}))


def test_missing_values_are_treated_as_empty_strings(make_df):
    X = make_df({"text": ["Hello", None, "World"]})
    transformer = TextFeatures(variables=["text"], features=["char_count"])
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "text": ["Hello", "", "World"],
        "text_char_count": [5, 0, 5],
    }
    assert frame_to_dict(X) == {"text": ["Hello", None, "World"]}


def test_missing_values_raise_returns_same_values(make_df):
    X = make_df({"text": ["Hello World", "Hi"]})
    transformer = TextFeatures(
        variables=["text"], features=["word_count"], missing_values="raise"
    )
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "text": ["Hello World", "Hi"],
        "text_word_count": [2, 1],
    }


def test_error_if_not_fitted(make_df):
    transformer = TextFeatures(variables=["text"])
    msg = (
        "This TextFeatures instance is not fitted yet. Call 'fit' with "
        "appropriate arguments before using this estimator."
    )
    with pytest.raises(NotFittedError, match=re.escape(msg)):
        transformer.transform(make_df({"text": ["Hello"]}))


def test_error_if_transform_gets_different_number_of_columns(make_df):
    transformer = TextFeatures(variables=["text"]).fit(make_df({"text": ["Hello"]}))
    msg = (
        "The number of columns in this dataset is different from the one used to "
        "fit this transformer (when using the fit() method)."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        transformer.transform(make_df({"text": ["Hello"], "other": [1]}))


def test_transform_on_new_data(make_df):
    transformer = TextFeatures(
        variables=["text"], features=["char_count", "has_digits"]
    )
    transformer.fit(make_df({"text": ["Hello World", "Foo Bar"]}))
    Xt = transformer.transform(make_df({"text": ["New Data", "Test 123"]}))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "text": ["New Data", "Test 123"],
        "text_char_count": [7, 7],
        "text_has_digits": [0, 1],
    }


def test_transform_reorders_columns_as_in_fit(make_df):
    transformer = TextFeatures(variables=["text"], features=["char_count"])
    transformer.fit(make_df({"text": ["Hello"], "other": [1]}))
    Xt = transformer.transform(make_df({"other": [2], "text": ["Hi"]}))

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"text": ["Hi"], "other": [2], "text_char_count": [2]}


def test_default_extracts_all_features(make_df):
    X = make_df({"text": ["Hello World!", "Python 123", "AI"]})
    Xt = TextFeatures(variables=["text"]).fit_transform(X)

    assert isinstance(Xt, make_df)
    assert list(Xt.columns) == ["text"] + [f"text_{f}" for f in TEXT_FEATURES]


def test_only_selected_variables_and_features_are_added(make_df):
    X = make_df({"a": ["Hello", "World"], "b": ["Foo", "Bar"], "numeric": [1, 2]})
    transformer = TextFeatures(
        variables=["b", "a"], features=["word_count", "is_empty"]
    )
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {
        "a": ["Hello", "World"],
        "b": ["Foo", "Bar"],
        "numeric": [1, 2],
        "b_word_count": [1, 1],
        "b_is_empty": [0, 0],
        "a_word_count": [1, 1],
        "a_is_empty": [0, 0],
    }


def test_drop_original(make_df):
    X = make_df({"text": ["Hello", "World"], "other": [1, 2]})
    transformer = TextFeatures(
        variables=["text"], features=["char_count"], drop_original=True
    )
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt) == {"other": [1, 2], "text_char_count": [5, 5]}


@pytest.mark.parametrize("feature", list(TEXT_FEATURES.keys()))
def test_feature_values(make_df, feature):
    X = make_df({"text": TEXT})
    Xt = TextFeatures(variables=["text"], features=[feature]).fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt)[f"text_{feature}"] == pytest.approx(EXPECTED[feature])


@pytest.mark.parametrize("feature", list(TEXT_FEATURES.keys()))
def test_feature_values_on_edge_cases(make_df, feature):
    X = make_df({"text": TEXT_EDGE_CASES})
    Xt = TextFeatures(variables=["text"], features=[feature]).fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt)[f"text_{feature}"] == pytest.approx(
        EXPECTED_EDGE_CASES[feature]
    )


@pytest.mark.parametrize("feature", list(TEXT_FEATURES.keys()))
def test_feature_values_on_other_backends(monkeypatch, feature):
    # makes polars take the path used by backends other than pandas and polars
    backend_checks = SimpleNamespace(
        is_pandas_dataframe=lambda X: False, is_polars_dataframe=lambda X: False
    )
    monkeypatch.setattr(text_features, "nwd", backend_checks)
    X = pl.DataFrame({"text": TEXT_EDGE_CASES})
    Xt = TextFeatures(variables=["text"], features=[feature]).fit_transform(X)

    assert isinstance(Xt, pl.DataFrame)
    assert frame_to_dict(Xt)[f"text_{feature}"] == pytest.approx(
        EXPECTED_EDGE_CASES[feature]
    )


def test_lexical_diversity_is_unique_words_over_total_words(make_df):
    X = make_df(
        {
            "text": [
                "the cat sat on the mat",  # 6 words, 5 unique
                "good good good good",  # 4 words, 1 unique
                "all words here are distinct",  # 5 words, 5 unique
            ]
        }
    )
    transformer = TextFeatures(variables=["text"], features=["lexical_diversity"])
    Xt = transformer.fit_transform(X)

    assert isinstance(Xt, make_df)
    assert frame_to_dict(Xt)["text_lexical_diversity"] == pytest.approx(
        [5 / 6, 1 / 4, 1]
    )


def test_output_dtypes(make_df):
    X = make_df({"text": ["Hello World", "Hi"]})
    features = ["char_count", "whitespace_ratio", "has_digits", "unique_word_count"]
    Xt = TextFeatures(variables=["text"], features=features).fit_transform(X)

    schema = nw.from_native(Xt).schema
    assert [schema[f"text_{f}"] for f in features] == [
        nw.Int64,
        nw.Float64,
        nw.Int64,
        nw.Int64,
    ]


@pytest.mark.parametrize(
    "drop_original, expected",
    [
        (False, ["text", "other", "text_char_count", "text_word_count"]),
        (True, ["other", "text_char_count", "text_word_count"]),
    ],
)
def test_get_feature_names_out(make_df, drop_original, expected):
    X = make_df({"text": ["Hello"], "other": [1]})
    transformer = TextFeatures(
        variables=["text"],
        features=["char_count", "word_count"],
        drop_original=drop_original,
    )
    Xt = transformer.fit_transform(X)

    assert transformer.get_feature_names_out() == expected
    assert list(Xt.columns) == expected


def test_integer_column_names():
    X = pd.DataFrame({0: [1, 2], "text": ["Hello World", None], 1: ["a", "b"]})
    transformer = TextFeatures(variables=["text"], features=["word_count"])
    Xt = transformer.fit_transform(X)

    expected = pd.DataFrame(
        {
            0: [1, 2],
            "text": ["Hello World", ""],
            1: ["a", "b"],
            "text_word_count": [2, 0],
        }
    )
    pd.testing.assert_frame_equal(Xt, expected)
    assert transformer.get_feature_names_out() == [0, "text", 1, "text_word_count"]


def test_pandas_index_is_kept():
    X = pd.DataFrame({"text": ["Hello World", "Hi", "Hey"]}, index=[10, 10, 3])
    transformer = TextFeatures(
        variables=["text"], features=["char_count", "unique_word_count"]
    )
    Xt = transformer.fit_transform(X)

    expected = pd.DataFrame(
        {
            "text": ["Hello World", "Hi", "Hey"],
            "text_char_count": [10, 2, 3],
            "text_unique_word_count": [2, 1, 1],
        },
        index=[10, 10, 3],
    )
    pd.testing.assert_frame_equal(Xt, expected)
