import pandas as pd
import pytest

from feature_engine.text import TextFeatures
from feature_engine.text.text_features import TEXT_FEATURES

# ==============================================================================
# INIT TESTS
# ==============================================================================


@pytest.mark.parametrize(
    "invalid_variables",
    [
        123,
        True,
        [1, 2],
        ["text", 123],
        {"text": 1},
    ],
)
def test_invalid_variables_raises_error(invalid_variables):
    with pytest.raises(ValueError, match="variables must be a string or a list of"):
        TextFeatures(variables=invalid_variables)


@pytest.mark.parametrize(
    "invalid_features, err_msg",
    [
        ("some_string", "features must be"),
        ([1, 2], "features must be"),
        (123, "features must be"),
        (True, "features must be"),
        (["some_string", True], "features must be"),
        ({"some_string": 1}, "features must be"),
        (["invalid_feature"], "Invalid features"),
        (["char_count", "invalid_feature"], "Invalid features"),
    ],
)
def test_invalid_features_raises_error(invalid_features, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        TextFeatures(variables=["text"], features=invalid_features)


# ==============================================================================
# FIT TESTS
# ==============================================================================


@pytest.mark.parametrize(
    "variables, features",
    [
        ("text", None),
        (["string"], ["char_count"]),
        (["text", "string"], ["sentence_count", "avg_word_length"]),
    ],
)
def test_fit_stores_attributes(variables, features):
    X = pd.DataFrame({"text": ["Hello"], "string": ["Bye"]})
    transformer = TextFeatures(variables=variables, features=features)
    transformer.fit(X)

    assert (
        transformer.variables_ == variables
        if isinstance(variables, list)
        else transformer.variables_ == [variables]
    )
    assert (
        transformer.features_ == list(TEXT_FEATURES.keys())
        if features is None
        else transformer.features_ == features
    )
    assert transformer.feature_names_in_ == ["text", "string"]
    assert transformer.n_features_in_ == 2


def test_missing_variable_raises_error():
    X = pd.DataFrame({"text": ["Hello"]})
    transformer = TextFeatures(variables=["nonexistent"])
    with pytest.raises(ValueError, match="not present in the dataframe"):
        transformer.fit(X)


@pytest.mark.parametrize("variables", ["Age","Marks", "dob"])
def test_no_text_columns_raises_error(df_vartypes, variables):
    transformer = TextFeatures(variables=variables)
    with pytest.raises(ValueError, match="not object or string"):
        transformer.fit(df_vartypes)


def test_nan_handling_raise_error_fit(df_na):
    transformer = TextFeatures(
        variables=["City"], features=["char_count"], missing_values="raise"
    )
    msg = "`missing_values='ignore'` when initialising this transformer"
    with pytest.raises(ValueError, match=msg):
        transformer.fit(df_na)


# ==============================================================================
# TRANSFORM TESTS - GENERAL
# ==============================================================================


def test_transform_on_new_data():
    """Test transform works on new data after fit."""
    X_train = pd.DataFrame({"text": ["Hello World", "Foo Bar"]})
    X_test = pd.DataFrame({"text": ["New Data", "Test 123"]})

    transformer = TextFeatures(
        variables=["text"], features=["char_count", "has_digits"]
    )
    transformer.fit(X_train)
    X_tr = transformer.transform(X_test)

    assert X_tr["text_char_count"].tolist() == [7, 7]
    assert X_tr["text_has_digits"].tolist() == [0, 1]


def test_nan_handling_raise_error_transform():
    """Test handling of NaN values when missing_values is 'raise' on transform."""
    X_train = pd.DataFrame({"text": ["Hello", "World"]})
    X_test = pd.DataFrame({"text": ["Hello", None, "World"]})
    transformer = TextFeatures(
        variables=["text"], features=["char_count"], missing_values="raise"
    )
    transformer.fit(X_train)
    with pytest.raises(ValueError):
        transformer.transform(X_test)


def test_nan_handling():
    """Test handling of NaN values."""
    X = pd.DataFrame({"text": ["Hello", None, "World"]})
    transformer = TextFeatures(variables=["text"], features=["char_count"])
    X_tr = transformer.fit_transform(X)

    # NaN should be filled with empty string, resulting in char_count of 0
    assert X_tr["text_char_count"].tolist() == [5, 0, 5]


def test_default_all_features():
    """Test extracting all features with default parameters."""
    X = pd.DataFrame({"text": ["Hello World!", "Python 123", "AI"]})
    transformer = TextFeatures(variables=["text"])
    X_tr = transformer.fit_transform(X)

    # Spot check a few features to ensure they were added and computed
    assert X_tr["text_char_count"].tolist() == [11, 9, 2]
    assert X_tr["text_word_count"].tolist() == [2, 2, 1]
    assert X_tr["text_digit_count"].tolist() == [0, 3, 0]


def test_specific_features():
    """Test extracting specific features only."""
    X = pd.DataFrame({"text": ["Hello", "World"]})
    transformer = TextFeatures(
        variables=["text"], features=["char_count", "word_count"]
    )
    X_tr = transformer.fit_transform(X)

    # Check only specified features are extracted
    assert X_tr.columns.tolist() == ["text", "text_char_count", "text_word_count"]


def test_specific_variables():
    """Test extracting features from specific variables only."""
    X = pd.DataFrame(
        {"text1": ["Hello", "World"], "text2": ["Foo", "Bar"], "numeric": [1, 2]}
    )
    transformer = TextFeatures(variables=["text1"], features=["char_count"])
    X_tr = transformer.fit_transform(X)

    # Only text1 should have features extracted
    assert X_tr.columns.tolist() == ["text1", "text2", "numeric", "text1_char_count"]


def test_drop_original():
    """Test drop_original parameter."""
    X = pd.DataFrame({"text": ["Hello", "World"], "other": [1, 2]})
    transformer = TextFeatures(
        variables=["text"], features=["char_count"], drop_original=True
    )
    X_tr = transformer.fit_transform(X)

    assert X_tr.columns.tolist() == ["other", "text_char_count"]


def test_string_variable_input():
    """Test that passing a single string variable works (auto-converted to list)."""
    X = pd.DataFrame({"text": ["Hello", "World"], "other": ["A", "B"]})
    transformer = TextFeatures(variables="text", features=["char_count"])
    X_tr = transformer.fit_transform(X)

    assert transformer.variables_ == ["text"]
    assert X_tr.columns.tolist() == ["text", "other", "text_char_count"]
    assert X_tr["text_char_count"].tolist() == [5, 5]


def test_multiple_text_columns():
    """Test extracting features from multiple text columns."""
    X = pd.DataFrame({"a": ["Hello", "World"], "b": ["Foo", "Bar"]})
    transformer = TextFeatures(
        variables=["a", "b"], features=["char_count", "word_count"]
    )
    X_tr = transformer.fit_transform(X)

    assert X_tr.columns.tolist() == [
        "a",
        "b",
        "a_char_count",
        "a_word_count",
        "b_char_count",
        "b_word_count",
    ]


# ==============================================================================
# TRANSFORM TESTS - INDIVIDUAL FEATURES
# ==============================================================================


def test_whitespace_features():
    """Test whitespace_features."""
    X = pd.DataFrame(
        {
            "text": [
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
        }
    )
    transformer = TextFeatures(
        variables=["text"], features=["whitespace_count", "whitespace_ratio"]
    )
    X_tr = transformer.fit_transform(X)
    assert X_tr["text_whitespace_count"].tolist() == [
        1,
        0,
        0,
        1,
        3,
        2,
        0,
        0,
        0,
        2,
        0,
        1,
        1,
        0,
        3,
        5,
        1,
        1,
        1,
        10,
    ]
    assert X_tr["text_whitespace_ratio"].tolist() == [
        0.08333333333333333,
        0.0,
        0.0,
        0.1111111111111111,
        1.0,
        0.2,
        0.0,
        0.0,
        0.0,
        0.25,
        0.0,
        0.16666666666666666,
        0.07692307692307693,
        0.0,
        0.15789473684210525,
        0.2631578947368421,
        0.08333333333333333,
        0.07692307692307693,
        0.058823529411764705,
        0.09900990099009901,
    ]


def test_digit_features():
    """Test digit_features."""
    X = pd.DataFrame(
        {
            "text": [
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
        }
    )
    transformer = TextFeatures(
        variables=["text"], features=["digit_count", "digit_ratio", "has_digits"]
    )
    X_tr = transformer.fit_transform(X)
    assert X_tr["text_digit_count"].tolist() == [
        0,
        0,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        4,
        0,
        0,
        5,
        0,
        0,
        0,
        0,
    ]
    assert X_tr["text_digit_ratio"].tolist() == [
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.3333333333333333,
        0.0,
        0.0,
        0.35714285714285715,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    assert X_tr["text_has_digits"].tolist() == [
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
    ]


def test_uppercase_features():
    """Test uppercase_features."""
    X = pd.DataFrame(
        {
            "text": [
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
        }
    )
    transformer = TextFeatures(
        variables=["text"],
        features=[
            "uppercase_count",
            "uppercase_ratio",
            "has_uppercase",
            "starts_with_uppercase",
        ],
    )
    X_tr = transformer.fit_transform(X)
    assert X_tr["text_uppercase_count"].tolist() == [
        2,
        5,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        3,
        3,
        1,
        2,
        0,
        0,
        2,
        2,
        2,
        2,
        1,
    ]
    assert X_tr["text_uppercase_ratio"].tolist() == [
        0.18181818181818182,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.5,
        0.6,
        0.2,
        0.16666666666666666,
        0.0,
        0.0,
        0.14285714285714285,
        0.18181818181818182,
        0.16666666666666666,
        0.125,
        0.01098901098901099,
    ]
    assert X_tr["text_has_uppercase"].tolist() == [
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
    ]
    assert X_tr["text_starts_with_uppercase"].tolist() == [
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
    ]


def test_punctuation_features():
    """Test punctuation_features."""
    X = pd.DataFrame(
        {
            "text": [
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
        }
    )
    transformer = TextFeatures(
        variables=["text"], features=["special_char_count", "ends_with_punctuation"]
    )
    X_tr = transformer.fit_transform(X)
    assert X_tr["text_special_char_count"].tolist() == [
        1,
        0,
        0,
        4,
        0,
        0,
        3,
        0,
        0,
        3,
        0,
        3,
        4,
        3,
        3,
        5,
        1,
        2,
        6,
        1,
    ]
    assert X_tr["text_ends_with_punctuation"].tolist() == [
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
    ]


def test_word_features():
    """Test word_features."""
    X = pd.DataFrame(
        {
            "text": [
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
        }
    )
    transformer = TextFeatures(
        variables=["text"],
        features=[
            "word_count",
            "unique_word_count",
            "lexical_diversity",
            "avg_word_length",
        ],
    )
    X_tr = transformer.fit_transform(X)
    assert X_tr["text_word_count"].tolist() == [
        2,
        1,
        1,
        2,
        0,
        1,
        1,
        0,
        0,
        3,
        1,
        2,
        2,
        1,
        4,
        6,
        2,
        2,
        2,
        11,
    ]
    assert X_tr["text_unique_word_count"].tolist() == [
        2,
        1,
        1,
        2,
        0,
        1,
        1,
        0,
        0,
        3,
        1,
        2,
        2,
        1,
        4,
        6,
        2,
        2,
        2,
        11,
    ]
    assert X_tr["text_lexical_diversity"].tolist() == [
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    ]
    assert X_tr["text_avg_word_length"].tolist() == [
        6.0,
        5.0,
        5.0,
        4.5,
        0.0,
        8.0,
        6.0,
        0.0,
        0.0,
        2.6666666666666665,
        5.0,
        3.0,
        6.5,
        3.0,
        4.75,
        3.1666666666666665,
        6.0,
        6.5,
        8.5,
        9.181818181818182,
    ]


def test_basic_features():
    """Test basic_features."""
    X = pd.DataFrame(
        {
            "text": [
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
        }
    )
    transformer = TextFeatures(
        variables=["text"],
        features=[
            "char_count",
            "sentence_count",
            "letter_count",
            "lowercase_count",
            "is_empty",
        ],
    )
    X_tr = transformer.fit_transform(X)
    assert X_tr["text_char_count"].tolist() == [
        11,
        5,
        5,
        8,
        0,
        8,
        6,
        0,
        0,
        6,
        5,
        5,
        12,
        3,
        16,
        14,
        11,
        12,
        16,
        91,
    ]
    assert X_tr["text_sentence_count"].tolist() == [
        1,
        0,
        0,
        4,
        0,
        0,
        1,
        0,
        0,
        3,
        0,
        1,
        1,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
    ]
    assert X_tr["text_letter_count"].tolist() == [
        10,
        5,
        0,
        4,
        0,
        8,
        3,
        0,
        0,
        3,
        5,
        2,
        4,
        0,
        13,
        4,
        10,
        10,
        10,
        90,
    ]
    assert X_tr["text_lowercase_count"].tolist() == [
        8,
        0,
        0,
        4,
        0,
        8,
        3,
        0,
        0,
        0,
        2,
        1,
        2,
        0,
        13,
        2,
        8,
        8,
        8,
        89,
    ]
    assert X_tr["text_is_empty"].tolist() == [
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]


# ==============================================================================
# OTHER METHOD TESTS
# ==============================================================================


def test_get_feature_names_out():
    """Test get_feature_names_out returns correct names."""
    X = pd.DataFrame({"text": ["Hello"], "other": [1]})
    transformer = TextFeatures(
        variables=["text"], features=["char_count", "word_count"]
    )
    transformer.fit(X)

    feature_names = transformer.get_feature_names_out()
    expected_features = ["text", "other", "text_char_count", "text_word_count"]
    assert feature_names == expected_features


def test_get_feature_names_out_with_drop():
    """Test get_feature_names_out with drop_original=True."""
    X = pd.DataFrame({"text": ["Hello"], "other": [1]})
    transformer = TextFeatures(
        variables=["text"], features=["char_count"], drop_original=True
    )
    transformer.fit(X)

    feature_names = transformer.get_feature_names_out()
    expected_features = ["other", "text_char_count"]
    assert feature_names == expected_features
