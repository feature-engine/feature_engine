import pandas as pd
import pytest

from feature_engine.text import TextFeatures


class TestTextFeatures:
    """Test cases for TextFeatures transformer."""

    def test_default_all_features(self):
        """Test extracting all features with default parameters."""
        X = pd.DataFrame({"text": ["Hello World!", "Python 123", "AI"]})
        transformer = TextFeatures()
        X_tr = transformer.fit_transform(X)

        # Check that new columns were added
        assert "text_char_count" in X_tr.columns
        assert "text_word_count" in X_tr.columns
        assert "text_digit_count" in X_tr.columns

        # Check char_count
        assert X_tr["text_char_count"].tolist() == [12, 10, 2]

        # Check word_count
        assert X_tr["text_word_count"].tolist() == [2, 2, 1]

        # Check digit_count
        assert X_tr["text_digit_count"].tolist() == [0, 3, 0]

    def test_specific_features(self):
        """Test extracting specific features only."""
        X = pd.DataFrame({"text": ["Hello", "World"]})
        transformer = TextFeatures(features=["char_count", "word_count"])
        X_tr = transformer.fit_transform(X)

        # Check only specified features are extracted
        assert "text_char_count" in X_tr.columns
        assert "text_word_count" in X_tr.columns
        assert "text_digit_count" not in X_tr.columns
        assert "text_uppercase_count" not in X_tr.columns

    def test_specific_variables(self):
        """Test extracting features from specific variables only."""
        X = pd.DataFrame(
            {"text1": ["Hello", "World"], "text2": ["Foo", "Bar"], "numeric": [1, 2]}
        )
        transformer = TextFeatures(variables=["text1"], features=["char_count"])
        X_tr = transformer.fit_transform(X)

        # Only text1 should have features extracted
        assert "text1_char_count" in X_tr.columns
        assert "text2_char_count" not in X_tr.columns

    def test_drop_original(self):
        """Test drop_original parameter."""
        X = pd.DataFrame({"text": ["Hello", "World"], "other": [1, 2]})
        transformer = TextFeatures(features=["char_count"], drop_original=True)
        X_tr = transformer.fit_transform(X)

        assert "text" not in X_tr.columns
        assert "text_char_count" in X_tr.columns
        assert "other" in X_tr.columns

    def test_empty_string_handling(self):
        """Test handling of empty strings."""
        X = pd.DataFrame({"text": ["", "Hello", ""]})
        transformer = TextFeatures(features=["char_count", "word_count", "is_empty"])
        X_tr = transformer.fit_transform(X)

        assert X_tr["text_char_count"].tolist() == [0, 5, 0]
        assert X_tr["text_is_empty"].tolist() == [1, 0, 1]

    def test_nan_handling(self):
        """Test handling of NaN values."""
        X = pd.DataFrame({"text": ["Hello", None, "World"]})
        transformer = TextFeatures(features=["char_count"])
        X_tr = transformer.fit_transform(X)

        # NaN should be filled with empty string, resulting in char_count of 0
        assert X_tr["text_char_count"].tolist() == [5, 0, 5]

    def test_uppercase_features(self):
        """Test uppercase-related features."""
        X = pd.DataFrame({"text": ["HELLO", "hello", "HeLLo"]})
        transformer = TextFeatures(
            features=["uppercase_count", "has_uppercase", "starts_with_uppercase"]
        )
        X_tr = transformer.fit_transform(X)

        assert X_tr["text_uppercase_count"].tolist() == [5, 0, 3]
        assert X_tr["text_has_uppercase"].tolist() == [1, 0, 1]
        assert X_tr["text_starts_with_uppercase"].tolist() == [1, 0, 1]

    def test_sentence_count(self):
        """Test sentence counting."""
        X = pd.DataFrame({"text": ["Hello. World!", "One sentence", "A? B! C."]})
        transformer = TextFeatures(features=["sentence_count"])
        X_tr = transformer.fit_transform(X)

        assert X_tr["text_sentence_count"].tolist() == [2, 0, 3]

    def test_unique_word_features(self):
        """Test unique word features."""
        X = pd.DataFrame({"text": ["the the the", "a b c", "x"]})
        transformer = TextFeatures(features=["unique_word_count", "unique_word_ratio"])
        X_tr = transformer.fit_transform(X)

        assert X_tr["text_unique_word_count"].tolist() == [1, 3, 1]
        assert X_tr["text_unique_word_ratio"].tolist() == [1 / 3, 1.0, 1.0]

    def test_invalid_feature_raises_error(self):
        """Test that invalid feature name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid features"):
            TextFeatures(features=["invalid_feature"])

    def test_invalid_variables_raises_error(self):
        """Test that invalid variables parameter raises ValueError."""
        with pytest.raises(ValueError, match="variables must be"):
            TextFeatures(variables=123)

    def test_missing_variable_raises_error(self):
        """Test that missing variable raises ValueError on fit."""
        X = pd.DataFrame({"text": ["Hello"]})
        transformer = TextFeatures(variables=["nonexistent"])
        with pytest.raises(ValueError, match="not present in the dataframe"):
            transformer.fit(X)

    def test_no_text_columns_raises_error(self):
        """Test that no text columns raises error when variables=None."""
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        transformer = TextFeatures()
        with pytest.raises(ValueError, match="No object/string columns found"):
            transformer.fit(X)

    def test_fit_stores_attributes(self):
        """Test that fit stores expected attributes."""
        X = pd.DataFrame({"text": ["Hello"]})
        transformer = TextFeatures()
        transformer.fit(X)

        assert hasattr(transformer, "variables_")
        assert hasattr(transformer, "features_")
        assert hasattr(transformer, "feature_names_in_")
        assert hasattr(transformer, "n_features_in_")

    def test_get_feature_names_out(self):
        """Test get_feature_names_out returns correct names."""
        X = pd.DataFrame({"text": ["Hello"], "other": [1]})
        transformer = TextFeatures(features=["char_count", "word_count"])
        transformer.fit(X)

        feature_names = transformer.get_feature_names_out()
        assert "text" in feature_names
        assert "other" in feature_names
        assert "text_char_count" in feature_names
        assert "text_word_count" in feature_names

    def test_get_feature_names_out_with_drop(self):
        """Test get_feature_names_out with drop_original=True."""
        X = pd.DataFrame({"text": ["Hello"], "other": [1]})
        transformer = TextFeatures(features=["char_count"], drop_original=True)
        transformer.fit(X)

        feature_names = transformer.get_feature_names_out()
        assert "text" not in feature_names
        assert "other" in feature_names
        assert "text_char_count" in feature_names

    def test_string_variable_input(self):
        """Test that passing a single string variable works (auto-converted to list)."""
        X = pd.DataFrame({"text": ["Hello", "World"], "other": ["A", "B"]})
        transformer = TextFeatures(variables="text", features=["char_count"])
        X_tr = transformer.fit_transform(X)

        assert "text_char_count" in X_tr.columns
        assert "other_char_count" not in X_tr.columns
        assert X_tr["text_char_count"].tolist() == [5, 5]

    def test_invalid_features_type_raises_error(self):
        """Test that invalid features type raises ValueError."""
        with pytest.raises(ValueError, match="features must be"):
            TextFeatures(features="char_count")

    def test_multiple_text_columns(self):
        """Test extracting features from multiple text columns."""
        X = pd.DataFrame({"a": ["Hello", "World"], "b": ["Foo", "Bar"]})
        transformer = TextFeatures(features=["char_count", "word_count"])
        X_tr = transformer.fit_transform(X)

        assert "a_char_count" in X_tr.columns
        assert "b_char_count" in X_tr.columns
        assert "a_word_count" in X_tr.columns
        assert "b_word_count" in X_tr.columns

    def test_transform_on_new_data(self):
        """Test transform works on new data after fit."""
        X_train = pd.DataFrame({"text": ["Hello World", "Foo Bar"]})
        X_test = pd.DataFrame({"text": ["New Data", "Test 123"]})

        transformer = TextFeatures(features=["char_count", "has_digits"])
        transformer.fit(X_train)
        X_tr = transformer.transform(X_test)

        assert X_tr["text_char_count"].tolist() == [8, 8]
        assert X_tr["text_has_digits"].tolist() == [0, 1]

    def test_punctuation_features(self):
        """Test punctuation-related features."""
        X = pd.DataFrame({"text": ["Hello.", "World", "Hi!"]})
        transformer = TextFeatures(
            features=["ends_with_punctuation", "special_char_count"]
        )
        X_tr = transformer.fit_transform(X)

        assert X_tr["text_ends_with_punctuation"].tolist() == [1, 0, 1]
        assert X_tr["text_special_char_count"].tolist() == [1, 0, 1]

    def test_ratio_features(self):
        """Test ratio features with known values."""
        X = pd.DataFrame({"text": ["AB12", "abcd"]})
        transformer = TextFeatures(
            features=["digit_ratio", "uppercase_ratio", "whitespace_ratio"]
        )
        X_tr = transformer.fit_transform(X)

        assert X_tr["text_digit_ratio"].tolist() == [0.5, 0.0]
        assert X_tr["text_uppercase_ratio"].tolist() == [0.5, 0.0]
        assert X_tr["text_whitespace_ratio"].tolist() == [0.0, 0.0]

    def test_avg_word_length(self):
        """Test average word length feature."""
        X = pd.DataFrame({"text": ["ab cd", "a"]})
        transformer = TextFeatures(features=["avg_word_length"])
        X_tr = transformer.fit_transform(X)

        assert X_tr["text_avg_word_length"].tolist() == [2.0, 1.0]

    def test_lowercase_count(self):
        """Test lowercase count feature."""
        X = pd.DataFrame({"text": ["Hello", "WORLD"]})
        transformer = TextFeatures(features=["lowercase_count"])
        X_tr = transformer.fit_transform(X)

        assert X_tr["text_lowercase_count"].tolist() == [4, 0]

    def test_variables_list_non_strings_raises_error(self):
        """Test that a list of non-string variables raises ValueError."""
        with pytest.raises(ValueError, match="variables must be"):
            TextFeatures(variables=[1, 2])

    def test_features_list_non_strings_raises_error(self):
        """Test that a list of non-string features raises ValueError."""
        with pytest.raises(ValueError, match="features must be"):
            TextFeatures(features=[1, 2])

    def test_more_tags(self):
        """Test _more_tags returns expected tags."""
        transformer = TextFeatures()
        tags = transformer._more_tags()
        assert tags["allow_nan"] is True
        assert tags["variables"] == "categorical"

    def test_sklearn_tags(self):
        """Test __sklearn_tags__ returns expected tags."""
        import sklearn

        if hasattr(sklearn, "__version__") and tuple(
            int(x) for x in sklearn.__version__.split(".")[:2]
        ) >= (1, 6):
            transformer = TextFeatures()
            tags = transformer.__sklearn_tags__()
            assert tags.input_tags.allow_nan is True
