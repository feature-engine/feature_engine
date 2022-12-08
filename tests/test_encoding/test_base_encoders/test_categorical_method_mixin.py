import numpy as np
import pandas as pd
import pytest

from feature_engine.encoding.base_encoder import CategoricalMethodsMixin


class MockClassFit(CategoricalMethodsMixin):
    def __init__(self, missing_values):
        self.missing_values = missing_values
        self.variables = None
        self.ignore_format = False


def test_underscore_fit_method():
    input_df = pd.DataFrame(
        {
            "words": ["dog", "dig", "cat"],
            "animals": ["bird", "tiger", np.nan],
        }
    )

    enc = MockClassFit(missing_values="raise")
    with pytest.raises(ValueError):
        enc._fit(input_df)

    enc = MockClassFit(missing_values="ignore")
    variables_ = enc._fit(input_df)
    assert variables_ == ["words", "animals"]


class MockClass(CategoricalMethodsMixin):
    def __init__(self, unseen=None):
        self.encoder_dict_ = {"words": {"dog": 1, "dig": 0.66, "cat": 0}}
        self.n_features_in_ = 1
        self.feature_names_in_ = ["words"]
        self.variables_ = ["words"]
        self.missing_values = "raise"
        self.unseen = unseen
        self._unseen = -1
        self.ignore_format = True
        self.variables = None

    def fit(self):
        return self

def test_categorical_methods_mixin_transform_no_unseen():
    input_df = pd.DataFrame({"words": ["dog", "dig", "cat"]})
    output_df = pd.DataFrame({"words": [1, 0.66, 0]})
    enc = MockClass()
    pd.testing.assert_frame_equal(enc.transform(input_df), output_df)


def test_categorical_methods_mixin_transform_ignore_unseen():
    input_df = pd.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pd.DataFrame({"words": [1, 0.66, np.nan]})
    enc = MockClass(unseen="ignore")
    pd.testing.assert_frame_equal(enc.transform(input_df), output_df)


def test_categorical_methods_mixin_transform_raise_unseen():
    input_df = pd.DataFrame({"words": ["dog", "dig", "bird"]})
    enc = MockClass(unseen="raise")
    with pytest.raises(ValueError):
        enc.transform(input_df)


def test_categorical_methods_mixin_transform_encode_unseen():
    input_df = pd.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pd.DataFrame({"words": [1, 0.66, -1]})
    enc = MockClass(unseen="encode")
    pd.testing.assert_frame_equal(enc.transform(input_df), output_df)


def test_categorical_methods_mixin_raises_error_when_nan_introduced():
    input_df = pd.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pd.DataFrame({"words": [1, 0.66, np.nan]})
    enc = MockClass(unseen="raise")
    msg = "During the encoding, NaN values were introduced in the feature(s) words."

    with pytest.raises(ValueError) as record:
        enc._check_nan_values_after_transformation(output_df)
    assert str(record.value) == msg

    with pytest.raises(ValueError) as record:
        enc.transform(input_df)
    assert str(record.value) == msg


def test_categorical_methods_mixin_raises_warning_when_nan_introduced():
    input_df = pd.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pd.DataFrame({"words": [1, 0.66, np.nan]})
    enc = MockClass(unseen="ignore")
    msg = "During the encoding, NaN values were introduced in the feature(s) words."

    with pytest.warns(UserWarning) as record:
        enc.transform(input_df)
    assert record[0].message.args[0] == msg

    with pytest.warns(UserWarning) as record:
        enc._check_nan_values_after_transformation(output_df)
    assert record[0].message.args[0] == msg


def test_categorical_methods_mixin_inverse_transform_no_unseen_categories():
    input_df = pd.DataFrame({"words": ["dog", "dig", "cat"]})
    output_df = pd.DataFrame({"words": [1, 0.66, 0]})

    # when no unseen categories
    enc = MockClass()
    pd.testing.assert_frame_equal(enc.inverse_transform(output_df), input_df)


def test_categorical_methods_mixin_inverse_transform_when_ignore_unseen():
    input_df = pd.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pd.DataFrame({"words": [1, 0.66, np.nan]})
    inverse_df = pd.DataFrame({"words": ["dog", "dig", np.nan]})

    # when no unseen categories
    enc = MockClass(unseen="ignore")
    pd.testing.assert_frame_equal(enc.transform(input_df), output_df)
    pd.testing.assert_frame_equal(enc.inverse_transform(output_df), inverse_df)


def test_categorical_methods_mixin_transform_raises_error_when_df_has_nan():
    input_df = pd.DataFrame({"words": ["dog", "dig", "cat", np.nan]})
    enc = MockClass()
    with pytest.raises(ValueError):
        enc.transform(input_df)


def test_categorical_methods_mixin_transform_ignores_nan_in_df_to_transform():
    input_df = pd.DataFrame({"words": ["dog", "dig", "cat", np.nan]})
    output_df = pd.DataFrame({"words": [1, 0.66, 0, np.nan]})
    enc = MockClass()
    enc.missing_values = "ignore"
    pd.testing.assert_frame_equal(enc.transform(input_df), output_df)
