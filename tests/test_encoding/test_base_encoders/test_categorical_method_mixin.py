import re

import narwhals as nw
import numpy as np
import pandas as pd
import pytest

from feature_engine.encoding.base_encoder import CategoricalMethodsMixin


class MockClassFit(CategoricalMethodsMixin):
    def __init__(self, missing_values="raise", ignore_format=False, return_empty=False):
        self.missing_values = missing_values
        self.variables = None
        self.return_empty = return_empty
        self.ignore_format = ignore_format


def test_underscore_check_na_method():
    input_df = pd.DataFrame(
        {
            "words": ["dog", "dig", "cat"],
            "animals": ["bird", "tiger", np.nan],
        }
    )
    variables = ["words", "animals"]

    enc = MockClassFit(missing_values="raise")
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        enc._check_na(input_df, variables)


def test_check_or_select_variables():
    input_df = pd.DataFrame(
        {
            "words": ["dog", "dig", "cat"],
            "animals": [1, 2, np.nan],
        }
    )

    enc = MockClassFit(ignore_format=False)
    assert enc._check_or_select_variables(input_df) == ["words"]

    enc = MockClassFit(ignore_format=True)
    assert enc._check_or_select_variables(input_df) == ["words", "animals"]


def test_get_feature_names_in():
    input_df = pd.DataFrame(
        {
            "words": ["dog", "dig", "cat"],
            "animals": [1, 2, np.nan],
        }
    )
    enc = MockClassFit()
    enc._get_feature_names_in(input_df)
    assert enc.feature_names_in_ == ["words", "animals"]
    assert enc.n_features_in_ == 2


class MockClass(CategoricalMethodsMixin):
    def __init__(self, unseen=None, missing_values="raise"):
        self.encoder_dict_ = {"words": {"dog": 1, "dig": 0.66, "cat": 0}}
        self.n_features_in_ = 1
        self.feature_names_in_ = ["words"]
        self.variables_ = ["words"]
        self.missing_values = missing_values
        self.unseen = unseen
        self._unseen = -1

    def fit(self):
        return self


def test_transform_no_unseen():
    input_df = pd.DataFrame({"words": ["dog", "dig", "cat"]})
    output_df = pd.DataFrame({"words": [1, 0.66, 0]})
    enc = MockClass()
    pd.testing.assert_frame_equal(enc.transform(input_df), output_df)


def test_transform_ignore_unseen():
    input_df = pd.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pd.DataFrame({"words": [1, 0.66, np.nan]})
    enc = MockClass(unseen="ignore")
    pd.testing.assert_frame_equal(enc.transform(input_df), output_df)


def test_transform_encode_unseen():
    input_df = pd.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pd.DataFrame({"words": [1, 0.66, -1]})
    enc = MockClass(unseen="encode")
    pd.testing.assert_frame_equal(enc.transform(input_df), output_df)


def test_raises_error_when_nan_introduced():
    input_df = pd.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pd.DataFrame({"words": [1, 0.66, np.nan]})
    enc = MockClass(unseen="raise")
    msg = "During the encoding, NaN values were introduced in the feature(s) words."

    with pytest.raises(ValueError, match=re.escape(msg)):
        enc._check_nan_values_after_transformation(nw.from_native(output_df))

    with pytest.raises(ValueError, match=re.escape(msg)):
        enc.transform(input_df)


def test_raises_warning_when_nan_introduced():
    input_df = pd.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pd.DataFrame({"words": [1, 0.66, np.nan]})
    enc = MockClass(unseen="ignore")
    msg = "During the encoding, NaN values were introduced in the feature(s) words."

    with pytest.warns(UserWarning, match=re.escape(msg)):
        enc.transform(input_df)

    with pytest.warns(UserWarning, match=re.escape(msg)):
        enc._check_nan_values_after_transformation(nw.from_native(output_df))


def test_transform_raises_error_when_df_has_nan():
    input_df = pd.DataFrame({"words": ["dog", "dig", "cat", np.nan]})
    enc = MockClass()
    msg = (
        "Some of the variables in the dataset contain NaN. Check and "
        "remove those before using this transformer or set the parameter "
        "`missing_values='ignore'` when initialising this transformer."
    )
    with pytest.raises(ValueError, match=re.escape(msg)):
        enc.transform(input_df)


def test_transform_ignores_nan_in_df_to_transform():
    input_df = pd.DataFrame({"words": ["dog", "dig", "cat", np.nan]})
    output_df = pd.DataFrame({"words": [1, 0.66, 0, np.nan]})
    enc = MockClass()
    enc.missing_values = "ignore"
    pd.testing.assert_frame_equal(enc.transform(input_df), output_df)


def test_inverse_transform_no_unseen_categories():
    input_df = pd.DataFrame({"words": ["dog", "dig", "cat"]})
    output_df = pd.DataFrame({"words": [1, 0.66, 0]})

    # when no unseen categories
    enc = MockClass()
    pd.testing.assert_frame_equal(enc.inverse_transform(output_df), input_df)


def test_inverse_transform_when_ignore_unseen():
    input_df = pd.DataFrame({"words": ["dog", "dig", "bird"]})
    output_df = pd.DataFrame({"words": [1, 0.66, np.nan]})
    inverse_df = pd.DataFrame({"words": ["dog", "dig", np.nan]})

    # when no unseen categories
    enc = MockClass(unseen="ignore")
    pd.testing.assert_frame_equal(enc.transform(input_df), output_df)
    pd.testing.assert_frame_equal(enc.inverse_transform(output_df), inverse_df)
