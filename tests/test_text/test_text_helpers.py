from datetime import datetime

import narwhals as nw
import pandas as pd

from feature_engine.dataframe_checks import check_X
from feature_engine.text._text_helpers import add_text_features, find_non_text_variables
from tests.backend_helpers import frame_to_dict


def test_find_non_text_variables(make_df):
    X = make_df(
        {
            "text": ["a", "b"],
            "category": ["a", "b"],
            "int": [1, 2],
            "float": [0.5, 1.5],
            "date": [datetime(2020, 1, 1), datetime(2020, 1, 2)],
        }
    )
    X = nw.from_native(X).with_columns(nw.col("category").cast(nw.Categorical))
    non_text = find_non_text_variables(check_X(X.to_native()), X.columns)
    assert non_text == ["int", "float", "date"]


def test_find_non_text_variables_with_pandas_categories():
    X = pd.DataFrame(
        {
            "text": pd.Series(["a", "b"], dtype="category"),
            "number": pd.Series([1, 2], dtype="category"),
        }
    )
    assert find_non_text_variables(check_X(X), ["text", "number"]) == ["number"]


def test_add_text_features_fills_missing_values(make_df):
    X = check_X(make_df({"text": ["Hi there", None], "other": [1, 2]}))
    Xt = add_text_features(X, ["text"], ["word_count", "is_empty"], fill_missing=True)

    assert isinstance(Xt, type(X))
    assert frame_to_dict(nw.to_native(Xt, pass_through=True)) == {
        "text": ["Hi there", ""],
        "other": [1, 2],
        "text_word_count": [2, 0],
        "text_is_empty": [0, 1],
    }


def test_add_text_features_without_filling_missing_values(make_df):
    X = check_X(make_df({"a": ["Hi there", "Bye"], "b": ["x", "y z"]}))
    Xt = add_text_features(X, ["b", "a"], ["word_count"], fill_missing=False)

    assert isinstance(Xt, type(X))
    assert frame_to_dict(nw.to_native(Xt, pass_through=True)) == {
        "a": ["Hi there", "Bye"],
        "b": ["x", "y z"],
        "b_word_count": [1, 2],
        "a_word_count": [2, 1],
    }
