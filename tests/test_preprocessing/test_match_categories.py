import warnings

import numpy as np
import pandas as pd
import pytest

from feature_engine.preprocessing import MatchCategories


def test_category_encoder_outputs_correct_dtype():
    df_str = pd.DataFrame({"col1": ["a", "b", "c"]})
    res_str = MatchCategories().fit(df_str).transform(df_str)
    assert res_str.dtypes["col1"] == "category"

    df_float = pd.DataFrame({"col1": [1.0, 2.0, 3.0]})
    tr = MatchCategories(variables=["col1"], ignore_format=True)
    res_float = tr.fit(df_float).transform(df_float)
    assert res_float.dtypes["col1"] == "category"

    df_obj = pd.DataFrame({"col1": ["a", None, -1.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_obj = MatchCategories(missing_values="ignore").fit(df_obj).transform(df_obj)
    assert res_obj.dtypes["col1"] == "category"

    df_categ = pd.DataFrame({"col1": pd.Categorical(pd.Series(["a", "b", "c"]))})
    res_categ = MatchCategories().fit(df_categ).transform(df_categ)
    assert res_categ.dtypes["col1"] == "category"


def test_category_encoder_handles_missing():
    df_no_nas = pd.DataFrame({"col1": ["a", "b", "c"]})
    df_nas = pd.DataFrame({"col1": ["a", "b", None]})
    df_new = pd.DataFrame({"col1": ["a", "b", "d"]})

    # check that it fails for missing values when using 'raise'
    tr = MatchCategories(missing_values="raise")
    with pytest.raises(ValueError):
        tr.fit(df_nas)

    tr.fit(df_no_nas)
    with pytest.raises(ValueError):
        tr.transform(df_nas)

    # check that it doens't fail for missing values when using 'ignore'
    tr = MatchCategories(missing_values="ignore").fit(df_nas)
    with pytest.warns(UserWarning):
        tr.transform(df_nas)

    # check that it doesn't fail at transforming new values when using 'ignore'
    tr = MatchCategories(missing_values="ignore").fit(df_no_nas)
    with pytest.warns(UserWarning):
        tr.transform(df_new)


def test_category_outputs_correct_results():
    df = pd.DataFrame({"col1": ["a", "b", "c"], "col2": [1.0, 2.0, 3.0]})
    res = MatchCategories(variables=["col1", "col2"], ignore_format=True).fit_transform(
        df
    )
    pd.testing.assert_frame_equal(df, res, check_dtype=False, check_categorical=False)

    df = pd.DataFrame({"col1": ["a", "b", "d"], "col2": [1.0, 2.0, np.nan]})
    res = MatchCategories(
        variables=["col1", "col2"], ignore_format=True, missing_values="ignore"
    ).fit_transform(df)
    pd.testing.assert_frame_equal(df, res, check_dtype=False, check_categorical=False)
