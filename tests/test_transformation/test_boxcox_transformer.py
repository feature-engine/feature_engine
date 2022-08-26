import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import BoxCoxTransformer


def test_automatically_finds_variables(df_vartypes):
    # test case 1: automatically select variables
    transformer = BoxCoxTransformer(variables=None)
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = [9.78731, 10.1666, 9.40189, 9.0099]
    transf_df["Marks"] = [-0.101687, -0.207092, -0.316843, -0.431788]

    # test init params
    assert transformer.variables is None
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 5
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)

    # test inverse_transform
    Xit = transformer.inverse_transform(X)

    # convert numbers to original format.
    Xit["Age"] = Xit["Age"].round().astype("int64")
    Xit["Marks"] = Xit["Marks"].round(1)

    # test
    pd.testing.assert_frame_equal(Xit, df_vartypes)


def test_fit_raises_error_if_df_contains_na(df_na):
    # test case 2: when dataset contains na, fit method
    transformer = BoxCoxTransformer()
    with pytest.raises(ValueError):
        transformer.fit(df_na)


def test_transform_raises_error_if_df_contains_na(df_vartypes, df_na):
    # test case 3: when dataset contains na, transform method
    transformer = BoxCoxTransformer()
    transformer.fit(df_vartypes)
    with pytest.raises(ValueError):
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_error_if_df_contains_negative_values(df_vartypes):
    # test error when data contains negative values
    df_neg = df_vartypes.copy()
    df_neg.loc[1, "Age"] = -1

    # test case 4: when variable contains negative value, fit
    transformer = BoxCoxTransformer()
    with pytest.raises(ValueError):
        transformer.fit(df_neg)

    # test case 5: when variable contains negative value, transform
    transformer = BoxCoxTransformer()
    transformer.fit(df_vartypes)
    with pytest.raises(ValueError):
        transformer.transform(df_neg)


def test_non_fitted_error(df_vartypes):
    transformer = BoxCoxTransformer()
    with pytest.raises(NotFittedError):
        transformer.transform(df_vartypes)
