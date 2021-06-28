import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import ReciprocalTransformer


def test_automatically_find_variables(df_vartypes):
    # test case 1: automatically select variables
    transformer = ReciprocalTransformer(variables=None)
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = [0.05, 0.047619, 0.0526316, 0.0555556]
    transf_df["Marks"] = [1.11111, 1.25, 1.42857, 1.66667]

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


def test_fit_raises_error_if_na_in_df(df_na):
    # test case 2: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(df_na)


def test_transform_raises_error_if_na_in_df(df_vartypes, df_na):
    # test case 3: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_error_if_df_contains_0_as_value(df_vartypes):
    # test error when data contains value zero
    df_neg = df_vartypes.copy()
    df_neg.loc[1, "Age"] = 0

    # test case 4: when variable contains zero, fit
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(df_neg)

    # test case 5: when variable contains zero, transform
    with pytest.raises(ValueError):
        transformer = ReciprocalTransformer()
        transformer.fit(df_vartypes)
        transformer.transform(df_neg)


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = ReciprocalTransformer()
        transformer.transform(df_vartypes)
