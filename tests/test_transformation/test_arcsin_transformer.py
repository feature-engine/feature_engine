import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import ArcsinTransformer


def test_transform_and_invert(df_vartypes):
    transformer = ArcsinTransformer(variables=["Marks"])
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Marks"] = [1.24905, 1.10715, 0.99116, 0.88607]

    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)

    # test inverse_transform
    Xit = transformer.inverse_transform(X)

    # convert numbers to original format.
    Xit["Marks"] = Xit["Marks"].round(1)

    # test
    pd.testing.assert_frame_equal(Xit, df_vartypes)


def test_fit_raises_error_if_na_in_df(df_na):
    # test case 2: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = ArcsinTransformer()
        transformer.fit(df_na)


def test_transform_raises_error_if_na_in_df(df_vartypes, df_na):
    # test case 3: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = ArcsinTransformer(variables=["Marks"])
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_error_if_df_contains_outside_range_value(df_vartypes):
    # test error when data contains value outside range [0, +1]
    df_neg = df_vartypes.copy()
    df_neg.loc[1, "Marks"] = 2

    transformer = ArcsinTransformer()

    # test case 4: when variable contains value outside range, fit
    with pytest.raises(ValueError):
        transformer.fit(df_neg)

    # test case 5: when variable contains value outside range, transform
    with pytest.raises(ValueError):
        transformer.transform(df_neg)


def test_non_fitted_error(df_vartypes):
    transformer = ArcsinTransformer()
    with pytest.raises(NotFittedError):
        transformer.transform(df_vartypes)
