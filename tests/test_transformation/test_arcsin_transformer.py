import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import ArcsinTransformer


def test_transform_and_inverse_transform(df_vartypes):
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
    transformer = ArcsinTransformer(variables=["Marks"])
    with pytest.raises(ValueError):
        transformer.fit(df_na)


def test_transform_raises_error_if_na_in_df(df_vartypes, df_na):
    # test case 3: when dataset contains na, transform method
    transformer = ArcsinTransformer(variables=["Marks"])
    transformer.fit(df_vartypes)
    with pytest.raises(ValueError):
        transformer.transform(df_na[df_vartypes.columns])


def test_error_if_df_contains_outside_range_values(df_vartypes):
    # test error when data contains value outside range [0, +1]
    df_out_range = df_vartypes.copy()
    df_out_range.loc[1, "Marks"] = 2

    transformer = ArcsinTransformer(variables=["Marks"])
    # test case 4: when variable contains value outside range, fit
    with pytest.raises(ValueError):
        transformer.fit(df_out_range)

    # test case 5: when variable contains value outside range, transform
    transformer.fit(df_vartypes)
    with pytest.raises(ValueError):
        transformer.transform(df_out_range)

    # when selecting variables automatically and some are outside range
    transformer = ArcsinTransformer()
    with pytest.raises(ValueError):
        transformer.fit(df_vartypes)


def test_non_fitted_error(df_vartypes):
    transformer = ArcsinTransformer(variables="Marks")
    with pytest.raises(NotFittedError):
        transformer.transform(df_vartypes)
