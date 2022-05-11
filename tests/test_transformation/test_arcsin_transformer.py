import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import ArcsinTransformer


def test_fit_raises_error_if_na_in_df(df_na):
    # test case 2: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = ArcsinTransformer()
        transformer.fit(df_na)


def test_transform_raises_error_if_na_in_df(df_vartypes, df_na):
    # test case 3: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = ArcsinTransformer()
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_error_if_df_contains_outside_range_value(df_vartypes):
    # test error when data contains value outside range [-1, +1]
    df_neg = df_vartypes.copy()
    df_neg.loc[1, "Age"] = 2

    # test case 4: when variable contains value outside range, fit
    with pytest.raises(ValueError):
        transformer = ArcsinTransformer()
        transformer.fit(df_neg)

    # test case 5: when variable contains value outside range, transform
    with pytest.raises(ValueError):
        transformer = ArcsinTransformer()
        transformer.fit(df_vartypes)
        transformer.transform(df_neg)


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = ArcsinTransformer()
        transformer.transform(df_vartypes)
