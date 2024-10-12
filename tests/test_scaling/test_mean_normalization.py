import re

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.scaling import MeanNormalizationScaler


def test_transforming_int_vars():
    # input test case
    df = pd.DataFrame(
        {
            "var1": [1.0, 2.0, 3.0],
            "var2": [4.0, 5.0, 3.0],
            "var3": [40.0, 20.0, 30.0],
        }
    )
    # expected output
    expected_df = pd.DataFrame(
        {
            "var1": [-0.5, 0.0, 0.5],
            "var2": [0, 0.5, -0.5],
            "var3": [0.5, -0.5, 0.0],
        }
    )

    transformer = MeanNormalizationScaler(variables=None)
    X = transformer.fit_transform(df)

    pd.testing.assert_frame_equal(X, expected_df)

    # test inverse_transform
    Xit = transformer.inverse_transform(X)

    pd.testing.assert_frame_equal(Xit, df)


def test_mean_normalization_plus_automatically_find_variables(df_vartypes):
    # test case 1: automatically select variables
    transformer = MeanNormalizationScaler(variables=None)
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = [0.16666, 0.5, -0.16666, -0.5]
    transf_df["Marks"] = [0.49999, 0.16666, -0.16666, -0.5]

    # test init params
    assert transformer.variables is None
    # test fit attr
    assert transformer.variables_ == ["Age", "Marks"]
    assert transformer.n_features_in_ == 5
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df, rtol=10e-3)

    # test inverse_transform
    Xit = transformer.inverse_transform(X)

    # convert numbers to original format.
    Xit["Age"] = Xit["Age"].round().astype("int64")
    Xit["Marks"] = Xit["Marks"].round(1)

    # test
    pd.testing.assert_frame_equal(Xit, df_vartypes, rtol=10e-3)


def test_mean_normalization_plus_user_passes_var_list(df_vartypes):
    # test case 2: user passes variables
    transformer = MeanNormalizationScaler(variables="Age")
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = [0.16666, 0.5, -0.16666, -0.5]

    # test init params
    assert transformer.variables == "Age"
    # test fit attr
    assert transformer.variables_ == ["Age"]
    assert transformer.n_features_in_ == 5
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df, rtol=10e-3)

    # test inverse_transform
    Xit = transformer.inverse_transform(X)

    # convert numbers to original format.
    Xit["Age"] = Xit["Age"].round().astype("int64")

    # test
    pd.testing.assert_frame_equal(Xit, df_vartypes, rtol=10e-3)


def test_fit_raises_error_if_na_in_df(df_na):
    # test case 3: when dataset contains na, fit method
    transformer = MeanNormalizationScaler()
    with pytest.raises(ValueError):
        transformer.fit(df_na)


def test_transform_raises_error_if_na_in_df(df_vartypes, df_na):
    # test case 4: when dataset contains na, transform method
    transformer = MeanNormalizationScaler()
    transformer.fit(df_vartypes)
    with pytest.raises(ValueError):
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_non_fitted_error(df_vartypes):
    transformer = MeanNormalizationScaler()
    with pytest.raises(NotFittedError):
        transformer.transform(df_vartypes)


def test_constant_columns_error():
    # input test case
    df = pd.DataFrame(
        {
            "var1": [1.0, 2.0, 3.0],
            "var2": [4.0, 5.0, 3.0],
            "var3": [7.0, 7.0, 7.0],
        }
    )

    transformer = MeanNormalizationScaler()
    with pytest.raises(ValueError, match=re.escape("Division by zero is not allowed")):
        transformer.fit(df)
