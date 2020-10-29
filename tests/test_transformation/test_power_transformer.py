import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from feature_engine.transformation import PowerTransformer


def test_defo_params_plus_automatically_find_variables(df_vartypes):
    # test case 1: automatically select variables
    transformer = PowerTransformer(variables=None)
    X = transformer.fit_transform(df_vartypes)

    # expected output
    transf_df = df_vartypes.copy()
    transf_df["Age"] = [4.47214, 4.58258, 4.3589, 4.24264]
    transf_df["Marks"] = [0.948683, 0.894427, 0.83666, 0.774597]

    # test init params
    assert transformer.exp == 0.5
    assert transformer.variables == ["Age", "Marks"]
    # test fit attr
    assert transformer.input_shape_ == (4, 5)
    # test transform output
    pd.testing.assert_frame_equal(X, transf_df)


def test_error_if_exp_value_not_allowed():
    with pytest.raises(ValueError):
        PowerTransformer(exp="other")


def test_fit_raises_error_if_na_in_df(df_na):
    # test case 2: when dataset contains na, fit method
    with pytest.raises(ValueError):
        transformer = PowerTransformer()
        transformer.fit(df_na)


def test_transform_raises_error_if_na_in_df(df_vartypes, df_na):
    # test case 3: when dataset contains na, transform method
    with pytest.raises(ValueError):
        transformer = PowerTransformer()
        transformer.fit(df_vartypes)
        transformer.transform(df_na[["Name", "City", "Age", "Marks", "dob"]])


def test_non_fitted_error(df_vartypes):
    with pytest.raises(NotFittedError):
        transformer = PowerTransformer()
        transformer.transform(df_vartypes)
